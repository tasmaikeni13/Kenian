"""Data prep + loaders for CIFAR-100 and WikiText-103.

Both cache a preprocessed artifact in data/ on first use, then load quickly.
The pipeline is identical across optimizers; seeds control order and augmentation.
"""

from __future__ import annotations

import os
import numpy as np
import torch

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
HF = os.path.join(DATA, "hf_cache")


# ------------------------------------------------------------------ CIFAR-100
def _prepare_cifar():
    cache = os.path.join(DATA, "cifar100.npz")
    if os.path.exists(cache):
        return cache
    from datasets import load_dataset

    ds = load_dataset("uoft-cs/cifar100", cache_dir=HF)

    def to_arrays(split):
        imgs = np.stack([np.array(im) for im in split["img"]]).astype(
            np.uint8
        )  # N,32,32,3
        labels = np.array(split["fine_label"], dtype=np.int64)
        return imgs, labels

    xtr, ytr = to_arrays(ds["train"])
    xte, yte = to_arrays(ds["test"])
    np.savez(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    return cache


# CIFAR-100 channel statistics
_C100_MEAN = torch.tensor([0.5071, 0.4865, 0.4409]).view(1, 3, 1, 1)
_C100_STD = torch.tensor([0.2673, 0.2564, 0.2762]).view(1, 3, 1, 1)


class CIFAR100Loader:
    """GPU-resident CIFAR-100 with random-crop+flip augmentation for train."""

    def __init__(self, batch=128, device="cuda", seed=0, train=True):
        cache = _prepare_cifar()
        d = np.load(cache)
        x = d["xtr"] if train else d["xte"]
        y = d["ytr"] if train else d["yte"]
        self.x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float().div_(255.0)
        self.x = (self.x - _C100_MEAN.to(device)) / _C100_STD.to(device)
        self.y = torch.from_numpy(y).to(device)
        self.n = self.x.shape[0]
        self.batch = batch
        self.train = train
        self.device = device
        self.g = torch.Generator(device=device).manual_seed(seed)
        # reflect-pad once for random crop
        if train:
            self.xpad = torch.nn.functional.pad(self.x, (4, 4, 4, 4), mode="reflect")

    def sample(self):
        idx = torch.randint(
            0, self.n, (self.batch,), generator=self.g, device=self.device
        )
        y = self.y[idx]
        if not self.train:
            return self.x[idx], y
        xb = self.xpad[idx]
        # random crop 32 from 40
        oy = torch.randint(0, 9, (1,), generator=self.g, device=self.device).item()
        ox = torch.randint(0, 9, (1,), generator=self.g, device=self.device).item()
        xb = xb[:, :, oy : oy + 32, ox : ox + 32]
        # random horizontal flip (per-batch)
        if torch.rand(1, generator=self.g, device=self.device).item() < 0.5:
            xb = torch.flip(xb, dims=[3])
        return xb, y

    def iter_eval(self, batch=500):
        for i in range(0, self.n, batch):
            yield self.x[i : i + batch], self.y[i : i + batch]


# --------------------------------------------------------------- WikiText-103
def _prepare_wikitext():
    tr = os.path.join(DATA, "wikitext103_train.bin")
    va = os.path.join(DATA, "wikitext103_val.bin")
    if os.path.exists(tr) and os.path.exists(va):
        return tr, va
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", cache_dir=HF)

    def tok_split(split, path):
        buf = []
        cur = []
        for i, row in enumerate(ds[split]):
            t = row["text"]
            if t:
                cur.extend(enc.encode_ordinary(t))
            if len(cur) > 1_000_000:
                buf.append(np.array(cur, dtype=np.uint16))
                cur = []
        cur.append(eot)
        buf.append(np.array(cur, dtype=np.uint16))
        arr = np.concatenate(buf)
        arr.tofile(path)
        return arr.shape[0]

    n_tr = tok_split("train", tr)
    n_va = tok_split("validation", va)
    print(f"WikiText-103 tokens: train {n_tr / 1e6:.1f}M, val {n_va / 1e3:.1f}k")
    return tr, va


class WikiTextLoader:
    def __init__(self, ctx=512, batch=24, device="cuda", seed=0, split="train"):
        tr, va = _prepare_wikitext()
        path = tr if split == "train" else va
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.ctx = ctx
        self.batch = batch
        self.device = device
        self.split = split
        self.g = torch.Generator().manual_seed(seed)
        self.n = len(self.data)

    def sample(self):
        ix = torch.randint(0, self.n - self.ctx - 1, (self.batch,), generator=self.g)
        x = torch.stack(
            [torch.from_numpy(self.data[i : i + self.ctx].astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [
                torch.from_numpy(self.data[i + 1 : i + 1 + self.ctx].astype(np.int64))
                for i in ix
            ]
        )
        return x.to(self.device), y.to(self.device)

    def iter_eval(self, max_windows=None):
        """Non-overlapping context windows for validation perplexity."""
        step = self.ctx
        n_win = (self.n - 1) // step
        if max_windows is not None:
            n_win = min(n_win, max_windows)
        for w in range(n_win):
            i = w * step
            x = torch.from_numpy(
                self.data[i : i + self.ctx].astype(np.int64)
            ).unsqueeze(0)
            y = torch.from_numpy(
                self.data[i + 1 : i + 1 + self.ctx].astype(np.int64)
            ).unsqueeze(0)
            yield x.to(self.device), y.to(self.device)


if __name__ == "__main__":
    import sys

    if "cifar" in sys.argv:
        _prepare_cifar()
        print("cifar cached")
    if "wiki" in sys.argv:
        _prepare_wikitext()
        print("wiki cached")
