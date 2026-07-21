"""Models for the Kenian benchmarks: a ~38M ViT and a ~39M decoder-only transformer.

Both use GELU, so the third-order derivative is well-defined, and pre-LayerNorm.
They are sized to roughly 39 million parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- ViT
class Attention(nn.Module):
    """Scaled-dot-product attention.

    Uses F.scaled_dot_product_attention, which dispatches to fast flash/efficient CUDA
    kernels for ordinary forward+backward. Those kernels do NOT implement double-backward,
    which the Kenian probe needs (𝒦 = ∇³L requires a graph through the first backward), so
    the probe forward is run under the MATH sdpa backend (see train.py) — that path is
    double-backward-able. Both optimizers share this identical module, keeping the
    comparison fair; only the small sub-batch probe pays the slower math kernel.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, causal=False):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        o = o.transpose(1, 2).reshape(B, T, C)
        return self.proj(o)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp, causal=False):
        super().__init__()
        self.causal = causal
        self.n1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Linear(mlp, dim))

    def forward(self, x):
        x = x + self.attn(self.n1(x), causal=self.causal)
        x = x + self.mlp(self.n2(x))
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img=32,
        patch=4,
        in_ch=3,
        dim=512,
        depth=12,
        heads=8,
        mlp=2048,
        n_classes=100,
    ):
        super().__init__()
        assert img % patch == 0
        n_patch = (img // patch) ** 2
        self.patch = patch
        self.proj = nn.Linear(in_ch * patch * patch, dim)
        self.pos = nn.Parameter(torch.zeros(1, n_patch, dim))
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp, causal=False) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(_init_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # B,C,H/p,W/p,p,p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        x = self.proj(x) + self.pos
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


# --------------------------------------------------------------------------- GPT
class GPT(nn.Module):
    def __init__(self, vocab=50257, ctx=512, dim=384, depth=11, heads=6, mlp=1536):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, ctx, dim))
        self.blocks = nn.ModuleList(
            [Block(dim, heads, mlp, causal=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.apply(_init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos[:, :T]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.trunc_normal_(m.weight, std=0.02)


def count_params(model):
    # count unique tensors (tied weights once)
    seen, total = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


if __name__ == "__main__":
    v = ViT()
    g = GPT()
    print(f"ViT params: {count_params(v) / 1e6:.2f}M")
    print(f"GPT params: {count_params(g) / 1e6:.2f}M")
