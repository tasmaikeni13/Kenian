"""Aggregate results across seeds, apply the pre-registered decision rule, make plots.

Reads results/<task>/<optimizer>_lr<lr>_seed<s>/{metrics.jsonl,final.json}, groups by
condition (optimizer, ignoring seed), computes summary statistics, and writes
summary.json plus PNG figures to results/<task>/figures/.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

# dataviz categorical palette (fixed order; entity-stable colors)
COLORS = {"adamw": "#2a78d6", "kenian": "#1baf7a"}
LABEL = {"adamw": "AdamW", "kenian": "Kenian"}
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "font.size": 11,
        "axes.grid": True,
        "grid.color": "#e6e6e3",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.edgecolor": "#b9b9b4",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def load_runs(task):
    runs = defaultdict(list)  # optimizer -> list of run dicts
    for d in sorted(glob.glob(os.path.join(RESULTS, task, "*"))):
        cfg_p = os.path.join(d, "config.json")
        if not os.path.isfile(cfg_p):
            continue
        cfg = json.load(open(cfg_p))
        name = os.path.basename(d)
        if name.startswith(("smoke", "timing", "pilot")):
            continue
        recs = [json.loads(line) for line in open(os.path.join(d, "metrics.jsonl"))]
        final = (
            json.load(open(os.path.join(d, "final.json")))
            if os.path.isfile(os.path.join(d, "final.json"))
            else None
        )
        runs[cfg["optimizer"]].append(
            {"cfg": cfg, "recs": recs, "final": final, "dir": d}
        )
    return runs


def series(recs, key, kind=None):
    xs, ys = [], []
    for r in recs:
        if key in r and (kind is None or r.get("val_kind") == kind):
            xs.append(r["step"])
            ys.append(r[key])
    return np.array(xs), np.array(ys)


def kenian_series(recs, key):
    xs, ys = [], []
    for r in recs:
        if "kenian" in r and key in r["kenian"]:
            xs.append(r["step"])
            ys.append(r["kenian"][key])
    return np.array(xs), np.array(ys)


def agg_over_seeds(runs_list, key, kind=None, kenian=False, xkey=None):
    """Align seeds on a common step grid, return (grid, mean, std, xmean_optional)."""
    curves = []
    xgrids = []
    for run in runs_list:
        if kenian:
            x, y = kenian_series(run["recs"], key)
        else:
            x, y = series(run["recs"], key, kind)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if len(x) < 2:
            continue
        curves.append((x, y))
        xgrids.append(x)
    if not curves:
        return None
    grid = xgrids[int(np.argmin([len(g) for g in xgrids]))]
    mat = np.array([np.interp(grid, x, y) for x, y in curves])
    out = [grid, mat.mean(0), mat.std(0)]
    return out


def smooth(y, k=9):
    if len(y) < k:
        return y
    ker = np.ones(k) / k
    return np.convolve(y, ker, mode="same")


def plot_curve(
    ax, task, runs, key, kind=None, kenian=False, sm=False, ylabel="", logy=False
):
    any_data = False
    for opt, runs_list in runs.items():
        a = agg_over_seeds(runs_list, key, kind, kenian)
        if a is None:
            continue
        grid, mean, std = a
        y = smooth(mean) if sm else mean
        ax.plot(
            grid,
            y,
            color=COLORS.get(opt, "#666"),
            lw=2.0,
            label=f"{LABEL.get(opt, opt)} (n={len(runs_list)})",
        )
        if len(runs_list) > 1:
            ax.fill_between(
                grid, y - std, y + std, color=COLORS.get(opt, "#666"), alpha=0.15, lw=0
            )
        any_data = True
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    if any_data:
        ax.legend(frameon=False, fontsize=9)
    return any_data


def plot_vs_wall(ax, runs, key, kind, ylabel):
    for opt, runs_list in runs.items():
        curves = []
        for run in runs_list:
            ys, ws = [], []
            for r in run["recs"]:
                if key in r and r.get("val_kind") == kind and "wall_s" in r:
                    ws.append(r["wall_s"] / 60.0)
                    ys.append(r[key])
            ws = np.asarray(ws)
            ys = np.asarray(ys)
            valid = np.isfinite(ws) & np.isfinite(ys)
            ws = ws[valid]
            ys = ys[valid]
            if len(ws) > 1:
                curves.append((ws, ys))
        if not curves:
            continue
        # plot each seed thin + mean is hard on irregular wall grids; plot seed 0 style mean by interp on common wall grid
        wmax = min(c[0][-1] for c in curves)
        grid = np.linspace(min(c[0][0] for c in curves), wmax, 100)
        mat = np.array([np.interp(grid, w, y) for w, y in curves])
        ax.plot(
            grid,
            mat.mean(0),
            color=COLORS.get(opt, "#666"),
            lw=2.0,
            label=LABEL.get(opt, opt),
        )
        if len(curves) > 1:
            ax.fill_between(
                grid,
                mat.mean(0) - mat.std(0),
                mat.mean(0) + mat.std(0),
                color=COLORS.get(opt, "#666"),
                alpha=0.15,
                lw=0,
            )
    ax.set_xlabel("wall-clock (min)")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9)


def decision(task, runs):
    """Apply the repository's validation-loss comparison rule."""
    floor = 0.02 if task == "vision" else 0.01
    out = {}
    finals = {}
    for opt, runs_list in runs.items():
        vals = [r["final"]["val_loss"] for r in runs_list if r.get("final")]
        accs = [
            r["final"].get("val_acc")
            for r in runs_list
            if r.get("final") and "val_acc" in r["final"]
        ]
        if vals:
            finals[opt] = {"val_loss": vals, "val_acc": accs}
    out["finals"] = {
        o: {
            "val_loss_mean": float(np.mean(v["val_loss"])),
            "val_loss_std": float(np.std(v["val_loss"])),
            "n": len(v["val_loss"]),
            "val_acc_mean": (float(np.mean(v["val_acc"])) if v["val_acc"] else None),
        }
        for o, v in finals.items()
    }
    if "adamw" in finals:
        base = np.array(finals["adamw"]["val_loss"])
        for opt in finals:
            if opt == "adamw":
                continue
            ken = np.array(finals[opt]["val_loss"])
            d = base.mean() - ken.mean()  # positive => kenian lower loss
            se = math.sqrt(
                (base.var(ddof=1) if len(base) > 1 else 0) / max(len(base), 1)
                + (ken.var(ddof=1) if len(ken) > 1 else 0) / max(len(ken), 1)
            )
            beats = (d > 0) and (d >= 2 * se) and (d >= floor)
            out[f"{opt}_vs_adamw"] = {
                "delta_val_loss": float(d),
                "SE_d": float(se),
                "floor": floor,
                "2SE": float(2 * se),
                "meets_margin": bool(beats),
                "verdict": (
                    "beats AdamW by pre-registered margin"
                    if beats
                    else "does NOT beat AdamW by pre-registered margin"
                ),
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vision", "language"], required=True)
    args = ap.parse_args()
    task = args.task
    runs = load_runs(task)
    if not runs:
        print(f"no runs found for {task}")
        return
    figdir = os.path.join(RESULTS, task, "figures")
    os.makedirs(figdir, exist_ok=True)

    # 1. train loss
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_curve(ax, task, runs, "train_loss", sm=True, ylabel="training loss")
    ax.set_title(f"{task}: training loss (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "train_loss.png"))
    plt.close(fig)

    # 2. val loss (full)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_curve(ax, task, runs, "val_loss", kind="full", ylabel="validation loss")
    ax.set_title(f"{task}: validation loss")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "val_loss.png"))
    plt.close(fig)

    # 3. val loss vs wall clock
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_vs_wall(ax, runs, "val_loss", "full", "validation loss")
    ax.set_title(f"{task}: validation loss vs wall-clock")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "val_loss_wallclock.png"))
    plt.close(fig)

    # 4. grad norm
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_curve(ax, task, runs, "grad_norm", sm=True, ylabel="gradient norm (pre-clip)")
    ax.set_title(f"{task}: gradient norm")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "grad_norm.png"))
    plt.close(fig)

    # 5. vision accuracy
    if task == "vision":
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plot_curve(
            ax, task, runs, "val_acc", kind="full", ylabel="validation top-1 accuracy"
        )
        ax.set_title("vision: validation accuracy")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "val_acc.png"))
        plt.close(fig)

    # 6. Kenian diagnostics
    diagnostics = [
        ("cap_applied", "cap applied"),
        ("correction_ratio", "‖correction‖_P / ‖base step‖_P"),
        ("slice_norm", "third-order slice norm"),
        ("probe_ms", "probe time (ms)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for (key, lab), ax in zip(diagnostics, axes.ravel()):
        got = False
        for opt in ("kenian",):
            if opt not in runs:
                continue
            a = agg_over_seeds(runs[opt], key, kenian=True)
            if a is None:
                continue
            grid, mean, std = a
            ax.plot(grid, smooth(mean), color=COLORS[opt], lw=1.8, label=LABEL[opt])
            got = True
        ax.set_xlabel("step")
        ax.set_ylabel(lab, fontsize=9)
        if got:
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"{task}: Kenian third-order diagnostics")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "kenian_diagnostics.png"))
    plt.close(fig)

    dec = decision(task, runs)
    with open(os.path.join(RESULTS, task, "summary.json"), "w") as f:
        json.dump(dec, f, indent=2)
    print(json.dumps(dec, indent=2))
    print(f"\nfigures -> {figdir}")


if __name__ == "__main__":
    main()
