""" Fig. 2-style plot from analyze_vv_vtab.py's summary.json (needs matplotlib, not in the pinned env):
  uv run --with matplotlib --with numpy==1.26.4 python plot_vv_vtab.py --modelset vv_ext
Left: intra-bucket mean alignment (mutual kNN, k=10) +- s.e. Right: 2-D embedding of -log(alignment),
colour = bucket (% VTAB tasks solved), marker = family. """
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BUCKET_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blue ramp, light -> dark
FAMILY_MARKERS = {"Random Initialization": "X", "MAE": "s", "CLIP": "D", "Contrastive": "o", "Classification": "^"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main(args):
    d = json.load(open(os.path.join(args.output_dir, args.modelset, "summary.json")))
    buckets = d["buckets"]
    names = [b["bucket"] for b in buckets]

    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]})

    # left: intra-bucket alignment
    for i, b in enumerate(buckets):
        if b["mean_alignment"] is None:
            ax1.text(i, 0.005, f"n={b['n_models']}\n(no pairs)", ha="center", va="bottom", color=INK2, fontsize=8)
            continue
        ax1.bar(i, b["mean_alignment"], width=0.62, color=BUCKET_COLORS[i], zorder=2)
        ax1.errorbar(i, b["mean_alignment"], yerr=b["sem"], color=INK2, capsize=3, lw=1, zorder=3)
        ax1.text(i, b["mean_alignment"] + b["sem"] + 0.008, f"{b['mean_alignment']:.2f}\nn={b['n_models']}",
                 ha="center", va="bottom", fontsize=8, color=INK)
    ax1.set_xticks(range(len(names)), names)
    ax1.set_xlabel("% VTAB tasks solved")
    ax1.set_ylabel("intra-bucket alignment (mutual kNN, k=10)")
    ax1.set_ylim(0, max(b["mean_alignment"] + b["sem"] for b in buckets if b["mean_alignment"]) * 1.25)
    ax1.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax1.spines[["top", "right"]].set_visible(False)
    rho = d["spearman_nsolved_vs_mean_alignment"]
    ax1.set_title(f"Spearman(tasks solved, mean alignment) = {rho['rho']:.2f}" if isinstance(rho, dict)
                  else f"Spearman(tasks solved, mean alignment) = {rho:.2f}", fontsize=9, color=INK2)

    # right: embedding
    for m in d["models"]:
        x, y = m["xy"]
        ax2.scatter(x, y, s=64, marker=FAMILY_MARKERS.get(m["family"], "o"), c=BUCKET_COLORS[names.index(m["bucket"])],
                    edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax2.set_xticks([]), ax2.set_yticks([])
    ax2.spines[:].set_visible(False)
    ax2.set_title(f"{d['embedding'].upper()} of −log(alignment), {len(d['models'])} models", fontsize=9, color=INK2)
    fam = [Line2D([], [], marker=mk, ls="", color=INK2, markersize=7, label=f) for f, mk in FAMILY_MARKERS.items()
           if any(m["family"] == f for m in d["models"])]
    buc = [Line2D([], [], marker="o", ls="", color=c, markersize=8, label=n) for n, c in zip(names, BUCKET_COLORS)]
    leg = ax2.legend(handles=buc, title="tasks solved", loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, title_fontsize=8)
    ax2.add_artist(leg)
    ax2.legend(handles=fam, title="family", loc="lower left", bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=8, title_fontsize=8)

    fig.tight_layout()
    out = os.path.join(args.output_dir, args.modelset, "fig2.png")
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelset",   type=str, default="vv_ext")
    parser.add_argument("--output_dir", type=str, default="./results/vtab_analysis")
    main(parser.parse_args())
