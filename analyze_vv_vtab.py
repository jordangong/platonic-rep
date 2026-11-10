"""
Paper Fig. 2: vision models converge as competence increases.

Inputs: vtab_probe.py results (results/vtab/<model>/<task>.json) and the vision-vision alignment
matrix (measure_vv_alignment.py, default --layers final).
  - A model "solves" a VTAB task if test acc >= 0.8 x the best acc on that task over all models
    of the modelset (paper App. C.1, which used its 78 models).
  - Models are bucketed by % of the 19 tasks solved: [0,20), [20,40), [40,60), [60,80), [80,100].
  - LEFT panel: mean alignment over model pairs within each bucket, +- standard error.
  - RIGHT panel: 2D embedding with distance = -log(alignment). UMAP if umap-learn is installed
    (as in the paper), else metric MDS.

Writes <output_dir>/<modelset>/{summary.json, models.csv}.
"""
import os
import json
import argparse

import numpy as np

from tasks import get_models, MODELSETS
from vtab_probe import TASKS, result_path

BUCKETS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def family(m):
    """ paper Fig. 2 legend groups """
    if m == "resnet50.random_init":
        return "Random Initialization"
    if ".mae" in m:
        return "MAE"
    if "_clip_" in m:
        return "CLIP"
    if "dinov2" in m or "mocov2" in m or m.startswith("smallalexnet.au_"):
        return "Contrastive"
    return "Classification"  # augreg_in21k ViTs, torchvision IN-1k, Places365-sup


def embed_2d(dist, seed=0):
    try:
        import umap
        return umap.UMAP(metric="precomputed", random_state=seed, n_neighbors=min(15, len(dist) - 1)).fit_transform(dist), "umap"
    except ImportError:
        from sklearn.manifold import MDS
        return MDS(n_components=2, dissimilarity="precomputed", random_state=seed, normalized_stress="auto").fit_transform(dist), "mds"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelset",   type=str, default="vv_ext", choices=MODELSETS)
    parser.add_argument("--alignment",  type=str, default=None, help="default: results/alignment/places365/<modelset>/vision_vision_layers-final/mutual_knn_k10.npy")
    parser.add_argument("--vtab_dir",   type=str, default="./results/vtab")
    parser.add_argument("--output_dir", type=str, default="./results/vtab_analysis")
    parser.add_argument("--threshold",  type=float, default=0.8)
    args = parser.parse_args()

    models = get_models(args.modelset, modality="vision")[1]
    align_path = args.alignment or f"./results/alignment/places365/{args.modelset}/vision_vision_layers-final/mutual_knn_k10.npy"
    a = np.load(align_path, allow_pickle=True).item()
    assert list(a["models"]) == models, "alignment matrix was computed for a different model list"
    A = a["scores"]

    acc = np.full((len(models), len(TASKS)), np.nan)
    for i, m in enumerate(models):
        for j, t in enumerate(TASKS):
            p = result_path(args.vtab_dir, m, t)
            if os.path.exists(p):
                with open(p) as f:
                    acc[i, j] = json.load(f)["test_acc"]
    missing = [(models[i], TASKS[j]) for i, j in zip(*np.where(np.isnan(acc)))]
    assert not missing, f"{len(missing)} missing VTAB results, e.g. {missing[:3]}"

    best = acc.max(0)
    solved = acc >= args.threshold * best
    n_solved = solved.sum(1)
    bucket = np.minimum((n_solved / len(TASKS) * 5).astype(int), 4)

    buckets = []
    for b, name in enumerate(BUCKETS):
        idx = np.where(bucket == b)[0]
        pairs = [A[i, j] for k, i in enumerate(idx) for j in idx[k + 1:]]
        buckets.append({"bucket": name, "n_models": len(idx), "n_pairs": len(pairs),
                        "mean_alignment": float(np.mean(pairs)) if pairs else None,
                        "sem": float(np.std(pairs, ddof=1) / np.sqrt(len(pairs))) if len(pairs) > 1 else None,
                        "models": [models[i] for i in idx]})

    D = -np.log(np.clip(A, 1e-6, 1.0))
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    xy, method = embed_2d(D)

    from scipy.stats import spearmanr
    mean_align = (A.sum(1) - 1) / (len(models) - 1)
    rho, pval = spearmanr(n_solved, mean_align)

    out = os.path.join(args.output_dir, args.modelset)
    os.makedirs(out, exist_ok=True)
    summary = {"modelset": args.modelset, "alignment": align_path, "threshold": args.threshold,
               "tasks": TASKS, "best_acc_per_task": dict(zip(TASKS, best.round(4).tolist())),
               "buckets": buckets, "embedding": method,
               "spearman_nsolved_vs_mean_alignment": {"rho": float(rho), "p": float(pval)},
               "models": [{"model": m, "family": family(m), "n_solved": int(n_solved[i]), "bucket": BUCKETS[bucket[i]],
                           "mean_acc": float(acc[i].mean()), "mean_alignment": float(mean_align[i]),
                           "xy": xy[i].round(4).tolist(), "acc": dict(zip(TASKS, acc[i].round(4).tolist()))}
                          for i, m in enumerate(models)]}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out, "models.csv"), "w") as f:
        f.write("model,family,n_solved,bucket,mean_acc,mean_alignment," + ",".join(TASKS) + "\n")
        for i, m in enumerate(models):
            f.write(f"{m},{family(m)},{n_solved[i]},{BUCKETS[bucket[i]]},{acc[i].mean():.4f},{mean_align[i]:.4f},"
                    + ",".join(f"{v:.4f}" for v in acc[i]) + "\n")

    print(f"{args.modelset}: {len(models)} models, embedding={method}, "
          f"spearman(n_solved, mean alignment) = {rho:.3f} (p={pval:.2g})")
    for b in buckets:
        ma = "-" if b["mean_alignment"] is None else f"{b['mean_alignment']:.3f}"
        se = "" if b["sem"] is None else f" +- {b['sem']:.3f}"
        print(f"  {b['bucket']:8s} n={b['n_models']:2d}  intra-bucket alignment {ma}{se}")
    print(f"saved to {out}")
