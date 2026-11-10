"""
Vision-vision alignment matrix (paper Sec. 2.2 / Fig. 2, App. C.1): pairwise alignment among all
vision models of a modelset, mutual k-NN with k=10 on 1000 Places365 val images by default.

--layers final : one score per pair from each model's final pooled representation ("prelogits")
--layers all   : max over all (block or final) pairs, as the cross-modal protocol of App. C.2

Features are preprocessed exactly as in measure_alignment.py: elements above the 95th percentile
are clamped, then l2-normalized inside compute_score.

Output: <output_dir>/<dataset>/<modelset>/vision_vision_layers-<layers>/<metric>[_k<topk>].npy
holding {"models", "scores" (M x M), "indices" (M x M x 2 best block pair, -1 for final)}, plus a .csv.
"""
import os
import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

import metrics
import utils
from tasks import get_models, MODELSETS
from measure_alignment import prepare_features, compute_score


def load_feats(path, layers, precise):
    d = torch.load(path, map_location="cpu")
    if layers == "final":
        return [prepare_features(d["prelogits"].float(), exact=precise)]
    feats = d["feats"]
    if isinstance(feats, torch.Tensor):  # N x L x D: clip threshold shared over layers, as for V-L
        f = prepare_features(feats.float(), exact=precise)
        layers = [f[:, i, :] for i in range(f.shape[1])]
    else:
        layers = [prepare_features(f.float(), exact=precise) for f in feats]
    # candidate set = every block + the final representation (index L), so "all" >= "final"
    return layers + [prepare_features(d["prelogits"].float(), exact=precise)]


def vv_alignment(paths, layers, metric, topk, precise):
    m = len(paths)
    scores = np.eye(m)
    indices = np.full((m, m, 2), -1)
    feats = [load_feats(p, layers, precise) for p in tqdm(paths, desc="loading")]
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    for i, j in tqdm(pairs, desc=metric):
        s, idx = compute_score(feats[i], feats[j], metric=metric, topk=topk)
        scores[i, j] = scores[j, i] = s
        if layers == "all" and idx is not None:  # compute_score returns None if every score <= 0
            indices[i, j] = idx
            indices[j, i] = idx[::-1]
    return scores, indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str, default="places365")
    parser.add_argument("--subset",       type=str, default="val_1000")
    parser.add_argument("--modelset",     type=str, default="vv", choices=MODELSETS)
    parser.add_argument("--pool",         type=str, default="cls")
    parser.add_argument("--layers",       type=str, default="final", choices=["final", "all"])
    parser.add_argument("--metric",       type=str, default="mutual_knn", choices=metrics.AlignmentMetrics.SUPPORTED_METRICS)
    parser.add_argument("--topk",         type=int, default=10)
    parser.add_argument("--input_dir",    type=str, default="./results/features")
    parser.add_argument("--output_dir",   type=str, default="./results/alignment")
    parser.add_argument("--precise",      action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    args = parser.parse_args()

    if not args.precise:
        torch.backends.cuda.matmul.allow_tf32 = True

    name = f"{args.metric}_k{args.topk}" if "knn" in args.metric else args.metric
    save_path = os.path.join(args.output_dir, args.dataset, args.modelset, f"vision_vision_layers-{args.layers}", f"{name}.npy")
    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()

    _, models = get_models(args.modelset, modality="vision")
    paths = [utils.to_feature_filename(args.input_dir, args.dataset, args.subset, m, args.pool) for m in models]
    for p in paths:
        assert os.path.exists(p), p

    print(f"dataset: {args.dataset}/{args.subset}  models: {len(models)}  metric: {name}  layers: {args.layers}")
    scores, indices = vv_alignment(paths, args.layers, args.metric, args.topk, args.precise)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"models": models, "scores": scores, "indices": indices})
    with open(save_path[:-4] + ".csv", "w") as f:
        f.write("model," + ",".join(models) + "\n")
        for m, row in zip(models, scores):
            f.write(m + "," + ",".join(f"{v:.6f}" for v in row) + "\n")
    print(f"saved to {save_path}")
