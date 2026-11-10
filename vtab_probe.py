"""
VTAB linear probing of vision models (paper App. C.1, the "representation quality" axis of Fig. 2).

Data: uint8 arrays written by task_adaptation/scripts/export_vtab_arrays.py at
$PLATO_DATA_ROOT/vtab-224/<task>/{train,val,test}_{images,labels}.npy. These are VTAB's own
splits/labels, resized to 224x224 (bilinear, no crop) as in VTAB. Train and val are subsampled
to <=10k each, and test is complete.

Per (model, task):
  1. features = the model's final pooled representation ("prelogits", as in measure_vv_alignment
     --layers final). Input: uint8 224px -> [0,1] -> resized to the model's input size if it
     differs (bilinear, antialiased; only the 64px SmallAlexNets) -> the model's mean/std.
  2. multinomial logistic regression on standardized features (full-batch L-BFGS, L2 penalty).
     The penalty is chosen on val accuracy over WEIGHT_DECAYS, then refit on train+val.
  3. top-1 accuracy on test -> results/vtab/<model>/<task>.json

Usage:
  python vtab_probe.py --modelset vv_ext [--tasks cifar100,dtd] [--shard 0/4]
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from paths import data_root
from tasks import get_models, MODELSETS
from vision_models import load_vision_model

TASKS = [
    "caltech101", "cifar100", "dtd", "flowers102", "pets", "sun397", "svhn",       # natural
    "camelyon", "eurosat", "resisc45", "retinopathy",                              # specialized
    "clevr_count", "clevr_distance", "dmlab", "dsprites_location", "dsprites_orientation",
    "kitti_distance", "smallnorb_azimuth", "smallnorb_elevation",                  # structured
]
WEIGHT_DECAYS = [1.0, 3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6]  # strongest first (warm start)


def vtab_root():
    return os.path.join(data_root(), "vtab-224")


def load_split(task, split):
    d = os.path.join(vtab_root(), task)
    return np.load(os.path.join(d, f"{split}_images.npy"), mmap_mode="r"), np.load(os.path.join(d, f"{split}_labels.npy"))


@torch.no_grad()
def extract_prelogits(vm, images, batch_size):
    mean = torch.tensor(vm.mean, device="cuda").view(1, 3, 1, 1)
    std = torch.tensor(vm.std, device="cuda").view(1, 3, 1, 1)
    out = []
    for i in range(0, len(images), batch_size):
        x = torch.from_numpy(np.array(images[i:i + batch_size])).cuda(non_blocking=True)  # copy out of the read-only memmap
        x = x.permute(0, 3, 1, 2).float().div_(255)
        if x.shape[-1] != vm.size:
            x = F.interpolate(x, size=(vm.size, vm.size), mode="bilinear", antialias=True, align_corners=False)
        _, pre = vm.extract((x - mean) / std)
        out.append(pre.float().clone())  # ViT prelogits are x[:, 0] views: clone, or each keeps its B x T x D tokens alive
    return torch.cat(out)


def standardize(train, *others):
    mu, sd = train.mean(0, keepdim=True), train.std(0, keepdim=True).clamp_min(1e-6)
    return [(x - mu) / sd for x in (train, *others)]


def fit_logreg(X, y, num_classes, wd, init=None, max_iter=300):
    """ min_W,b mean CE(XW + b, y) + wd * ||W||^2, full-batch L-BFGS """
    W = torch.zeros(X.shape[1], num_classes, device=X.device) if init is None else init[0].clone()
    b = torch.zeros(num_classes, device=X.device) if init is None else init[1].clone()
    W.requires_grad_(True)
    b.requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + wd * W.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    return W.detach(), b.detach()


@torch.no_grad()
def accuracy(X, y, Wb):
    return ((X @ Wb[0] + Wb[1]).argmax(1) == y).float().mean().item()


def probe(feats, labels, num_classes):
    ytr, yva, yte = labels
    Xtr, Xva = standardize(feats[0], feats[1])
    val_acc, sols, init = {}, {}, None
    for wd in WEIGHT_DECAYS:
        init = sols[wd] = fit_logreg(Xtr, ytr, num_classes, wd, init)
        val_acc[wd] = accuracy(Xva, yva, init)
    best_wd = max(WEIGHT_DECAYS, key=lambda wd: (val_acc[wd], wd))  # ties -> stronger penalty
    Xtv, Xte = standardize(torch.cat([feats[0], feats[1]]), feats[2])
    Wb = fit_logreg(Xtv, torch.cat([ytr, yva]), num_classes, best_wd, init=sols[best_wd])  # warm start
    return {"best_wd": best_wd, "val_acc": {str(k): v for k, v in val_acc.items()},
            "val_acc_best": val_acc[best_wd], "test_acc": accuracy(Xte, yte, Wb)}


def result_path(output_dir, model, task):
    return os.path.join(output_dir, model.replace("/", "_"), f"{task}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelset",   type=str, default="vv_ext", choices=MODELSETS)
    parser.add_argument("--models",     type=str, default=None, help="comma list, overrides --modelset")
    parser.add_argument("--tasks",      type=str, default=",".join(TASKS))
    parser.add_argument("--shard",      type=str, default="0/1", help="i/n: this job runs every n-th model starting at i")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="./results/vtab")
    parser.add_argument("--force_remake", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    models = args.models.split(",") if args.models else get_models(args.modelset, modality="vision")[1]
    i, n = map(int, args.shard.split("/"))
    models = models[i::n]
    tasks = args.tasks.split(",")
    assert all(t in TASKS for t in tasks), tasks
    print(f"shard {args.shard}: {len(models)} models x {len(tasks)} tasks, data {vtab_root()}")

    for model_name in models:
        todo = [t for t in tasks if args.force_remake or not os.path.exists(result_path(args.output_dir, model_name, t))]
        if not todo:
            print(f"{model_name}: done")
            continue
        vm = load_vision_model(model_name)
        for task in todo:
            t0 = time.time()
            with open(os.path.join(vtab_root(), task, "meta.json")) as f:
                meta = json.load(f)
            feats, labels = [], []
            for split in ("train", "val", "test"):
                ims, ys = load_split(task, split)
                feats.append(extract_prelogits(vm, ims, args.batch_size))
                labels.append(torch.from_numpy(ys).cuda())
            t1 = time.time()
            res = probe(feats, labels, meta["num_classes"])
            res.update(model=model_name, task=task, feat_dim=feats[0].shape[1],
                       n={s: len(l) for s, l in zip(("train", "val", "test"), labels)},
                       extract_s=round(t1 - t0, 1), probe_s=round(time.time() - t1, 1))
            path = result_path(args.output_dir, model_name, task)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1)
            print(f"{model_name} {task}: test {res['test_acc']:.4f} (val {res['val_acc_best']:.4f}, wd {res['best_wd']:g}) "
                  f"extract {res['extract_s']}s probe {res['probe_s']}s", flush=True)
            del feats, labels
            torch.cuda.empty_cache()
        del vm
        torch.cuda.empty_cache()
