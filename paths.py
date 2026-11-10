"""
Resolve model identifiers to local checkpoint pools.

Pools hold plain model dirs (not an HF hub cache), laid out as
    $PLATO_CKPT_ROOT/gen/<modality-tag>/<Vendor>/[<Family>/]<model>/   (HF causal LMs)
    $PLATO_CKPT_ROOT/timm/<timm-model-name>/model.safetensors        (timm models)

PLATO_CKPT_ROOT defaults to ~/checkpoints. If a model is not found in the pool,
the original identifier is returned so HF Hub / timm download as usual.
"""
import glob
import os

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "open_clip_pytorch_model.bin")


def ckpt_root():
    return os.path.expanduser(os.environ.get("PLATO_CKPT_ROOT", "~/checkpoints"))


def resolve_llm(model_name):
    """ HF id (e.g. 'huggyllama/llama-7b') -> local dir if found in the pool, else the id """
    if os.path.isdir(model_name):
        return model_name
    base = model_name.split("/")[-1]
    root = ckpt_root()
    hits = []
    for pattern in (f"{root}/gen/*/*/*/{base}", f"{root}/gen/*/*/{base}"):
        hits += [d for d in glob.glob(pattern) if os.path.isfile(os.path.join(d, "config.json"))]
    if len(hits) > 1:
        # prefer text-only LLM dirs (hpc-2 layout) over VLM dirs
        txt_hits = [d for d in hits if f"{root}/gen/txt/" in d]
        hits = txt_hits if len(txt_hits) == 1 else hits
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous local checkpoints for {model_name}: {hits}")
    if hits:
        print(f"resolved {model_name} -> {hits[0]}")
        return hits[0]
    print(f"{model_name} not in {root}, falling back to HF hub")
    return model_name


def resolve_timm(model_name):
    """ timm name -> local weight file if found in the pool, else None """
    for fn in WEIGHT_FILES:
        path = os.path.join(ckpt_root(), "timm", model_name, fn)
        if os.path.isfile(path):
            print(f"resolved {model_name} -> {path}")
            return path
    print(f"{model_name} not in {ckpt_root()}/timm, falling back to HF hub")
    return None


def resolve_img_ckpt(rel_path):
    """ image-encoder checkpoint under $PLATO_CKPT_ROOT/emb/img/ (no hub fallback: these are not on the hub) """
    path = os.path.join(ckpt_root(), "emb", "img", rel_path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} missing; see docs/platonic-rep.md for where to fetch it")
    print(f"resolved {rel_path} -> {path}")
    return path


def data_root():
    return os.path.expanduser(os.environ.get("PLATO_DATA_ROOT", "~/datasets"))


def load_hf_dataset(dataset, subset, split="train"):
    """
    Load `dataset` at revision `subset`. If a plain parquet copy exists at
    $PLATO_DATA_ROOT/<org>--<name>/<subset>/ (e.g. ~/datasets/minhuh--prh/wit_1024/data/*.parquet),
    load that instead of going through the Hub.
    """
    if dataset == "places365":
        return Places365Val(subset)
    from datasets import load_dataset
    local = os.path.join(data_root(), dataset.replace("/", "--"), subset)
    if os.path.isdir(local):
        print(f"loading {dataset}@{subset} from {local}")
        return load_dataset("parquet", data_dir=local, split=split)
    return load_dataset(dataset, revision=subset, split=split)


class Places365Val:
    """
    Images from the Places365 validation set at $PLATO_DATA_ROOT/places365/val_256/
    (Places365_val_%08d.jpg, 36500 images). subset 'val_<N>' draws N images uniformly at random
    with a fixed seed (paper App. C.1: 1000 Places365 val images for vision-vision alignment).
    Items are dicts {'image': PIL.Image, 'path': str} like the HF datasets used elsewhere.
    """
    NUM_VAL = 36500

    def __init__(self, subset, seed=0):
        import random
        assert subset.startswith("val_"), f"places365 subset must be val_<N>, got {subset}"
        n = int(subset[len("val_"):])
        self.root = os.path.join(data_root(), "places365", "val_256")
        names = [f"Places365_val_{i:08d}.jpg" for i in range(1, self.NUM_VAL + 1)]
        idx = sorted(random.Random(seed).sample(range(self.NUM_VAL), n))
        self.paths = [os.path.join(self.root, names[i]) for i in idx]
        missing = [p for p in self.paths if not os.path.isfile(p)]
        assert not missing, f"{len(missing)} Places365 val images missing, e.g. {missing[0]}"
        print(f"loading places365@{subset} from {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import PIL.Image
        with PIL.Image.open(self.paths[i]) as im:
            return {"image": im.convert("RGB"), "path": self.paths[i]}
