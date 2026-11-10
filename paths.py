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


def data_root():
    return os.path.expanduser(os.environ.get("PLATO_DATA_ROOT", "~/datasets"))


def load_hf_dataset(dataset, subset, split="train"):
    """
    Load `dataset` at revision `subset`. If a plain parquet copy exists at
    $PLATO_DATA_ROOT/<org>--<name>/<subset>/ (e.g. ~/datasets/minhuh--prh/wit_1024/data/*.parquet),
    load that instead of going through the Hub.
    """
    from datasets import load_dataset
    local = os.path.join(data_root(), dataset.replace("/", "--"), subset)
    if os.path.isdir(local):
        print(f"loading {dataset}@{subset} from {local}")
        return load_dataset("parquet", data_dir=local, split=split)
    return load_dataset(dataset, revision=subset, split=split)
