"""
Vision model zoo for feature extraction: timm ViTs, torchvision ResNets (random init,
MoCo-format contrastive checkpoints, Places365-CNN, torchvision IN-1k weights) and the
Baradad et al. 2021 SmallAlexNet Alignment&Uniformity encoders.

load_vision_model(name) -> VisionModel(model, transform, extract, size, mean, std):
    transform: PIL -> tensor, the model's own eval transform
    size, mean, std: the model's input resolution and normalization, for already-resized
                     uint8 tensors (VTAB arrays, see vtab.py)
    extract(images) returns
    layers:    list of N x D_l tensors, one per block (ViT: CLS token of each block output,
               identical to the fx node `blocks.{i}.add_1`; ResNet: global-avg-pooled
               output of each bottleneck/basic block; SmallAlexNet: each of its 8 blocks,
               conv blocks spatially averaged)
    prelogits: N x D tensor, the model's final pooled representation (what a linear
               probe would see). ViT: timm forward_head(pre_logits=True). ResNet:
               avgpool(layer4) == layers[-1]. SmallAlexNet: fc6 == layers[5].
"""
from collections import namedtuple

import torch
import torchvision
import torchvision.transforms as T

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from paths import resolve_timm, resolve_img_ckpt, resolve_torch_hub


VisionModel = namedtuple("VisionModel", ["model", "transform", "extract", "size", "mean", "std"])

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ALEXNET_MEAN = (0.44087801806139126, 0.42790631331699347, 0.3867879370752931)
ALEXNET_STD = (0.26826768628079806, 0.2610450402318512, 0.26866836876860795)


# ResNets used by the paper's vision-vision study (App. C.1) that are publicly available.
# ckpt paths are relative to $PLATO_CKPT_ROOT/emb/img/.
_NOISE = "MIT/Baradad-Noise"      # Baradad et al. 2021, large-scale MoCo v2 encoders
_SHADERS = "MIT/Baradad-Shaders"  # Baradad et al. 2022, MoCo v2 encoders
RESNETS = {
    "resnet50.random_init":               dict(arch="resnet50", ckpt=None),
    "resnet50.mocov2_200ep_in1k":         dict(arch="resnet50", fmt="moco", ckpt="Meta/MoCo/moco_v2_200ep/moco_v2_200ep_pretrain.pth.tar"),
    "resnet50.mocov2_dead_leaves_mixed":  dict(arch="resnet50", fmt="moco", ckpt=f"{_NOISE}/dead_leaves-mixed/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_feature_vis_dead_leaves": dict(arch="resnet50", fmt="moco", ckpt=f"{_NOISE}/feature_vis-dead_leaves/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_stat_spectrum_color_wmm": dict(arch="resnet50", fmt="moco", ckpt=f"{_NOISE}/stat-spectrum_color_wmm/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_stylegan_oriented":  dict(arch="resnet50", fmt="moco", ckpt=f"{_NOISE}/stylegan-oriented/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_noise_mixed4":       dict(arch="resnet50", fmt="moco", ckpt=f"{_NOISE}/mixed-4/checkpoint_0799.pth.tar"),
    "resnet50.mocov2_shaders1k":          dict(arch="resnet50", fmt="moco", ckpt=f"{_SHADERS}/shaders1k/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_shaders21k":         dict(arch="resnet50", fmt="moco", ckpt=f"{_SHADERS}/shaders21k/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_shaders21k_6mixup":  dict(arch="resnet50", fmt="moco", ckpt=f"{_SHADERS}/shaders21k_6mixup/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_shaders21k_stylegan": dict(arch="resnet50", fmt="moco", ckpt=f"{_SHADERS}/shaders21k_stylegan/checkpoint_0199.pth.tar"),
    "resnet50.mocov2_shaders21k_4mixup_live": dict(arch="resnet50", fmt="moco", ckpt=f"{_SHADERS}/shaders21k_4mixup_live/checkpoint_0199.pth.tar"),
    # NOT the paper's model (theirs is contrastive on Places365, unreleased): supervised Places365-CNN
    "resnet50.places365_sup":             dict(arch="resnet50", fmt="places", ckpt="MIT/Places365-CNN/resnet50_places365/resnet50_places365.pth.tar"),
    # zoo stand-ins for the paper's unreleased ResNet-18s: torchvision IMAGENET1K_V1 weights,
    # resolved from $PLATO_CKPT_ROOT/torch-hub/ (filename carries the sha256 prefix)
    "resnet18.tv_in1k":                   dict(arch="resnet18", fmt="torchvision", hub_file="resnet18-f37072fd.pth"),
    "resnet34.tv_in1k":                   dict(arch="resnet34", fmt="torchvision", hub_file="resnet34-b627a593.pth"),
    "resnet50.tv_in1k":                   dict(arch="resnet50", fmt="torchvision", hub_file="resnet50-0676ba61.pth"),
    "resnet101.tv_in1k":                  dict(arch="resnet101", fmt="torchvision", hub_file="resnet101-63fe2227.pth"),
    "resnet152.tv_in1k":                  dict(arch="resnet152", fmt="torchvision", hub_file="resnet152-394f9c45.pth"),
}

# Baradad et al. 2021 small-scale encoders: SmallAlexNet trained with Alignment & Uniformity on 105k
# 64x64 images per noise dataset. Same objective/data family as the paper's unreleased A&U ResNet-18s.
BARADAD_SMALL_DATASETS = [
    "dead_leaves-squares", "dead_leaves-oriented", "dead_leaves-textures", "dead_leaves-mixed",
    "stat-spectrum", "stat-wmm", "stat-spectrum_color", "stat-spectrum_color_wmm",
    "stylegan-random", "stylegan-highfreq", "stylegan-sparse", "stylegan-oriented",
    "feature_vis-random", "feature_vis-dead_leaves",
]
ALEXNETS = {f"smallalexnet.au_{d}": dict(ckpt=f"MIT/Baradad-Noise-small/{d}/encoder.pth") for d in BARADAD_SMALL_DATASETS}
ALEXNET_PRELOGITS_BLOCK = 5  # fc6, the layer their linear_eval_imagenet100.py probes (--layer_index 5)

# their eval transform (align_uniform/linear_eval_imagenet100.py)
ALEXNET_TRANSFORM = T.Compose([
    T.Resize(70),
    T.CenterCrop(64),
    T.ToTensor(),
    T.Normalize(ALEXNET_MEAN, ALEXNET_STD),
])

IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def is_resnet(name):
    return name in RESNETS


class SmallAlexNet(torch.nn.Module):
    """
    Verbatim architecture of mbaradad/learning_with_noise align_uniform/encoder.py (adapted there
    from HobbitLong/CMC), so its encoder.pth state_dicts load with strict=True. 64x64 input.
    """
    def __init__(self, in_channel=3, feat_dim=128):
        super().__init__()
        nn = torch.nn

        def conv(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
            return nn.Sequential(*(layers + [nn.MaxPool2d(3, 2)] if pool else layers))

        class L2Norm(nn.Module):
            def forward(self, x):
                return x / x.norm(p=2, dim=1, keepdim=True)

        self.blocks = nn.ModuleList([
            conv(in_channel, 96, True),
            conv(96, 192, True),
            conv(192, 384, False),
            conv(384, 384, False),
            conv(384, 192, True),
            nn.Sequential(nn.Flatten(), nn.Linear(192 * 7 * 7, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(inplace=True)),  # fc6
            nn.Sequential(nn.Linear(4096, 4096, bias=False), nn.BatchNorm1d(4096), nn.ReLU(inplace=True)),  # fc7
            nn.Sequential(nn.Linear(4096, feat_dim), L2Norm()),  # fc8
        ])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


def load_alexnet(name):
    model = SmallAlexNet()
    sd = torch.load(resolve_img_ckpt(ALEXNETS[name]["ckpt"]), map_location="cpu", weights_only=True)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return model


def _strip_state_dict(sd, fmt):
    """ MoCo: keep module.encoder_q.* ; Places365-CNN: strip module. ; drop fc (head) """
    if fmt == "moco":
        prefix = "module.encoder_q."
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    elif fmt == "places":
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    elif fmt == "torchvision":
        pass
    else:
        raise ValueError(fmt)
    return {k: v for k, v in sd.items() if not k.startswith("fc.")}


def load_resnet(name):
    spec = RESNETS[name]
    torch.manual_seed(0)  # random_init is reproducible
    model = getattr(torchvision.models, spec["arch"])(weights=None)
    if spec.get("ckpt") or spec.get("hub_file"):
        path = resolve_img_ckpt(spec["ckpt"]) if spec.get("ckpt") else resolve_torch_hub(spec["hub_file"])
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        sd = _strip_state_dict(ckpt.get("state_dict", ckpt), spec["fmt"])
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected, f"{name}: unexpected keys {unexpected[:5]}"
        assert all(k.startswith("fc.") for k in missing), f"{name}: missing keys {missing[:5]}"
    model.fc = torch.nn.Identity()
    return model


class _BlockTaps:
    """ forward hooks that record a reduced view of each block's output """
    def __init__(self, modules, reduce):
        self.outs = []
        self.handles = [m.register_forward_hook(lambda m, i, o: self.outs.append(reduce(o))) for m in modules]

    def pop(self):
        outs, self.outs = self.outs, []
        return outs


def load_vision_model(name):
    if is_resnet(name):
        model = load_resnet(name).cuda().eval()
        blocks = [b for stage in (model.layer1, model.layer2, model.layer3, model.layer4) for b in stage]
        taps = _BlockTaps(blocks, lambda o: o.mean(dim=(2, 3)))
        transform = IMAGENET_TRANSFORM
        size, mean, std = 224, IMAGENET_MEAN, IMAGENET_STD

        def extract(ims):
            taps.pop()  # drop leftovers of a forward that raised (e.g. OOM)
            prelogits = model(ims)
            layers = taps.pop()
            return layers, prelogits

    elif name in ALEXNETS:
        model = load_alexnet(name).cuda().eval()
        # conv blocks: spatial average (as their forward(average=True)); fc blocks are already N x D
        taps = _BlockTaps(list(model.blocks), lambda o: o.mean(dim=(2, 3)) if o.ndim == 4 else o)
        transform = ALEXNET_TRANSFORM
        size, mean, std = 64, ALEXNET_MEAN, ALEXNET_STD

        def extract(ims):
            taps.pop()
            model(ims)
            layers = taps.pop()
            return layers, layers[ALEXNET_PRELOGITS_BLOCK]

    else:
        assert "vit" in name, f"only timm ViTs and registered ResNets/AlexNets are supported: {name}"
        local_file = resolve_timm(name)
        # pool files are the hub's timm-format safetensors -> plain state_dict load (not the .npz custom loader)
        overlay = dict(file=local_file, custom_load=False) if local_file else None
        model = timm.create_model(name, pretrained=True, pretrained_cfg_overlay=overlay).cuda().eval()
        taps = _BlockTaps(list(model.blocks), lambda o: o[:, 0, :].clone())  # clone: a view keeps the whole token tensor alive
        data_cfg = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**data_cfg)
        size, mean, std = data_cfg["input_size"][-1], tuple(data_cfg["mean"]), tuple(data_cfg["std"])

        def extract(ims):
            taps.pop()
            prelogits = model.forward_head(model.forward_features(ims), pre_logits=True)
            layers = taps.pop()
            return layers, prelogits

    return VisionModel(model, transform, extract, size, mean, std)
