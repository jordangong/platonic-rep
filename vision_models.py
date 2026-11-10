"""
Vision model zoo for feature extraction: timm ViTs + torchvision ResNets (random init,
MoCo-format contrastive checkpoints, Places365-CNN checkpoints).

load_vision_model(name) -> (model, transform, extract) where extract(images) returns
    layers:    list of N x D_l tensors, one per block (ViT: CLS token of each block output,
               identical to the fx node `blocks.{i}.add_1`; ResNet: global-avg-pooled
               output of each bottleneck/basic block)
    prelogits: N x D tensor, the model's final pooled representation (what a linear
               probe would see). ViT: timm forward_head(pre_logits=True). ResNet:
               avgpool(layer4) == layers[-1].
"""
import torch
import torchvision
import torchvision.transforms as T

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from paths import resolve_timm, resolve_img_ckpt


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
}

IMAGENET_TRANSFORM = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def is_resnet(name):
    return name in RESNETS


def _strip_state_dict(sd, fmt):
    """ MoCo: keep module.encoder_q.* ; Places365-CNN: strip module. ; drop fc (head) """
    if fmt == "moco":
        prefix = "module.encoder_q."
        sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    elif fmt == "places":
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    else:
        raise ValueError(fmt)
    return {k: v for k, v in sd.items() if not k.startswith("fc.")}


def load_resnet(name):
    spec = RESNETS[name]
    torch.manual_seed(0)  # random_init is reproducible
    model = getattr(torchvision.models, spec["arch"])(weights=None)
    if spec["ckpt"] is not None:
        path = resolve_img_ckpt(spec["ckpt"])
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

        def extract(ims):
            prelogits = model(ims)
            layers = taps.pop()
            return layers, prelogits

    else:
        assert "vit" in name, f"only timm ViTs and registered ResNets are supported: {name}"
        local_file = resolve_timm(name)
        # pool files are the hub's timm-format safetensors -> plain state_dict load (not the .npz custom loader)
        overlay = dict(file=local_file, custom_load=False) if local_file else None
        model = timm.create_model(name, pretrained=True, pretrained_cfg_overlay=overlay).cuda().eval()
        taps = _BlockTaps(list(model.blocks), lambda o: o[:, 0, :])
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

        def extract(ims):
            prelogits = model.forward_head(model.forward_features(ims), pre_logits=True)
            layers = taps.pop()
            return layers, prelogits

    return model, transform, extract
