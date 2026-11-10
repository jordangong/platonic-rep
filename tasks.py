VAL_VITS = [
    "vit_tiny_patch16_224.augreg_in21k",
    "vit_small_patch16_224.augreg_in21k",
    "vit_base_patch16_224.augreg_in21k",
    "vit_large_patch16_224.augreg_in21k",
    "vit_base_patch16_224.mae",
    "vit_large_patch16_224.mae",
    "vit_huge_patch14_224.mae",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
    "vit_large_patch14_dinov2.lvd142m",
    "vit_giant_patch14_dinov2.lvd142m",
    "vit_base_patch16_clip_224.laion2b",
    "vit_large_patch14_clip_224.laion2b",
    "vit_huge_patch14_clip_224.laion2b",
    "vit_base_patch16_clip_224.laion2b_ft_in12k",
    "vit_large_patch14_clip_224.laion2b_ft_in12k",
    "vit_huge_patch14_clip_224.laion2b_ft_in12k",
]

# paper App. C.1 ResNets that are public: 1 random-init + MoCo v2 on ImageNet-1k + 10 synthetic-data
# MoCo v2 encoders (Baradad et al. 2021/2022). Missing vs the paper: the contrastive Places365
# ResNet-50 and the 49 Alignment&Uniformity ResNet-18s (never released).
VV_RESNETS = [
    "resnet50.random_init",
    "resnet50.mocov2_200ep_in1k",
    "resnet50.mocov2_dead_leaves_mixed",
    "resnet50.mocov2_feature_vis_dead_leaves",
    "resnet50.mocov2_stat_spectrum_color_wmm",
    "resnet50.mocov2_stylegan_oriented",
    "resnet50.mocov2_noise_mixed4",
    "resnet50.mocov2_shaders1k",
    "resnet50.mocov2_shaders21k",
    "resnet50.mocov2_shaders21k_6mixup",
    "resnet50.mocov2_shaders21k_stylegan",
    "resnet50.mocov2_shaders21k_4mixup_live",
]

# stand-ins for the paper's 49 unreleased A&U ResNet-18s + contrastive Places365 R50 (user choice,
# 2026-11-10): the 14 released A&U SmallAlexNets (same objective/data family) + a public zoo
VV_EXT = [
    "smallalexnet.au_dead_leaves-squares",
    "smallalexnet.au_dead_leaves-oriented",
    "smallalexnet.au_dead_leaves-textures",
    "smallalexnet.au_dead_leaves-mixed",
    "smallalexnet.au_stat-spectrum",
    "smallalexnet.au_stat-wmm",
    "smallalexnet.au_stat-spectrum_color",
    "smallalexnet.au_stat-spectrum_color_wmm",
    "smallalexnet.au_stylegan-random",
    "smallalexnet.au_stylegan-highfreq",
    "smallalexnet.au_stylegan-sparse",
    "smallalexnet.au_stylegan-oriented",
    "smallalexnet.au_feature_vis-random",
    "smallalexnet.au_feature_vis-dead_leaves",
    "resnet18.tv_in1k",
    "resnet34.tv_in1k",
    "resnet50.tv_in1k",
    "resnet101.tv_in1k",
    "resnet152.tv_in1k",
    "resnet50.places365_sup",
]

MODELSETS = ["val", "test", "custom", "smoke", "vv", "vv_ext", "vv_smoke"]


def get_models(modelset, modality='all'):
    
    assert modality in ['all', 'vision', 'language']
    
    if modelset == 'val':
        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
        ]

        lvm_models = list(VAL_VITS)
        
    elif modelset == 'test':
        llm_models = [
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-7B-hf", 
            "google/gemma-2b",
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            # "mistralai/Mixtral-8x22B-v0.1",
            "NousResearch/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-70B",
        ]
        
        lvm_models = []
        
    elif modelset == 'custom':
        llm_models = [
            "bigscience/bloomz-560m",
            "bigscience/bloomz-1b1",
            "bigscience/bloomz-1b7",
            "bigscience/bloomz-3b",
            "bigscience/bloomz-7b1",
            "openlm-research/open_llama_3b",
            "openlm-research/open_llama_7b",
            "openlm-research/open_llama_13b",
            "huggyllama/llama-7b",
            "huggyllama/llama-13b",
            "huggyllama/llama-30b",
            "huggyllama/llama-65b",
            "allenai/OLMo-1B-hf",
            "allenai/OLMo-7B-hf", 
            "google/gemma-2b",
            "google/gemma-7b",
            "mistralai/Mistral-7B-v0.1",
            "mistralai/Mixtral-8x7B-v0.1",
            # "mistralai/Mixtral-8x22B-v0.1", # was too big so did not use
            "NousResearch/Meta-Llama-3-8B",
            "NousResearch/Meta-Llama-3-70B",
        ]
        lvm_models = [
            "vit_giant_patch14_dinov2.lvd142m",
        ]
    elif modelset == 'smoke':
        # tiny pipeline check (fits a 12GB GPU)
        llm_models = ["bigscience/bloomz-560m"]
        lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_small_patch16_224.augreg_in21k",
        ]
    elif modelset == 'vv':
        # vision-vision alignment (paper Fig. 2 / App. C.1), vision only
        llm_models = []
        lvm_models = VAL_VITS + VV_RESNETS
    elif modelset == 'vv_ext':
        # vv + stand-ins for the unreleased models
        llm_models = []
        lvm_models = VAL_VITS + VV_RESNETS + VV_EXT
    elif modelset == 'vv_smoke':
        llm_models = []
        lvm_models = [
            "vit_tiny_patch16_224.augreg_in21k",
            "vit_small_patch16_224.augreg_in21k",
            "resnet50.random_init",
            "resnet50.mocov2_shaders1k",
            "resnet18.tv_in1k",
            "smallalexnet.au_stat-spectrum",
        ]
    else:
        raise ValueError(f"Unknown modelset: {modelset}")
    
    if modality == "vision":
        llm_models = []
    elif modality == "language":
        lvm_models = []

    return llm_models, lvm_models
