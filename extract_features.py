import gc
import os
import argparse

from tqdm import trange

import torch

from tasks import get_models, MODELSETS
from models import load_llm, load_tokenizer
from vision_models import load_vision_model
from paths import load_hf_dataset
import utils 
    

def extract_llm_features(filenames, dataset, args):
    """
    Extracts features from language models.
    Args:
        filenames: list of language model names by huggingface identifiers
        dataset: huggingface dataset
        args: argparse arguments
    """

    texts = [str(x['text'][args.caption_idx]) for x in dataset]
        
    for llm_model_name in filenames[::-1]:
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, llm_model_name,
            pool=args.pool, prompt=args.prompt, caption_idx=args.caption_idx,
        )
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{llm_model_name}")
        print(f'save_path: \t{save_path}')
        
        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue
        
        language_model = load_llm(llm_model_name, qlora=args.qlora, force_download=args.force_download)
        llm_param_count = sum([p.numel() for p in language_model.parameters()])
        tokenizer = load_tokenizer(llm_model_name)
    
        tokens = tokenizer(texts, padding="longest", return_tensors="pt")        
        llm_feats, losses, bpb_losses = [], [], []

        # hack to get around HF mapping data incorrectly when using model-parallel
        device = next(language_model.parameters()).device

        for i in trange(0, len(dataset), args.batch_size):
            # get embedding cuda device
            token_inputs = {k: v[i:i+args.batch_size].to(device).long() for (k, v) in tokens.items()}

            with torch.no_grad():
                if "olmo" in llm_model_name.lower():
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                        output_hidden_states=True,
                    )
                else:
                    llm_output = language_model(
                        input_ids=token_inputs["input_ids"],
                        attention_mask=token_inputs["attention_mask"],
                    )

                loss, avg_loss = utils.cross_entropy_loss(token_inputs, llm_output)
                losses.extend(avg_loss.cpu())
                
                bpb = utils.cross_entropy_to_bits_per_unit(loss.cpu(), texts[i:i+args.batch_size], unit="byte")
                bpb_losses.extend(bpb)
                
                # make sure to do all the processing in cpu to avoid memory problems
                if args.pool == 'avg':
                    feats = torch.stack(llm_output["hidden_states"]).permute(1, 0, 2, 3)
                    mask = token_inputs["attention_mask"].unsqueeze(-1).unsqueeze(1)
                    feats = (feats * mask).sum(2) / mask.sum(2)
                elif args.pool == 'last':
                    feats = [v[:, -1, :] for v in llm_output["hidden_states"]]
                    feats = torch.stack(feats).permute(1, 0, 2) 
                else:
                    raise NotImplementedError(f"unknown pooling {args.pool}")
                llm_feats.append(feats.cpu())

        print(f"average loss:\t{torch.stack(losses).mean().item()}")
        save_dict = {
            "feats": torch.cat(llm_feats).cpu(),
            "num_params": llm_param_count,
            "mask": tokens["attention_mask"].cpu(),
            "loss": torch.stack(losses).mean(),
            "bpb": torch.stack(bpb_losses).mean(),
        }

        torch.save(save_dict, save_path)

        del language_model, tokenizer, llm_feats, llm_output
        gc.collect()  # free cyclic refs first, else empty_cache can't release them and the next
        torch.cuda.empty_cache()  # device_map="auto" sizes against stale memory -> CPU offload
        torch.cuda.ipc_collect()
    return
    
        
def extract_lvm_features(filenames, dataset, args):
    """
    Extracts features from vision models (timm ViTs and registered ResNets, see vision_models.py).
    Saves per-block features under "feats" (N x L x D tensor if all blocks share D, else a list
    of N x D_l tensors) and the final pooled representation under "prelogits" (N x D).
    Args:
        filenames: list of vision model names (timm identifiers or vision_models.RESNETS keys)
        dataset: indexable dataset whose items have an 'image' (PIL)
        args: argparse arguments
    """
    assert args.pool == 'cls', "pooling is not supported for lvm features"

    for lvm_model_name in filenames:
        save_path = utils.to_feature_filename(
            args.output_dir, args.dataset, args.subset, lvm_model_name,
            pool=args.pool, prompt=None, caption_idx=None,
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        print(f"\ndataset: \t{args.dataset}")
        print(f"subset:    \t{args.subset}")
        print(f"processing:\t{lvm_model_name}")
        print(f'save_path: \t{save_path}')

        if os.path.exists(save_path) and not args.force_remake:
            print("file exists. skipping")
            continue

        vision_model, transform, extract, *_ = load_vision_model(lvm_model_name)
        lvm_param_count = sum([p.numel() for p in vision_model.parameters()])

        lvm_feats, prelogits = [], []
        for i in trange(0, len(dataset), args.batch_size):
            with torch.no_grad():
                ims = torch.stack([transform(dataset[j]['image']) for j in range(i, min(i+args.batch_size, len(dataset)))]).cuda()
                layers, pre = extract(ims)
                lvm_feats.append([v.cpu() for v in layers])
                prelogits.append(pre.cpu())

        layers = [torch.cat([b[l] for b in lvm_feats]) for l in range(len(lvm_feats[0]))]
        if len({v.shape[-1] for v in layers}) == 1:
            layers = torch.stack(layers).permute(1, 0, 2)  # N x L x D, same layout as before
        torch.save({"feats": layers, "prelogits": torch.cat(prelogits), "num_params": lvm_param_count}, save_path)

        del vision_model, transform, extract, lvm_feats, prelogits, layers
        gc.collect()  # free cyclic refs first, else empty_cache can't release them and the next
        torch.cuda.empty_cache()  # device_map="auto" sizes against stale memory -> CPU offload
        torch.cuda.ipc_collect()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--force_remake",   action="store_true")
    parser.add_argument("--num_samples",    type=int, default=1024)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--pool",           type=str, default='avg', choices=['avg', 'cls'])
    parser.add_argument("--prompt",         action="store_true")
    parser.add_argument("--dataset",        type=str, default="prh")
    parser.add_argument("--subset",         type=str, default="wit_1024")
    parser.add_argument("--caption_idx",    type=int, default=0)
    parser.add_argument("--modelset",       type=str, default="val", choices=MODELSETS)
    parser.add_argument("--modality",       type=str, default="all", choices=["vision", "language", "all"])
    parser.add_argument("--output_dir",     type=str, default="./results/features")
    parser.add_argument("--qlora",          action="store_true")
    args = parser.parse_args()

    if args.qlora:
        print(f"QLoRA is set to True. The alignment score will be slightly off.")

    llm_models, lvm_models = get_models(args.modelset, modality=args.modality)
    
    # load dataset once outside    
    dataset = load_hf_dataset(args.dataset, args.subset, split='train')

    if args.modality in ["all", "language"]:
        # extract all language model features
        extract_llm_features(llm_models, dataset, args)
    
    if args.modality in ["all", "vision"]:
        # extract all vision model features
        extract_lvm_features(lvm_models, dataset, args)
