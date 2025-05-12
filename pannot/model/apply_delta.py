"""
Usage:
python3 -m fastchat.model.apply_delta \
    --base ~/model_weights/pannot-llama-7b \
    --target ~/model_weights/pannot-vicuna-7b \
    --delta lmsys/vicuna-7b-delta
"""
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pannot import PannotLlamaForCausalLM

def apply_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    base = PannotLlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("Loading delta model")
    delta = PannotLlamaForCausalLM.from_pretrained(
        delta_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    print("Loading tokenizer (delta)")
    tokenizer = AutoTokenizer.from_pretrained(delta_path)

    print("Applying delta weights into base")
    base_state = base.state_dict()
    delta_state = delta.state_dict()

    # projector fields that may exist only in delta
    optional_proj = {
        "model.mm_seq_projector.weight",
        "model.mm_seq_projector.bias",
        "model.mm_str_projector.weight",
        "model.mm_str_projector.bias",
    }

    for name, dparam in tqdm(delta_state.items(), desc="Applying delta"):
        if name not in base_state:
            # allow missing projector params
            assert name in optional_proj, f"Unexpected param in delta but not in base: {name}"
            continue

        bparam = base_state[name]
        if dparam.shape == bparam.shape:
            # same shape: simple additive merge
            dparam.data += bparam.data
        else:
            # embedding matrices often padded: add only overlapping slice
            assert name in {
                "model.embed_tokens.weight",
                "lm_head.weight",
            }, f"Size mismatch for non-embed param {name}: {dparam.shape} vs {bparam.shape}"
            # get minimal slice
            # min_rows = min(dparam.shape[0], bparam.shape[0])
            # min_cols = min(dparam.shape[1], bparam.shape[1])
            # dparam.data[:min_rows, :min_cols] += bparam.data[:min_rows, :min_cols]
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

    print("Saving merged model to", target_model_path)
    delta.save_pretrained(target_model_path)       # delta now holds base+delta
    tokenizer.save_pretrained(target_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base",   dest="base_model_path",   required=True)
    parser.add_argument("--target", dest="target_model_path", required=True)
    parser.add_argument("--delta",  dest="delta_path",        required=True)
    args = parser.parse_args()

    apply_delta(args.base_model_path, args.target_model_path, args.delta_path)

# """
# Usage:
# python3 -m fastchat.model.apply_delta --base ~/model_weights/llama-7b --target ~/model_weights/vicuna-7b --delta lmsys/vicuna-7b-delta
# """
# import argparse

# import torch
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from pannot import PannotLlamaForCausalLM


# def apply_delta(base_model_path, target_model_path, delta_path):
#     print("Loading base model")
#     base = AutoModelForCausalLM.from_pretrained(
#         base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

#     print("Loading delta")
#     delta = PannotLlamaForCausalLM.from_pretrained(delta_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
#     delta_tokenizer = AutoTokenizer.from_pretrained(delta_path)

#     print("Applying delta")
#     for name, param in tqdm(delta.state_dict().items(), desc="Applying delta"):
#         if name not in base.state_dict():
#             assert name in ['model.mm_projector.weight', 'model.mm_projector.bias'], f'{name} not in base model'
#             continue
#         if param.data.shape == base.state_dict()[name].shape:
#             param.data += base.state_dict()[name]
#         else:
#             assert name in ['model.embed_tokens.weight', 'lm_head.weight'], \
#                 f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
#             bparam = base.state_dict()[name]
#             param.data[:bparam.shape[0], :bparam.shape[1]] += bparam

#     print("Saving target model")
#     delta.save_pretrained(target_model_path)
#     delta_tokenizer.save_pretrained(target_model_path)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--base-model-path", type=str, required=True)
#     parser.add_argument("--target-model-path", type=str, required=True)
#     parser.add_argument("--delta-path", type=str, required=True)

#     args = parser.parse_args()

#     apply_delta(args.base_model_path, args.target_model_path, args.delta_path)
