import argparse
import json
import os
import torch
from tqdm import tqdm

from pannot.constants import (
    DEFAULT_SEQ_TOKEN, DEFAULT_SEQ_START_TOKEN, DEFAULT_SEQ_END_TOKEN,
    DEFAULT_STR_TOKEN, DEFAULT_STR_START_TOKEN, DEFAULT_STR_END_TOKEN,
    SEQUENCE_PLACEHOLDER
)
from pannot.conversation import conv_templates
from pannot.model.builder import load_pretrained_model
from pannot.mm_utils import tokenizer_protein_token, get_model_name_from_path
from pannot.utils import disable_torch_init


def evaluate_opi_dataset(args):
    disable_torch_init()

    # Load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        use_flash_attn=args.use_flash_attn if hasattr(args, 'use_flash_attn') else False
    )

    # Load dataset
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    outputs = []

    for item in tqdm(data):
        instruction = item["instruction"]
        instance_group = item["instances"]

        for instance in instance_group:
            sequence = instance["input"]
            target = instance["output"]

            # Prompt construction
            seq_token_se = DEFAULT_SEQ_START_TOKEN + DEFAULT_SEQ_TOKEN + DEFAULT_SEQ_END_TOKEN
            if SEQUENCE_PLACEHOLDER in instruction:
                if getattr(model.config, 'mm_use_seq_start_end', False):
                    prompt = instruction.replace(SEQUENCE_PLACEHOLDER, seq_token_se)
                else:
                    prompt = instruction.replace(SEQUENCE_PLACEHOLDER, DEFAULT_SEQ_TOKEN)
            else:
                if getattr(model.config, 'mm_use_seq_start_end', False):
                    prompt = seq_token_se + "\n" + instruction
                else:
                    prompt = DEFAULT_SEQ_TOKEN + "\n" + instruction

            # Conversation mode
            if "llama" in model_name.lower():
                conv_mode = "pannot_llama_2"
            elif "mistral" in model_name.lower():
                conv_mode = "mistral_instruct"
            elif "v1.6-34b" in model_name.lower():
                conv_mode = "chatml_direct"
            elif "v1" in model_name.lower():
                conv_mode = "pannot_v1"
            else:
                conv_mode = "pannot_v0"

            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            full_prompt = conv.get_prompt()

            input_ids = tokenizer_protein_token(full_prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)
            seq_tensor = torch.tensor(model.get_seq_tower().tokenizer.tokenize(sequence, add_special_tokens=False)).to(model.device)

            # Run generation
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs=input_ids,
                    seqs=[seq_tensor],
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True
                )

            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            outputs.append({
                "Instruction": instruction,
                "input": sequence,
                "target": target.replace(" ; ", "; "),
                "predict": pred.replace(" ; ", "; ")
            })
    with open(args.output_file, "w") as fout:
        json.dump(outputs, fout, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    evaluate_opi_dataset(args)
