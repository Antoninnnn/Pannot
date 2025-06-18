# CUDA_VISIBLE_DEVICES=0 python pannot/eval/eval_opi/eval_opi_dataset_pannot.py \
#   --input_file /scratch/user/yining_yang/TAMU/PhD/Pannot/data/opi/OPI_DATA/SU/EC_number/test/CLEAN_EC_number_price_test.jsonl \
#   --output_file /scratch/user/yining_yang/TAMU/PhD/Pannot/results/ec_price_test_predictions.json \
#   --model-path /scratch/user/yining_yang/TAMU/PhD/Pannot/checkpoints/pannot-Meta-Llama-3.1-8B-Instruct-pretrain-v00 \
#   --model-base /scratch/user/yining_yang/TAMU/PhD/Pannot/local_pretrained_llm/Meta-Llama-3.1-8B-Instruct \
#   --temperature 0.2 \
#   --top_p 0.9 \
#   --num_beams 1 \
#   --max_new_tokens 2048


import argparse
import json
import os
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    StoppingCriteria
)
from peft import PeftModel, PeftConfig
from pannot.constants import DEFAULT_SEQ_TOKEN, DEFAULT_SEQ_START_TOKEN, DEFAULT_SEQ_END_TOKEN
from pannot.conversation import conv_templates
from pannot.mm_utils import tokenizer_protein_token, get_model_name_from_path
from pannot.model.builder import load_pretrained_model
from pannot.utils import disable_torch_init

SEQUENCE_PLACEHOLDER = '<seq-placeholder>'


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        self.tokenizer = tokenizer

        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True

        outputs = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        for keyword in self.keywords:
            if len(keyword) > 1 and keyword in outputs:
                return True

        return False

def evaluate_opi_dataset(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    is_lora = os.path.exists(os.path.join(args.model_path, 'adapter_config.json'))

    if is_lora:
        config = PeftConfig.from_pretrained(args.model_path)
        base_model_path = config.base_model_name_or_path or args.model_base
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map='auto'
        )
        model = PeftModel.from_pretrained(base_model, args.model_path, torch_dtype=torch.float16, device_map='auto')
        try:
            model = model.merge_and_unload()
        except Exception as e:
            print(f"LoRA merge failed: {e}, using unmerged model")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    else:
        tokenizer, model, _, _ = load_pretrained_model(
            args.model_path, args.model_base, model_name,
            use_flash_attn=getattr(args, 'use_flash_attn', False)
        )

    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    if hasattr(model, 'get_seq_tower'):
        model.get_seq_tower().to(model_device)

    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    eos_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"] if tokenizer.convert_tokens_to_ids(t) is not None]
    pad_token_id = tokenizer.pad_token_id or eos_token_ids[0]

    stopping_keywords = ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"]

    outputs = []

    for item in tqdm(data):
        instruction = item["instruction"]
        instance_group = item["instances"]

        for instance in instance_group:
            sequence = instance["input"]
            target = instance["output"]

            seq_token_se = DEFAULT_SEQ_START_TOKEN + DEFAULT_SEQ_TOKEN + DEFAULT_SEQ_END_TOKEN
            if SEQUENCE_PLACEHOLDER in instruction:
                prompt = instruction.replace(SEQUENCE_PLACEHOLDER, seq_token_se)
            else:
                prompt = seq_token_se + "\n" + instruction

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

            tokenized_prompt = tokenizer_protein_token(full_prompt, tokenizer, return_tensors="pt").unsqueeze(0)
            input_ids = tokenized_prompt.to(model.device)
            attention_mask = torch.ones_like(input_ids)

            if hasattr(model, 'get_seq_tower'):
                tokenized_seq = model.get_seq_tower().tokenize(sequence)
                seq_input_id = tokenized_seq["input_ids"].squeeze(0).to(model.device)
                seq_attention_mask = tokenized_seq["attention_mask"].squeeze(0).to(model.device)
                seqs = [seq_input_id]
                seq_attention_masks = [seq_attention_mask]
            else:
                seqs = None
                seq_attention_masks = None

            stopping_criteria = [KeywordsStoppingCriteria(stopping_keywords, tokenizer, input_ids)]

            with torch.no_grad():
                output_ids = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    seqs=seqs,
                    seq_attention_mask=seq_attention_masks,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    eos_token_id=eos_token_ids,
                    pad_token_id=pad_token_id,
                    stopping_criteria=stopping_criteria
                )

            pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            outputs.append({
                "Instruction": instruction,
                "input": sequence,
                "target": target.replace(" ; ", "; "),
                "predict": pred.replace(" ; ", "; ")
            })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
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
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    evaluate_opi_dataset(args)

# def evaluate_opi_dataset(args):
#     disable_torch_init()

#     # Load model
#     model_name = get_model_name_from_path(args.model_path)
#     tokenizer, model, _, _ = load_pretrained_model(
#         args.model_path, args.model_base, model_name,
#         use_flash_attn=args.use_flash_attn if hasattr(args, 'use_flash_attn') else False
#     )

#     model_device = next(model.parameters()).device
#     print(model_device)
#     model_dtype = next(model.parameters()).dtype  # typically torch.float16 or torch.bfloat16
#     print(model_dtype)

#     model.get_seq_tower().to(model_device)

#     # Load dataset
#     with open(args.input_file, "r") as f:
#         data = [json.loads(line) for line in f]

#     outputs = []

#     for item in tqdm(data):
#         instruction = item["instruction"]
#         instance_group = item["instances"]

#         for instance in instance_group:
#             sequence = instance["input"]
#             target = instance["output"]

#             # Prompt construction
#             seq_token_se = DEFAULT_SEQ_START_TOKEN + DEFAULT_SEQ_TOKEN + DEFAULT_SEQ_END_TOKEN
#             if SEQUENCE_PLACEHOLDER in instruction:
#                 if getattr(model.config, 'mm_use_seq_start_end', False):
#                     prompt = instruction.replace(SEQUENCE_PLACEHOLDER, seq_token_se)
#                 else:
#                     prompt = instruction.replace(SEQUENCE_PLACEHOLDER, DEFAULT_SEQ_TOKEN)
#             else:
#                 if getattr(model.config, 'mm_use_seq_start_end', False):
#                     prompt = seq_token_se + "\n" + instruction
#                 else:
#                     prompt = DEFAULT_SEQ_TOKEN + "\n" + instruction

#             # Conversation mode
#             if "llama" in model_name.lower():
#                 conv_mode = "pannot_llama_2"
#             elif "mistral" in model_name.lower():
#                 conv_mode = "mistral_instruct"
#             elif "v1.6-34b" in model_name.lower():
#                 conv_mode = "chatml_direct"
#             elif "v1" in model_name.lower():
#                 conv_mode = "pannot_v1"
#             else:
#                 conv_mode = "pannot_v0"

#             conv = conv_templates[conv_mode].copy()
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], None)
#             full_prompt = conv.get_prompt()

#             input_ids = tokenizer_protein_token(full_prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)
#             tokenized = model.get_seq_tower().tokenize(sequence)
#             seq_input_id = tokenized["input_ids"].squeeze(0).to(model.device)
#             seq_attention_mask = tokenized["attention_mask"].squeeze(0).to(model.device)
            
#             print("prompt: ",full_prompt)
#             print("prompt id: ",input_ids)
#             print("seq id",seq_input_id)
#             print("seq attention",seq_attention_mask)
            
#             # Run generation
#             with torch.inference_mode():
#                 output_ids = model.generate(
#                     inputs=input_ids,
#                     seqs=[seq_input_id],
#                     seq_attention_mask=[seq_attention_mask],
#                     do_sample=args.temperature > 0,
#                     temperature=args.temperature,
#                     top_p=args.top_p,
#                     num_beams=args.num_beams,
#                     max_new_tokens=args.max_new_tokens,
#                     use_cache=True
#                 )

#             pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
#             outputs.append({
#                 "Instruction": instruction,
#                 "input": sequence,
#                 "target": target.replace(" ; ", "; "),
#                 "predict": pred.replace(" ; ", "; ")
#             })
#     # Ensure output directory exists
#     os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            
#     with open(args.output_file, "w") as fout:
#         json.dump(outputs, fout, indent=2)


