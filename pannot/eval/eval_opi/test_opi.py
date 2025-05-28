import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from pannot.model.builder import load_pretrained_model
from pannot.mm_utils import tokenizer_protein_token
from pannot.utils import disable_torch_init


def load_test_data(jsonl_file):
    with open(jsonl_file, 'r') as f:
        return [json.loads(line) for line in f]

def build_prompt(instruction, sequence, model_config):
    seq_token = "<seq>" if not getattr(model_config, 'mm_use_seq_start_end', False) else "<s_seq><seq></s_seq>"
    prompt = f"{seq_token}\n{sequence}\n{instruction}"
    return prompt

def evaluate(model, tokenizer, test_data, device):
    predictions = []
    for instance in tqdm(test_data, desc="Evaluating"):
        sequence = instance['input']
        instruction = instance.get('instruction', '')
        reference = instance.get('output', '')

        prompt = build_prompt(instruction, sequence, model.config)

        input_ids = tokenizer_protein_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
        seq_tensor = torch.tensor(model.get_seq_tower().tokenizer.tokenize(sequence, add_special_tokens=False)).unsqueeze(0).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                inputs=input_ids,
                seqs=[seq_tensor],
                max_new_tokens=256,
                do_sample=False,
                use_cache=True
            )

        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        predictions.append({
            "input": sequence,
            "instruction": instruction,
            "reference": reference,
            "prediction": output
        })
    return predictions

def save_predictions(predictions, out_path):
    with open(out_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--use-flash-attn", action="store_true")
    args = parser.parse_args()

    disable_torch_init()
    model_name = args.model_path.split("/")[-1]
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name,
        use_flash_attn=args.use_flash_attn
    )

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_data = load_test_data(args.test_file)
    predictions = evaluate(model, tokenizer, test_data, device)
    save_predictions(predictions, args.out_file)

if __name__ == "__main__":
    main()