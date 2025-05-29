import argparse
import torch

# Constants (if defined for Pannot multimodal tags)
from pannot.constants import (
    PROT_TOKEN_INDEX,
    DEFAULT_PROT_TOKEN,
    DEFAULT_PROT_PATCH_TOKEN,
    DEFAULT_PROT_START_TOKEN,
    DEFAULT_PROT_END_TOKEN,

    PROT_PLACEHOLDER,         # analogous to IMAGE_PLACEHOLDER

    SEQ_TOKEN_INDEX,             # analogous to IMAGE_TOKEN_INDEX
    DEFAULT_SEQ_TOKEN,           # analogous to DEFAULT_IMAGE_TOKEN
    DEFAULT_SEQ_START_TOKEN,     # analogous to DEFAULT_IM_START_TOKEN
    DEFAULT_SEQ_END_TOKEN,       # analogous to DEFAULT_IM_END_TOKEN

    DEFAULT_STR_TOKEN,
    STR_TOKEN_INDEX,
    DEFAULT_STR_PATCH_TOKEN,
    DEFAULT_STR_START_TOKEN,
    DEFAULT_STR_END_TOKEN,
)

# Conversation templates and separator logic
from pannot.conversation import conv_templates, SeparatorStyle

# Pretrained model loader
from pannot.model.builder import load_pretrained_model  

# Utilities
from pannot.utils import disable_torch_init               # same purpose: disable dropout, etc.
from pannot.mm_utils import (
    load_structure_from_pkl,

    tokenizer_protein_token,     # similar to tokenizer_image_token
    get_model_name_from_path      # if reused or overridden in pannot
)
import requests
# from PIL import Image
from io import BytesIO
import re


# def image_parser(args):
#     out = args.image_file.split(args.sep)
#     return out

def seq_parser(args):
    out = args.seq_file.split(args.sep)
    return out

def struc_parser(args):
    out = args.struc_file.split(args.sep)
    return out

def load_seq(seq_file):
    if seq_file.endswith(".fasta") or seq_file.endswith(".fa"):
        record = next(SeqIO.parse(seq_file, "fasta"))
        return str(record.seq)
    else:
        with open(seq_file, "r") as f:
            return f.read().strip()

# Load multiple sequences

def load_seqs(seq_files):
    return [load_seq(f) for f in seq_files]


# Load a single structure (as .npy or .json for coords/residue_index)

def load_struc(struc_file, chain=None):
    if struc_file.endswith(".npy"):
        data = np.load(struc_file, allow_pickle=True).item()
    elif struc_file.endswith(".json"):
        with open(struc_file, "r") as f:
            data = json.load(f)
    elif struc_file.endswith(".pkl"):
        atom_array = load_structure_from_pkl(struc_file, chain)
        data = {
            "coords": atom_array.coord.astype(np.float32),
            "residue_index": atom_array.res_id.astype(np.int64)
        }    
    else:
        raise ValueError("Unsupported structure file format")

    coords = torch.tensor(data["coords"]).float()
    residue_index = torch.tensor(data["residue_index"]).long()
    return {"coords": coords, "residue_index": residue_index}

# Load multiple structures

def load_strucs(struc_files):
    return [load_struc(f) for f in struc_files]



def eval_model(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, use_flash_attn=args.use_flash_attn if hasattr(args, 'use_flash_attn') else False
    )

    qs = args.query
    # image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    # if IMAGE_PLACEHOLDER in qs:
    #     if model.config.mm_use_im_start_end:
    #         qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
    #     else:
    #         qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    # else:
    #     if model.config.mm_use_im_start_end:
    #         qs = image_token_se + "\n" + qs
    #     else:
    #         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    seq_token_se = DEFAULT_SEQ_START_TOKEN + DEFAULT_SEQ_TOKEN + DEFAULT_SEQ_END_TOKEN
    str_token_se = DEFAULT_STR_START_TOKEN + DEFAULT_STR_TOKEN + DEFAULT_STR_END_TOKEN

    if SEQUENCE_PLACEHOLDER in qs:
        if getattr(model.config, 'mm_use_seq_start_end', False):
            qs = re.sub(SEQUENCE_PLACEHOLDER, seq_token_se, qs)
        else:
            qs = re.sub(SEQUENCE_PLACEHOLDER, DEFAULT_SEQ_TOKEN, qs)
    else:
        if getattr(model.config, 'mm_use_seq_start_end', False):
            qs = seq_token_se + "\n" + qs
        else:
            qs = DEFAULT_SEQ_TOKEN + "\n" + qs

    if DEFAULT_STR_TOKEN in qs:
        if getattr(model.config, 'mm_use_str_start_end', False):
            qs = qs.replace(DEFAULT_STR_TOKEN, str_token_se)
        # if mm_use_str_start_end is False, leave as is

    if "llama" in model_name.lower():
        conv_mode = "pannot_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "pannot_v1"
    # elif "mpt" in model_name.lower():
    #     conv_mode = "mpt"
    else:
        conv_mode = "pannot_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # image_files = image_parser(args)
    # images = load_images(image_files)
    # image_sizes = [x.size for x in images]
    # images_tensor = process_images(
    #     images,
    #     image_processor,
    #     model.config
    # ).to(model.device, dtype=torch.float16)

    # input_ids = (
    #     tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    #     .unsqueeze(0)
    #     .cuda()
    # )

    seq_files = seq_parser(args)
    sequences = load_seqs(seq_files)
    input_ids = tokenizer_protein_token(prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(model.device)
    seqs_tensor = [torch.tensor(model.get_seq_tower().tokenizer.tokenize(s, add_special_tokens=False)).to(model.device) for s in sequences]

    struc_files = struc_parser(args)
    strucs = load_strucs(struc_files) if struc_files else None
    strs_input = [torch.tensor(struc["coords"]).to(model.device) for struc in strucs]

    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            seqs=seqs_tensor,
            strs=strs_input,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--seq-file", type=str, required=True)
    parser.add_argument("--struc-file", type=str, default=None)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    eval_model(args)
