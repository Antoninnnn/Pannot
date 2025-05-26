# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch

import transformers
import tokenizers

from pannot.constants import IGNORE_INDEX, SEQ_TOKEN_INDEX, STR_TOKEN_INDEX, DEFAULT_SEQ_TOKEN, DEFAULT_STR_TOKEN, DEFAULT_SEQ_START_TOKEN, DEFAULT_SEQ_END_TOKEN, DEFAULT_STR_START_TOKEN, DEFAULT_STR_END_TOKEN
from torch.utils.data import Dataset
from pannot.train.pannot_trainer import PannotTrainer

from pannot import conversation as conversation_lib
from pannot.model import *
from pannot.mm_utils import tokenizer_protein_token

from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    # Core model
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")

    # Global control
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)


    # ===== SEQUENCE TOWER =====
    use_seq_tower: bool = field(default=True)
    mm_seq_tower: Optional[str] = field(default="ESM")  # One of: "ProtST", "ESM"
    mm_seq_select_layer: Optional[int] = field(default=-1)
    mm_seq_select_feature: Optional[str] = field(default=None)  # 👈 NEW
    mm_seq_projector_type: Optional[str] = field(default="linear")
    mm_use_seq_start_end: bool = field(default=False)
    mm_use_seq_patch_token: bool = field(default=False)
    mm_seq_no_pooling: bool = field(default=False)  # 👈 NEW

    # ===== STRUCTURE TOWER =====
    use_str_tower: bool = field(default=True)
    mm_struc_tower: Optional[str] = field(default="ESMIF")  # One of: "ESMIF", "ESM3"
    mm_str_select_layer: Optional[int] = field(default=-1)  # 👈 NEW
    mm_str_select_feature: Optional[str] = field(default=None)  # 👈 NEW
    mm_str_projector_type: Optional[str] = field(default="linear")
    mm_use_str_start_end: bool = field(default=False)
    mm_use_str_patch_token: bool = field(default=False)
    # ===== Fusion control (optional) =====
    mm_fusion_type: Optional[str] = field(default="concat")  # e.g., "concat", "sum", "crossattn"
    
    # # original llava args
    # model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    # version: Optional[str] = field(default="v0")
    # freeze_backbone: bool = field(default=False)
    # tune_mm_mlp_adapter: bool = field(default=False)
    # vision_tower: Optional[str] = field(default=None)
    # mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    # pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # mm_projector_type: Optional[str] = field(default='linear')
    # mm_use_im_start_end: bool = field(default=False)
    # mm_use_im_patch_token: bool = field(default=True)
    # mm_patch_merge_type: Optional[str] = field(default='flat')
    # mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sequence_folder: Optional[str] = field(default=None)

    structure_folder: Optional[str] = field(default=None)
    # image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    """
    Find all Linear module names for LoRA injection, excluding multimodal adapters.
    Adapted for Pannot: skips sequence/structure towers and projectors.
    """
    cls = torch.nn.Linear
    lora_module_names = set()

    # Keywords to exclude: protein-specific adapters (Pannot)
    multimodal_keywords = [
        'mm_projector', 
        'mm_seq_projector', 'mm_str_projector',
        'seq_resampler', 'str_resampler',
        'seq_tower', 'str_tower',
       
    ]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            parts = name.split('.')
            # Add only the final component (e.g., 'q_proj') for LoRA target_modules
            lora_module_names.add(parts[-1])

    if 'lm_head' in lora_module_names:  # typically excluded from LoRA
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_seq_projector', 'mm_str_projector', 'seq_resampler', 'str_resampler']

        # if getattr(trainer.args, "use_im_start_end", False):
        #     keys_to_match.extend(['embed_tokens', 'embed_in'])
        if getattr(trainer.args, "use_seq_start_end", False):
            keys_to_match.extend(['seq_embed_tokens', 'seq_embed_in'])

        if getattr(trainer.args, "use_str_start_end", False):
            keys_to_match.extend(['str_embed_tokens', 'str_embed_in'])


        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    if not data_args.is_multimodal:
        return sources

    use_seq_start_end = getattr(data_args, "use_seq_start_end", False)
    use_str_start_end = getattr(data_args, "use_str_start_end", False)

    for source in sources:
        for sentence in source:
            value = sentence['value']

            # Handle <seq> tokens
            if DEFAULT_SEQ_TOKEN in value:
                value = value.replace(DEFAULT_SEQ_TOKEN, '').strip()
                value = DEFAULT_SEQ_TOKEN + '\n' + value
                if use_seq_start_end:
                    value = value.replace(
                        DEFAULT_SEQ_TOKEN,
                        DEFAULT_SEQ_START_TOKEN + DEFAULT_SEQ_TOKEN + DEFAULT_SEQ_END_TOKEN
                    )

            # Handle <str> tokens
            if DEFAULT_STR_TOKEN in value:
                value = value.replace(DEFAULT_STR_TOKEN, '').strip()
                value = DEFAULT_STR_TOKEN + '\n' + value
                if use_str_start_end:
                    value = value.replace(
                        DEFAULT_STR_TOKEN,
                        DEFAULT_STR_START_TOKEN + DEFAULT_STR_TOKEN + DEFAULT_STR_END_TOKEN
                    )

            sentence['value'] = value.strip()

    return sources


# def preprocess_llama_2(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

#     # Mask targets
#     sep = "[/INST] "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# def preprocess_v1(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()

#     assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

#     # Mask targets
#     sep = conv.sep + conv.roles[1] + ": "
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep2)
#         cur_len = 1
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 2

#             if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
#                 round_len -= 1
#                 instruction_len -= 1

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# def preprocess_mpt(
#     sources,
#     tokenizer: transformers.PreTrainedTokenizer,
#     has_image: bool = False
# ) -> Dict:
#     conv = conversation_lib.default_conversation.copy()
#     roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

#     # Apply prompt templates
#     conversations = []
#     for i, source in enumerate(sources):
#         if roles[source[0]["from"]] != conv.roles[0]:
#             # Skip the first one if it is not from human
#             source = source[1:]

#         conv.messages = []
#         for j, sentence in enumerate(source):
#             role = roles[sentence["from"]]
#             assert role == conv.roles[j % 2], f"{i}"
#             conv.append_message(role, sentence["value"])
#         conversations.append(conv.get_prompt())

#     # Tokenize conversations

#     if has_image:
#         input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
#     else:
#         input_ids = tokenizer(
#             conversations,
#             return_tensors="pt",
#             padding="longest",
#             max_length=tokenizer.model_max_length,
#             truncation=True,
#         ).input_ids

#     targets = input_ids.clone()
#     assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

#     # Mask targets
#     sep = conv.sep + conv.roles[1]
#     for conversation, target in zip(conversations, targets):
#         total_len = int(target.ne(tokenizer.pad_token_id).sum())

#         rounds = conversation.split(conv.sep)
#         re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
#         for conv_idx in range(3, len(rounds), 2):
#             re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
#         cur_len = 0
#         target[:cur_len] = IGNORE_INDEX
#         for i, rou in enumerate(re_rounds):
#             if rou == "":
#                 break

#             parts = rou.split(sep)
#             if len(parts) != 2:
#                 break
#             parts[0] += sep

#             if has_image:
#                 round_len = len(tokenizer_image_token(rou, tokenizer))
#                 instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
#             else:
#                 round_len = len(tokenizer(rou).input_ids)
#                 instruction_len = len(tokenizer(parts[0]).input_ids) - 1

#             if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
#                 round_len += 1
#                 instruction_len += 1

#             target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

#             cur_len += round_len
#         target[cur_len:] = IGNORE_INDEX

#         if cur_len < tokenizer.model_max_length:
#             if cur_len != total_len:
#                 target[:] = IGNORE_INDEX
#                 print(
#                     f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
#                     f" (ignored)"
#                 )

#     return dict(
#         input_ids=input_ids,
#         labels=targets,
#     )


# def preprocess_plain(
#     sources: Sequence[str],
#     tokenizer: transformers.PreTrainedTokenizer,
# ) -> Dict:
#     # add end signal and concatenate together
#     conversations = []
#     for source in sources:
#         assert len(source) == 2
#         assert DEFAULT_IMAGE_TOKEN in source[0]['value']
#         source[0]['value'] = DEFAULT_IMAGE_TOKEN
#         conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
#         conversations.append(conversation)
#     # tokenize conversations
#     input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
#     targets = copy.deepcopy(input_ids)
#     for target, source in zip(targets, sources):
#         tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
#         target[:tokenized_len] = IGNORE_INDEX

#     return dict(input_ids=input_ids, labels=targets)

def preprocess_llama_2_protein(sources, tokenizer, has_protein=True) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_protein:
        input_ids = torch.stack(
            [tokenizer_protein_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(conversations, return_tensors="pt", padding="longest",
                              max_length=tokenizer.model_max_length, truncation=True).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2
    sep = "[/INST] "

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if not rou:
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_protein:
                round_len = len(tokenizer_protein_token(rou, tokenizer))
                instr_len = len(tokenizer_protein_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instr_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len:cur_len + instr_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        # if cur_len != total_len:
        #     target[:] = IGNORE_INDEX
        #     print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")


        if abs(cur_len - total_len) > 2:
            print(f"[W] Mismatch ignored: cur_len={cur_len}, total_len={total_len}")


    return dict(input_ids=input_ids, labels=targets)

def preprocess_v1_protein(sources, tokenizer, has_protein=True) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_protein:
        input_ids = torch.stack(
            [tokenizer_protein_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(conversations, return_tensors="pt", padding="longest",
                              max_length=tokenizer.model_max_length, truncation=True).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    sep = conv.sep + conv.roles[1] + ": "

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, rou in enumerate(rounds):
            if not rou:
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_protein:
                round_len = len(tokenizer_protein_token(rou, tokenizer))
                instr_len = len(tokenizer_protein_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instr_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len:cur_len + instr_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        # if cur_len != total_len:
        #     target[:] = IGNORE_INDEX
        #     print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")


        if abs(cur_len - total_len) > 2:
            print(f"[W] Mismatch ignored: cur_len={cur_len}, total_len={total_len}")

    return dict(input_ids=input_ids, labels=targets)


def preprocess_mpt_protein(sources, tokenizer, has_protein=True) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = []

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_protein:
        input_ids = torch.stack(
            [tokenizer_protein_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(conversations, return_tensors="pt", padding="longest",
                              max_length=tokenizer.model_max_length, truncation=True).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT
    sep = conv.sep + conv.roles[1]

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for k in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[k:k+2]))

        cur_len = 0
        for i, rou in enumerate(re_rounds):
            if not rou:
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_protein:
                round_len = len(tokenizer_protein_token(rou, tokenizer))
                instr_len = len(tokenizer_protein_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instr_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len:cur_len + instr_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        # if cur_len != total_len:
        #     target[:] = IGNORE_INDEX
        #     print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")


        if abs(cur_len - total_len) > 2:
            print(f"[W] Mismatch ignored: cur_len={cur_len}, total_len={total_len}")

    return dict(input_ids=input_ids, labels=targets)


def preprocess_plain_protein(
    sources: Sequence[Dict[str, str]],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_SEQ_TOKEN in source[0]['value'] or DEFAULT_STR_TOKEN in source[0]['value'], \
            "Expected <seq> or <str> in the input."

        # Construct conversation string
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # Tokenize each prompt with protein tokenizer
    input_ids = [tokenizer_protein_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]

    # Clone to labels
    targets = copy.deepcopy(input_ids)

    # Mask the prompt (first part) in the target
    for target, source in zip(targets, sources):
        prompt_len = len(tokenizer_protein_token(source[0]['value'], tokenizer))
        target[:prompt_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)

    
def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_protein: bool = True  # replaces has_image
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # Dispatch to conversation-style specific preprocessors
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain_protein(sources, tokenizer)

    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2_protein(sources, tokenizer, has_protein=has_protein)

    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1_protein(sources, tokenizer, has_protein=has_protein)

    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt_protein(sources, tokenizer, has_protein=has_protein)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    
    def get_tokenize_len(prompts):
        return [len(tokenizer_protein_token(prompt, tokenizer)) for prompt in prompts]

        # Tokenize full conversations
    if has_protein:
        input_ids = [
            tokenizer_protein_token(prompt, tokenizer, return_tensors='pt')
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)

    for target, source in zip(targets, sources):
        if has_protein:
            # Compute token lengths for header and each utterance using protein-aware tokenizer
            prompts = [f"{conversation_lib.default_conversation.system}\n\n"] + [s["value"] for s in source]
            tokenized_lens = [
                len(tokenizer_protein_token(p, tokenizer))
                for p in prompts
            ]
        else:
            tokenized_lens = _tokenize_fn(
                [f"{conversation_lib.default_conversation.system}\n\n"] + [s["value"] for s in source],
                tokenizer
            )["input_ids_lens"]


    # if has_image:
    #     input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    # else:
    #     conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    #     input_ids = conversations_tokenized["input_ids"]

    # targets = copy.deepcopy(input_ids)
    # for target, source in zip(targets, sources):
    #     if has_image:
    #         tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
    #     else:
    #         tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
    #     speakers = [sentence["from"] for sentence in source]
    #     _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedProteinDataset(Dataset):
    """Protein multimodal dataset for instruction tuning."""


    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 seq_tower=None,
                 struc_tower=None):
        super().__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.seq_tower = seq_tower  # Should have .tokenize()
        self.struc_tower = struc_tower  # Should have .structure_processor()

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return [
            sum(len(conv['value'].split()) for conv in sample['conversations']) +
            (128 if 'seq' in sample or 'str' in sample else 0)
            for sample in self.list_data_dict
        ]

    @property
    def modality_lengths(self):
        lengths = []
        for sample in self.list_data_dict:
            base_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if 'seq' in sample or 'str' in sample:
                lengths.append(base_len)
            else:
                lengths.append(-base_len)
        return lengths

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.list_data_dict[idx]
        conversations = copy.deepcopy(sample["conversations"])
        # Step 1: Insert <seq>/<str> tokens
        sources = preprocess_multimodal([conversations], self.data_args)

        # Step 2: Tokenize conversation for decoder
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_protein=('sequence' in sample or 'structure_path' in sample)
        )
        data_dict = {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0]
        }

        # Step 3: Protein sequence processing
        if "sequence" in sample and self.seq_tower is not None:
            seq_tokenized = self.seq_tower.tokenize([sample["sequence"]],
                                                    return_tensors='pt', padding=True, truncation=True)
            data_dict["seq_input_ids"] = seq_tokenized["input_ids"][0]
            data_dict["seq_attention_mask"] = seq_tokenized["attention_mask"][0]

        # Step 4: Structure preprocessing (L, 3, 3)
        if "structure_path" in sample and self.struc_tower is not None:
            try:
                coords = self.struc_tower.structure_processor(
                    sample["structure_path"],
                    chain=sample.get("structure_chain", "A")
                )
                data_dict["struc_coords"] = coords  # tensor, will be moved in collator
            except Exception as e:
                print(f"[WARN] Structure loading failed for idx {idx}: {e}")
                data_dict["struc_coords"] = None
        return data_dict

@dataclass
class DataCollatorForSupervisedProteinDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        labels = [instance["labels"] for instance in instances]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }

        # Handle optional protein sequence features
        if "seq_input_ids" in instances[0]:
            seq_input_ids = [inst["seq_input_ids"] for inst in instances]
            seq_attention_mask = [inst["seq_attention_mask"] for inst in instances]
            batch["seq_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                seq_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch["seq_attention_mask"] = torch.nn.utils.rnn.pad_sequence(
                seq_attention_mask, batch_first=True, padding_value=0
            )

        # Handle structure features (pad to max L)
        if "struc_coords" in instances[0]:
            coords_list = []
            max_len = max((inst["struc_coords"].shape[0] if inst["struc_coords"] is not None else 0)
                          for inst in instances)

            for inst in instances:
                coord = inst["struc_coords"]
                if coord is None:
                    padded = torch.full((max_len, 3, 3), float("nan"))
                else:
                    pad_len = max_len - coord.shape[0]
                    padded = torch.nn.functional.pad(coord, (0, 0, 0, 0, 0, pad_len), value=float("nan"))
                coords_list.append(padded)

            batch["struc_coords"] = torch.stack(coords_list)

        return batch

# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances]
#                                   for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids,
#             batch_first=True,
#             padding_value=self.tokenizer.pad_token_id)
#         labels = torch.nn.utils.rnn.pad_sequence(labels,
#                                                  batch_first=True,
#                                                  padding_value=IGNORE_INDEX)
#         input_ids = input_ids[:, :self.tokenizer.model_max_length]
#         labels = labels[:, :self.tokenizer.model_max_length]
#         batch = dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )

#         if 'image' in instances[0]:
#             images = [instance['image'] for instance in instances]
#             if all(x is not None and x.shape == images[0].shape for x in images):
#                 batch['images'] = torch.stack(images)
#             else:
#                 batch['images'] = images

#         return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedProteinDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedProteinDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_seq_projector", "mm_str_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.mm_seq_tower is not None or model_args.mm_str_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = PannotMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = PannotLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # if model_args.vision_tower is not None:
    #     model.get_model().initialize_vision_modules(
    #         model_args=model_args,
    #         fsdp=training_args.fsdp
    #     )
        
    #     vision_tower = model.get_vision_tower()
    #     vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

    #     data_args.image_processor = vision_tower.image_processor
    #     data_args.is_multimodal = True

    #     model.config.image_aspect_ratio = data_args.image_aspect_ratio
    #     model.config.tokenizer_padding_side = tokenizer.padding_side
    #     model.config.tokenizer_model_max_length = tokenizer.model_max_length

    #     model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    #     if model_args.tune_mm_mlp_adapter:
    #         model.requires_grad_(False)
    #         for p in model.get_model().mm_projector.parameters():
    #             p.requires_grad = True

    #     model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    #     if training_args.freeze_mm_mlp_adapter:
    #         for p in model.get_model().mm_projector.parameters():
    #             p.requires_grad = False

    #     if training_args.bits in [4, 8]:
    #         model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    #     model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    #     model.config.mm_projector_lr = training_args.mm_projector_lr
    #     training_args.use_im_start_end = model_args.mm_use_im_start_end
    #     model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    #     model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # if training_args.bits in [4, 8]:
    #     from peft.tuners.lora import LoraLayer
    #     for name, module in model.named_modules():
    #         if isinstance(module, LoraLayer):
    #             if training_args.bf16:
    #                 module = module.to(torch.bfloat16)
    #         if 'norm' in name:
    #             module = module.to(torch.float32)
    #         if 'lm_head' in name or 'embed_tokens' in name:
    #             if hasattr(module, 'weight'):
    #                 if training_args.bf16 and module.weight.dtype == torch.float32:
    #                     module = module.to(torch.bfloat16)
    if model_args.use_seq_tower:
        model.get_model().initialize_seq_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

    # Initialize structure tower
    if model_args.use_str_tower:
        model.get_model().initialize_str_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

    # Add <seq_start>, <seq_end>, etc.
    model.config.mm_use_seq_start_end = data_args.mm_use_seq_start_end = model_args.mm_use_seq_start_end
    model.config.mm_use_str_start_end = data_args.mm_use_str_start_end = model_args.mm_use_str_start_end
    training_args.use_seq_start_end = model_args.mm_use_seq_start_end
    training_args.use_str_start_end = model_args.mm_use_str_start_end

    # Set lr and control gradients
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.mm_projector_lr = training_args.mm_projector_lr

    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if any(k in name for k in ["mm_seq_projector", "mm_str_projector", "seq_resampler", "str_resampler"]):
                param.requires_grad = True

    # Apply LoRA
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bf16:
            model.to(torch.bfloat16)
        elif training_args.fp16:
            model.to(torch.float16)

        model = get_peft_model(model, lora_config)

    # Final quantization casting
    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if 'norm' in name:
                module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight') and compute_dtype == torch.bfloat16:
                    module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = PannotTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
