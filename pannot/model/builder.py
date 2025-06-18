#    Copyright 2023 Haotian Liu
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
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from pannot.model import *
from pannot.constants import IGNORE_INDEX, SEQ_TOKEN_INDEX, DEFAULT_SEQ_TOKEN, DEFAULT_SEQ_PATCH_TOKEN ,DEFAULT_SEQ_START_TOKEN ,DEFAULT_SEQ_END_TOKEN ,STR_TOKEN_INDEX ,DEFAULT_STR_TOKEN ,DEFAULT_STR_PATCH_TOKEN ,DEFAULT_STR_START_TOKEN ,DEFAULT_STR_END_TOKEN 




def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'


        # ─── PANNOT CASE ───
    if "pannot" in model_name.lower():
        # Choose the correct class
        if "mpt" in model_name.lower():
            ModelClass = PannotMptForCausalLM
        elif "mistral" in model_name.lower():
            ModelClass = PannotMistralForCausalLM
        else:
            ModelClass = PannotLlamaForCausalLM

        tokenizer = None
        model = None

        # 1) LoRA‐off‐base path for any Pannot variant
        if "lora" in model_name.lower() and model_base is not None:
            # load the delta config
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            # instantiate base with that config
            print("loading pannot from base model")
            model = ModelClass.from_pretrained(
                model_base, config=cfg, low_cpu_mem_usage=True, **kwargs
            )

            # If vocab changed, reinit embedding rows
            tok_num, emb_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != tok_num:
                model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(tok_num, emb_dim, device=model.device, dtype=model.dtype)
                )
                model.get_model().embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(tok_num, emb_dim, device=model.device, dtype=model.dtype)
                )

            from peft import PeftModel

            # load & merge the LoRA adapter
            print(f"→ Applying LoRA delta from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            model = model.merge_and_unload()

        # 2) Projector‐only path (you provided mm_*_projector bins)
        elif model_base is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            # model = ModelClass.from_pretrained(
            #     model_base, config=cfg, low_cpu_mem_usage=True, **kwargs
            # )
            
            try:
                model = ModelClass.from_pretrained(
                    model_base, config=cfg, low_cpu_mem_usage=True, **kwargs
                )
            except ValueError as e:
                print("[Warning] model.from_pretrained failed due to shape mismatch.")
                print("Error:", e)
                print("Attempting to patch vocab size and reload...")

                # Adjust vocab_size in config to match tokenizer
                cfg.vocab_size = len(tokenizer)

                model = ModelClass.from_pretrained(
                    model_base, config=cfg, low_cpu_mem_usage=True, **kwargs
                )

                # Resize embedding layers to match tokenizer
                model.resize_token_embeddings(len(tokenizer))

                # Initialize added tokens if any
                embedding = model.get_input_embeddings().weight.data
                out_embedding = model.get_output_embeddings().weight.data
                added = len(tokenizer) - embedding.shape[0]
                if added > 0:
                    print(f"Resizing: {added} new tokens initialized with mean embedding")
                    embedding[-added:] = embedding[:-added].mean(dim=0, keepdim=True)
                    out_embedding[-added:] = out_embedding[:-added].mean(dim=0, keepdim=True)

            # optionally load seq/str projector weights if they exist
            seq_proj = os.path.join(model_path, "mm_seq_projector.bin")
            if os.path.isfile(seq_proj):
                weights = torch.load(seq_proj, map_location="cpu")
                model.load_state_dict(weights, strict=False)
            str_proj = os.path.join(model_path, "mm_str_projector.bin")
            if os.path.isfile(str_proj):
                weights = torch.load(str_proj, map_location="cpu")
                model.load_state_dict(weights, strict=False)

        # 3) Plain from‐pretrained
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = ModelClass.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

        # ─── Add seq/str tokens & resize embeddings ───
        to_add = []
        cfg = model.config

        if getattr(cfg, "mm_use_seq_patch_token", False):
            to_add.append(DEFAULT_SEQ_PATCH_TOKEN)
        if getattr(cfg, "mm_use_seq_start_end", False):
            to_add += [DEFAULT_SEQ_START_TOKEN, DEFAULT_SEQ_END_TOKEN]

        if getattr(cfg, "mm_use_str_patch_token", False):
            to_add.append(DEFAULT_STR_PATCH_TOKEN)
        if getattr(cfg, "mm_use_str_start_end", False):
            to_add += [DEFAULT_STR_START_TOKEN, DEFAULT_STR_END_TOKEN]

        if to_add:
            tokenizer.add_tokens(to_add, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))

        # ─── (Re)initialize towers & projectors if using delay_load ───
        if hasattr(model, "initialize_seq_modules"):
            model.initialize_seq_modules(cfg, fsdp=None)
        if hasattr(model, "initialize_str_modules"):
            model.initialize_str_modules(cfg, fsdp=None)

        context_len = getattr(cfg, "max_sequence_length", 2048)
        return tokenizer, model, None, context_len

    # if 'llava' in model_name.lower():
    #     # Load LLaVA model
    #     if 'lora' in model_name.lower() and model_base is None:
    #         warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
    #     if 'lora' in model_name.lower() and model_base is not None:
    #         from llava.model.language_model.llava_llama import LlavaConfig
    #         lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #         print('Loading LLaVA from base model...')
    #         model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
    #         token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    #         if model.lm_head.weight.shape[0] != token_num:
    #             model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #             model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    #         print('Loading additional LLaVA weights...')
    #         if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
    #             non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    #         else:
    #             # this is probably from HF Hub
    #             from huggingface_hub import hf_hub_download
    #             def load_from_hf(repo_id, filename, subfolder=None):
    #                 cache_file = hf_hub_download(
    #                     repo_id=repo_id,
    #                     filename=filename,
    #                     subfolder=subfolder)
    #                 return torch.load(cache_file, map_location='cpu')
    #             non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    #         non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    #         if any(k.startswith('model.model.') for k in non_lora_trainables):
    #             non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    #         model.load_state_dict(non_lora_trainables, strict=False)

    #         from peft import PeftModel
    #         print('Loading LoRA weights...')
    #         model = PeftModel.from_pretrained(model, model_path)
    #         print('Merging LoRA weights...')
    #         model = model.merge_and_unload()
    #         print('Model is loaded...')
    #     elif model_base is not None:
    #         # this may be mm projector only
    #         print('Loading LLaVA from base model...')
    #         if 'mpt' in model_name.lower():
    #             if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
    #                 shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
    #             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
    #             cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    #             model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #             cfg_pretrained = AutoConfig.from_pretrained(model_path)
    #             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

    #         mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
    #         mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
    #         model.load_state_dict(mm_projector_weights, strict=False)
    #     else:
    #         if 'mpt' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #             model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    #         elif 'mistral' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path)
    #             model = LlavaMistralForCausalLM.from_pretrained(
    #                 model_path,
    #                 low_cpu_mem_usage=True,
    #                 **kwargs
    #             )
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #             model = LlavaLlamaForCausalLM.from_pretrained(
    #                 model_path,
    #                 low_cpu_mem_usage=True,
    #                 **kwargs
    #             )
    # else:
    #     # Load language model
    #     if model_base is not None:
    #         # PEFT model
    #         from peft import PeftModel
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #         model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    #         print(f"Loading LoRA weights from {model_path}")
    #         model = PeftModel.from_pretrained(model, model_path)
    #         print(f"Merging weights")
    #         model = model.merge_and_unload()
    #         print('Convert to FP16...')
    #         model.to(torch.float16)
    #     else:
    #         use_fast = False
    #         if 'mpt' in model_name.lower():if 'llava' in model_name.lower():
    #     # Load LLaVA model
    #     if 'lora' in model_name.lower() and model_base is None:
    #         warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
    #     if 'lora' in model_name.lower() and model_base is not None:
    #         from llava.model.language_model.llava_llama import LlavaConfig
    #         lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #         print('Loading LLaVA from base model...')
    #         model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
    #         token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
    #         if model.lm_head.weight.shape[0] != token_num:
    #             model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
    #             model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    #         print('Loading additional LLaVA weights...')
    #         if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
    #             non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    #         else:
    #             # this is probably from HF Hub
    #             from huggingface_hub import hf_hub_download
    #             def load_from_hf(repo_id, filename, subfolder=None):
    #                 cache_file = hf_hub_download(
    #                     repo_id=repo_id,
    #                     filename=filename,
    #                     subfolder=subfolder)
    #                 return torch.load(cache_file, map_location='cpu')
    #             non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
    #         non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    #         if any(k.startswith('model.model.') for k in non_lora_trainables):
    #             non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    #         model.load_state_dict(non_lora_trainables, strict=False)

    #         from peft import PeftModel
    #         print('Loading LoRA weights...')
    #         model = PeftModel.from_pretrained(model, model_path)
    #         print('Merging LoRA weights...')
    #         model = model.merge_and_unload()
    #         print('Model is loaded...')
    #     elif model_base is not None:
    #         # this may be mm projector only
    #         print('Loading LLaVA from base model...')
    #         if 'mpt' in model_name.lower():
    #             if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
    #                 shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
    #             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
    #             cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    #             model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #             cfg_pretrained = AutoConfig.from_pretrained(model_path)
    #             model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

    #         mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
    #         mm_projector_weights = {k: v.to(torch.float1    # else:
    #     # Load language model
    #     if model_base is not None:
    #         # PEFT model
    #         from peft import PeftModel
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #         model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    #         print(f"Loading LoRA weights from {model_path}")
    #         model = PeftModel.from_pretrained(model, model_path)
    #         print(f"Merging weights")
    #         model = model.merge_and_unload()
    #         print('Convert to FP16...')
    #         model.to(torch.float16)
    #     else:
    #         use_fast = False
    #         if 'mpt' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)6) for k, v in mm_projector_weights.items()}
    #         model.load_state_dict(mm_projector_weights, strict=False)
    #     else:
    #         if 'mpt' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #             model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    #         elif 'mistral' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path)
    #             model = LlavaMistralForCausalLM.from_pretrained(
    #                 model_path,
    #                 low_cpu_mem_usage=True,
    #                 **kwargs
    #             )
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #             model = LlavaLlamaForCausalLM.from_pretrained(
    #                 model_path,
    #                 low_cpu_mem_usage=True,
    #                 **kwargs
    #             )
    # else:
    #     # Load language model
    #     if model_base is not None:
    #         # PEFT model
    #         from peft import PeftModel
    #         tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    #         model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    #         print(f"Loading LoRA weights from {model_path}")
    #         model = PeftModel.from_pretrained(model, model_path)
    #         print(f"Merging weights")
    #         model = model.merge_and_unload()
    #         print('Convert to FP16...')
    #         model.to(torch.float16)
    #     else:
    #         use_fast = False
    #         if 'mpt' in model_name.lower():
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # image_processor = None

    # if 'llava' in model_name.lower():
    #     mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    #     mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    #     if mm_use_im_patch_token:
    #         tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #     if mm_use_im_start_end:
    #         tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    #     model.resize_token_embeddings(len(tokenizer))

    #     vision_tower = model.get_vision_tower()
    #     if not vision_tower.is_loaded:
    #         vision_tower.load_model(device_map=device_map)
    #     if device_map != 'auto':
    #         vision_tower.to(device=device_map, dtype=torch.float16)
    #     image_processor = vision_tower.image_processor

    # if hasattr(model.config, "max_sequence_length"):
    #     context_len = model.config.max_sequence_length
    # else:
    #     context_len = 2048

    # return tokenizer, model, image_processor, context_len
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
    #         else:
    #             tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    #             model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    # image_processor = None

    # if 'llava' in model_name.lower():
    #     mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    #     mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    #     if mm_use_im_patch_token:
    #         tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    #     if mm_use_im_start_end:
    #         tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    #     model.resize_token_embeddings(len(tokenizer))

    #     vision_tower = model.get_vision_tower()
    #     if not vision_tower.is_loaded:
    #         vision_tower.load_model(device_map=device_map)
    #     if device_map != 'auto':
    #         vision_tower.to(device=device_map, dtype=torch.float16)
    #     image_processor = vision_tower.image_processor

    # if hasattr(model.config, "max_sequence_length"):
    #     context_len = model.config.max_sequence_length
    # else:
    #     context_len = 2048

    # return tokenizer, model, image_processor, context_len
