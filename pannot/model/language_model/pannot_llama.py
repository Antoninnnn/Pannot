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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..pannot_arch import PannotMetaModel, PannotMetaForCausalLM


class PannotConfig(LlamaConfig):
    model_type = "pannot_llama"

    ## Llama3 's config has a rope_scaling need to be patched, it can not be dealt with transformers lib(v4.37.0)
    def __init__(self, **kwargs):
        # Patch rope_scaling before initializing
        rope_scaling = kwargs.get("rope_scaling", None)
        if isinstance(rope_scaling, dict) and "type" not in rope_scaling:
            kwargs["rope_scaling"] = {
                "type": "linear",
                "factor": rope_scaling.get("factor", 1.0)
            }
        super().__init__(**kwargs)

class PannotLlamaModel(PannotMetaModel, LlamaModel):
    config_class = PannotConfig

    def __init__(self, config: LlamaConfig):
        super(PannotLlamaModel, self).__init__(config)


class PannotLlamaForCausalLM(LlamaForCausalLM, PannotMetaForCausalLM):
    config_class = PannotConfig


    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = PannotLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        seqs: Optional[List[torch.Tensor]] = None,
        strs: Optional[List[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seq_input_ids: Optional[List[torch.Tensor]] = None,
        seq_attention_mask: Optional[List[torch.Tensor]] = None,
        struc_coords: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # only build inputs_embeds once
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids =input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                # seqs=seqs,
                # strs=strs,
                seqs=seq_input_ids,
                seq_attention_mask=seq_attention_mask,
                strs=struc_coords,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.LongTensor] = None,
    #     seqs: Optional[List[torch.Tensor]] = None,
    #     strs: Optional[List[torch.Tensor]] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #     # pop out any kwargs to avoid collision
    #     position_ids = kwargs.pop("position_ids", None)
    #     attention_mask = kwargs.pop("attention_mask", None)
    #     if "inputs_embeds" in kwargs:
    #         raise NotImplementedError("`inputs_embeds` is not supported for PannotLlama")

    #     # build inputs_embeds from seqs/strs
    #     if seqs is not None or strs is not None:
    #         (
    #             inputs,
    #             position_ids,
    #             attention_mask,
    #             _,
    #             inputs_embeds,
    #             _
    #         ) = self.prepare_inputs_labels_for_multimodal(
    #             inputs,
    #             position_ids,
    #             attention_mask,
    #             None,
    #             None,
    #             seqs=seqs,
    #             strs=strs,
    #         )
    #     else:
    #         inputs_embeds = self.get_model().embed_tokens(inputs)

    #     return super().generate(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         **kwargs
    #     )
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.LongTensor] = None,
        seqs: Optional[List[torch.Tensor]] = None,
        seq_attention_mask: Optional[List[torch.Tensor]] = None,
        strs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Pop out any kwargs to avoid collision
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for PannotLlama")

        # Build inputs_embeds and optionally input_ids
        if seqs is not None or strs is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                seqs=seqs,
                seq_attention_mask=seq_attention_mask,
                strs=strs,
            )
        else:
            assert inputs is not None, "`inputs` must be provided when no seqs/strs"
            input_ids = inputs
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # 🔐 Sanity check for token indices
        vocab_size = self.get_model().embed_tokens.num_embeddings
        print(inputs)
        print(input_ids)
        # if input_ids.max().item() >= vocab_size:
        #     raise ValueError(f"Found token ID >= vocab size ({vocab_size}). Likely invalid tokenization.")

        return super().generate(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,  # ✅ crucial to prevent HF from guessing input_ids
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        # carry seqs/strs through beam search, etc.
        seqs = kwargs.pop("seqs", None)
        strs = kwargs.pop("strs", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if seqs is not None:
            inputs["seqs"] = seqs
        if strs is not None:
            inputs["strs"] = strs
        return inputs

AutoConfig.register("pannot_llama", PannotConfig)
AutoModelForCausalLM.register(PannotConfig, PannotLlamaForCausalLM)
