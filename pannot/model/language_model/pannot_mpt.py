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


from typing import Optional, Tuple

import torch

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MptConfig, MptForCausalLM, MptModel


from ..pannot_arch import PannotMetaModel, PannotMetaForCausalLM


class PannotMptConfig(MptConfig):
    model_type = "pannot_mpt"


class PannotMptModel(PannotMetaModel, MptModel):
    config_class = PannotMptConfig

    def __init__(self, config: MptConfig):
        config.hidden_size = config.d_model
        super(PannotMptModel, self).__init__(config)
    
    def embed_tokens(self, x):
        return self.wte(x)


class PannotMptForCausalLM(MptForCausalLM, PannotMetaForCausalLM):
    config_class = PannotMptConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MptForCausalLM, self).__init__(config)
        self.transformer = PannotMptModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PannotMptModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seqs: Optional[List[torch.Tensor]] = None,
        strs: Optional[List[torch.Tensor]] = None,
    ):
        # Build inputs_embeds (and updated masks/labels) if not provided
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    labels=labels,
                    seqs=seqs,
                    strs=strs
                )

        return super().forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        seqs: Optional[List[torch.Tensor]] = None,
        strs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # pop out any existing embeddings or masks
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for PannotMpt")

        # build inputs_embeds from seqs/strs if provided
        if seqs is not None or strs is not None:
            input_ids, position_ids, attention_mask, _, inputs_embeds, _ = \
                self.prepare_inputs_labels_for_multimodal(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    labels=None,
                    seqs=seqs,
                    strs=strs
                )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        return super().generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        # carry seqs/strs through generator state
        seqs = kwargs.pop("seqs", None)
        strs = kwargs.pop("strs", None)
        model_kwargs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if seqs is not None:
            model_kwargs["seqs"] = seqs
        if strs is not None:
            model_kwargs["strs"] = strs
        return model_kwargs


AutoConfig.register("pannot_mpt", PannotMptConfig)
AutoModelForCausalLM.register(PannotMptConfig, PannotMptForCausalLM)