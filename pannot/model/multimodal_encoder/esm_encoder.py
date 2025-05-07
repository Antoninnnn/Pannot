# esm_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PretrainedConfig

class ESMEncoder(nn.Module):
    def __init__(self, model_name: str = 'facebook/esm2_t6_8M_UR50D', args=None, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False

        self.model_name = model_name
        self.select_layer = getattr(args, 'protein_select_layer', -1)  # Default to last layer
        self.pooling = getattr(args, 'protein_pooling', 'cls')  # Options: 'cls' or 'mean'

        if not delay_load:
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.model_name} is already loaded. Skipping load.')
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(self.model_name, device_map=device_map, trust_remote_code=True)
        self.encoder.requires_grad_(False)

        self.is_loaded = True

    def tokenize(self, sequences):
        return self.tokenizer(
            sequences,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024
        )

    @torch.no_grad()
    def forward(self, sequences):
        inputs = self.tokenize(sequences)
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}

        outputs = self.encoder(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[self.select_layer]

        if self.pooling == 'cls':
            features = hidden_states[:, 0]  # CLS token
        elif self.pooling == 'mean':
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * attention_mask, dim=1)
            sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            features = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        return features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.encoder.dtype

    @property
    def device(self):
        return self.encoder.device

    @property
    def config(self):
        return self.encoder.config if self.is_loaded else PretrainedConfig.from_pretrained(self.model_name)

    @property
    def hidden_size(self):
        return self.config.hidden_size