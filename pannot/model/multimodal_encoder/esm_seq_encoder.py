# esm_encoder.py

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PretrainedConfig

class ESMSeqTower(nn.Module):
    def __init__(
        self,
        model_name: str = 'facebook/esm2_t6_8M_UR50D',
        args=None,
        delay_load: bool = False,
        no_pooling: bool = True,
    ):
        super().__init__()
        self.is_loaded = False
        self.model_name = model_name
        self.args = args

        self.select_layer = getattr(args, 'mm_seq_select_layer', -1)
        self.pooling = getattr(args, 'mm_seq_select_feature', 'cls')  # 'cls' or 'mean'
        self.no_pooling = getattr(args, 'mm_seq_no_pooling', no_pooling)

        if not delay_load or getattr(args, 'unfreeze_mm_seq_tower', False):
            self.load_model()

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.model_name} is already loaded. Skipping load.')
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            output_hidden_states=True,
            device_map=device_map
        )
        self.encoder.requires_grad_(False)
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, input_ids, attention_mask):
        if not self.is_loaded:
            self.load_model()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(self.device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[self.select_layer]

        if self.no_pooling:
            return hidden_states  # (B, L, D)

        if self.pooling == 'cls':
            return hidden_states[:, 0, :]  # (B, D)
        elif self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
            sum_emb = torch.sum(hidden_states * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return sum_emb / counts
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

    def tokenize(self, sequences, return_tensors='pt', padding=True, truncation=True, max_length=1024):
        if not self.is_loaded:
            self.load_model()
        return self.tokenizer(
            sequences,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )

    @property
    def dummy_feature(self):
        if self.no_pooling:
            return torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.encoder.dtype if self.is_loaded else torch.get_default_dtype()

    @property
    def device(self):
        return next(self.encoder.parameters()).device if self.is_loaded else torch.device('cpu')

    @property
    def config(self):
        return self.encoder.config if self.is_loaded else PretrainedConfig.from_pretrained(self.model_name)

    @property
    def hidden_size(self):
        # the hidden size of the last layer is stored in the config of the model
        return self.config.hidden_size


# class ESMSeqEncoder(nn.Module):
#     def __init__(
#         self,
#         model_name: str = 'facebook/esm2_t6_8M_UR50D',
#         args=None,
#         delay_load: bool = False,
#         no_pooling: bool = False,   # NEW: return full per-residue embeddings?
#     ):
#         super().__init__()
#         self.is_loaded = False

#         self.model_name = model_name
#         self.select_layer = getattr(args, 'protein_select_layer', -1)
#         self.pooling = getattr(args, 'protein_pooling', 'cls')  # 'cls' or 'mean'
#         self.no_pooling = no_pooling  # NEW flag

#         if not delay_load:
#             self.load_model()

#     def load_model(self, device_map=None):
#         if self.is_loaded:
#             print(f'{self.model_name} is already loaded. Skipping load.')
#             return

#         # Load tokenizer and model
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
#         self.encoder = AutoModel.from_pretrained(
#             self.model_name,
#             device_map=device_map,
#             trust_remote_code=True,
#             output_hidden_states=True
#         )
#         # Freeze encoder weights by default
#         self.encoder.requires_grad_(False)

#         self.is_loaded = True

#     def tokenize(self, sequences):
#         return self.tokenizer(
#             sequences,
#             return_tensors='pt',
#             padding=True,
#             truncation=True,
#             max_length=1024
#         )

#     @torch.no_grad()
#     def forward(self, sequences):
#         if not self.is_loaded:
#             self.load_model()

#         # Tokenize & move to model device
#         inputs = self.tokenize(sequences)
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}

#         # Run through ESM, grab hidden states
#         outputs = self.encoder(**inputs)
#         # hidden_states is a tuple: (layer0, layer1, ..., layerN)
#         hidden_states = outputs.hidden_states[self.select_layer]  # (batch, seq_len, hidden_size)

#         if self.no_pooling:
#             # Return full sequence embeddings
#             return hidden_states

#         # Otherwise pool to single vector per sequence
#         if self.pooling == 'cls':
#             # CLS token is at position 0
#             features = hidden_states[:, 0, :]
#         elif self.pooling == 'mean':
#             mask = inputs['attention_mask'].unsqueeze(-1).expand_as(hidden_states)
#             sum_emb = torch.sum(hidden_states * mask, dim=1)
#             counts = mask.sum(dim=1).clamp(min=1e-9)
#             features = sum_emb / counts
#         else:
#             raise ValueError(f"Unsupported pooling type: {self.pooling}")

#         return features

#     @property
#     def dtype(self):
#         if not self.is_loaded:
#             # If not loaded, infer from config (usually fp32)
#             return torch.get_default_dtype()
#         return self.encoder.dtype

#     @property
#     def device(self):
#         if not self.is_loaded:
#             return torch.device('cpu')
#         # encoder.device may be a map for multi-GPU; pick first
#         dev = next(self.encoder.parameters()).device
#         return dev

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.encoder.config
#         return PretrainedConfig.from_pretrained(self.model_name)

#     @property
#     def hidden_size(self):
#         return self.config.hidden_size

#     @property
#     def dummy_feature(self):
#         """
#         Returns a zero tensor matching the shape of the output:
#         - (1, seq_len, hidden_size) if sequence_output, else (1, hidden_size)
#         Note: seq_len = 1 for dummy by default.
#         """
#         if self.no_pooling:
#             # dummy single residue embedding
#             return torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)
