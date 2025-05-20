import torch
import torch.nn as nn
from typing import Optional
from biotite.structure.io.pdbx import CIFFile, get_structure as get_structure_cif
from biotite.structure.io.pdb import PDBFile, get_structure as get_structure_pdb
import biotite.structure as struc
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.vqvae import StructureTokenEncoder


## Usage
# tower = ESM3StructureTower(weight_path="data/weights/esm3_structure_encoder_v0.pth", device="cuda")

# # Extract coords and residue_index from a PDB file
# coords, residue_index = tower.structure_processor("4hhb.pdb", chain_id="A")

# # Make sure they are batch-shaped: (1, L, ...)
# coords = coords.unsqueeze(0)
# residue_index = residue_index.unsqueeze(0)

# # Forward pass
# embedding = tower(coords, residue_index)  # shape: (1, 1, D) or (1, L, D)


class ESM3StructureTower(nn.Module):
    def __init__(
        self,
        weight_path: str = "esm3_structure_encoder_v0.pth",
        device: torch.device = torch.device("cpu"),
        args=None,
        delay_load=False
    ):
        super().__init__()
        self.device = device
        self.weight_path = weight_path
        self.select_feature = getattr(args, 'mm_str_select_feature', 'mean')  # mean or residue

        self.is_loaded = False
        if not delay_load or getattr(args, 'unfreeze_mm_str_tower', False):
            self.load_model()
        self.structure_processor = self._build_structure_processor()

    def load_model(self):
        if self.is_loaded:
            print(f"ESM3 Structure Encoder already loaded.")
            return

        self.structure_encoder = (
            StructureTokenEncoder(d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096)
            .to(self.device)
            .eval()
        )

        state_dict = torch.load(self.weight_path, map_location=self.device)
        self.structure_encoder.load_state_dict(state_dict)
        self.is_loaded = True

    def _build_structure_processor(self):
        def processor(pdb_path: str, chain_id: Optional[str] = None):
            chain = ProteinChain.from_pdb(pdb_path, chain_id=chain_id)
            coords, plddt, residue_index = chain.to_structure_encoder_inputs()
            return coords, residue_index  # both are torch.Tensors
        return processor

    def feature_select(self, x: torch.Tensor) -> torch.Tensor:
        if self.select_feature == "mean":
            return x.mean(dim=1, keepdim=True)  # (1, 1, D)
        elif self.select_feature == "residue":
            return x  # (1, L, D)
        else:
            raise ValueError(f"Unsupported feature selection: {self.select_feature}")

    @torch.no_grad()
    def forward(self, coords: torch.Tensor, residue_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape (1, L, 3, 3)
            residue_index: Tensor of shape (1, L)
        Returns:
            Tensor of shape (1, L, D) or (1, 1, D)
        """
        coords = coords.to(self.device)
        residue_index = residue_index.to(self.device)
        reprs, _ = self.structure_encoder.encode(coords, residutower = ESM3StructureTower(weight_path="data/weights/esm3_structure_encoder_v0.pth", device="cuda")

# # Extract coords and residue_index from a PDB file
# coords, residue_index = tower.structure_processor("4hhb.pdb", chain_id="A")

# # Make sure they are batch-shaped: (1, L, ...)
# coords = coords.unsqueeze(0)
# residue_index = residue_index.unsqueeze(0)

# # Forward pass
# embedding = tower(coords, residue_index)  # shape: (1, 1, D) or (1, L, D)

_index=residue_index)  # (1, L, D)
        return self.feature_select(reprs)

    @property
    def dummy_feature(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.structure_encoder.parameters()).dtype

    @property
    def hidden_size(self):
        return self.structure_encoder.d_out