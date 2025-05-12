import torch
import torch.nn as nn
import esm
from esm.inverse_folding.util import extract_coords_from_structure, get_encoder_output

from typing import Optional
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure as get_structure_cif
from biotite.structure.io.pdb import PDBFile, get_structure as get_structure_pdb
import biotite.structure as struc  # for filter_peptide_backbone

def load_structure(
    file_path: str,
    chain: Optional[str] = None,
    model: int = 1
) -> AtomArray:
    """
    Load a protein structure from .cif/.mmcif or .pdb, select one model & chain,
    then filter to peptide backbone atoms only.
    """
    ext = file_path.split('.')[-1].lower()
    # Read & convert to AtomArray
    if ext in ("cif", "mmcif"):
        cif    = CIFFile.read(file_path)
        struct = get_structure_cif(cif, model=model)
    elif ext == "pdb":
        pdb    = PDBFile.read(file_path)
        struct = get_structure_pdb(pdb, model=model)
    else:
        raise ValueError(f"Unsupported extension '.{ext}'")

    # Optional chain selection
    if chain is not None:
        struct = struct[struct.chain_id == chain]

    # **Filter to peptide backbone (drops waters, side-chains, non-standard residues)**
    backbone_mask = struc.filter_peptide_backbone(struct)
    struct = struct[backbone_mask]

    return struct

class ESMIFEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "esm_if1_gvp4_t16_142M_UR50",
        args=None,
        delay_load: bool = False,
        no_pooling: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.no_pooling = no_pooling
        self.is_loaded = False

        if not delay_load:
            self.load_model()

    def load_model(self):
        if self.is_loaded:
            print(f"{self.model_name} already loaded. Skipping.")
            return

        # Load the inverse-folding model and its alphabet
        model, alphabet = getattr(esm.pretrained, self.model_name)()
        model = model.eval().requires_grad_(False)
        self.model = model
        self.alphabet = alphabet
        self.is_loaded = True

    @torch.no_grad()
    def forward(self, structure_path: str, chain: str = None):
        if not self.is_loaded:
            self.load_model()

        # 1) Load and filter backbone atoms / select chain
        structure = load_structure(structure_path, chain)

        # 2) Extract (L × 3 × 3) coords tensor + sequence string
        coords, seq = extract_coords_from_structure(structure)

        # 3) Convert coords to torch tensor
        coords_tensor = torch.tensor(coords, dtype=torch.float32)

        # 4) Run the inverse-folding model
        encoder_out = get_encoder_output(self.model, self.alphabet, coords_tensor)
        # embeddings = encoder_out["representations"]  # (L, hidden_size)
        embeddings = encoder_out
        

        if self.no_pooling:
            # Return per-residue (1, L, hidden_size)
            return embeddings.unsqueeze(0)
        else:
            # Mean pool over L residues → (1, hidden_size)
            return embeddings.mean(dim=0, keepdim=True)

    @property
    def device(self):
        if not self.is_loaded:
            return torch.device("cpu")
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        if not self.is_loaded:
            return torch.get_default_dtype()
        return next(self.model.parameters()).dtype

    @property
    def hidden_size(self):
        if not self.is_loaded:
            self.load_model()
        return self.model.embed_dim

    @property
    def dummy_feature(self):
        """
        - If no_pooling=True: returns (1,1,hidden_size)
        - Else: (1,hidden_size)
        """
        if self.no_pooling:
            return torch.zeros(1, 1, self.hidden_size,
                               device=self.device, dtype=self.dtype)
        else:
            return torch.zeros(1, self.hidden_size,
                               device=self.device, dtype=self.dtype)