import torch
import torch.nn as nn
import esm
from esm.inverse_folding.util import extract_coords_from_structure, get_encoder_output

from typing import Optional
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure as get_structure_cif
from biotite.structure.io.pdb import PDBFile, get_structure as get_structure_pdb
import biotite.structure as struc  # for filter_peptide_backbone



### Usage
#tower = ESMIFTower("esm_if1_gvp4_t16_142M_UR50", args=config)

# # Preprocess manually
# coords = tower.structure_processor("4hhb.pdb", chain="A")  # → (L, 3, 3)

# # Forward tensor
# features = tower(coords.to("cuda"))  # (1, D) or (1, L, D)

class ESMIFTower(nn.Module):
    def __init__(self, model_name: str = "esm_if1_gvp4_t16_142M_UR50", args=None, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.structure_tower_name = model_name
        self.select_layer = getattr(args, 'mm_str_select_layer', -1)
        self.select_feature = getattr(args, 'mm_str_select_feature', 'mean')  # 'mean' or 'residue'

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_str_tower', False):
            self.load_model()
        else:
            self.cfg_only = {"model_name": self.structure_tower_name}

    def load_model(self):
        if self.is_loaded:
            print(f"{self.structure_tower_name} is already loaded. Skipping.")
            return

        model, alphabet = getattr(esm.pretrained, self.structure_tower_name)()
        self.structure_tower = model.eval().requires_grad_(False)
        self.structure_processor = self._build_structure_processor()
        self.alphabet = alphabet
        self.is_loaded = True

    def _build_structure_processor(self):
        def processor(file_path: str, chain: Optional[str] = None, model: int = 1):
            ext = file_path.split('.')[-1].lower()
            if ext in ("cif", "mmcif"):
                cif = CIFFile.read(file_path)
                structure = get_structure_cif(cif, model=model)
            elif ext == "pdb":
                pdb = PDBFile.read(file_path)
                structure = get_structure_pdb(pdb, model=model)
            else:
                raise ValueError(f"Unsupported file extension '.{ext}'")

            if chain is not None:
                structure = structure[structure.chain_id == chain]

            backbone_mask = struc.filter_peptide_backbone(structure)
            structure = structure[backbone_mask]

            coords, _ = extract_coords_from_structure(structure)
            return torch.tensor(coords, dtype=torch.float32)  # (L, 3, 3)
        return processor

    def feature_select(self, encoder_out: torch.Tensor) -> torch.Tensor:
        if self.select_feature == "mean":
            return encoder_out.mean(dim=0, keepdim=True)  # (1, D)
        elif self.select_feature == "residue":
            return encoder_out.unsqueeze(0)  # (1, L, D)
        else:
            raise ValueError(f"Unexpected select_feature: {self.select_feature}")

    @torch.no_grad()
    def forward(self, coords):
        """
        Args:
            coords: torch.Tensor of shape (L, 3, 3) or List[Tensor] for batch
        Returns:
            torch.Tensor: (1, D) or (1, L, D) depending on `select_feature`
        """
        if not self.is_loaded:
            self.load_model()

        if isinstance(coords, list):
            outputs = []
            for coord in coords:
                coord = coord.to(device=self.device, dtype=self.dtype)
                enc_out = get_encoder_output(self.structure_tower, self.alphabet, coord)
                outputs.append(self.feature_select(enc_out).to(coord.dtype))
            return torch.cat(outputs, dim=0)
        else:
            coords = coords.to(device=self.device, dtype=self.dtype)
            enc_out = get_encoder_output(self.structure_tower, self.alphabet, coords)
            return self.feature_select(enc_out).to(coords.dtype)

    @property
    def dummy_feature(self):
        if self.select_feature == "residue":
            return torch.zeros(1, 1, self.hidden_size, device=self.device, dtype=self.dtype)
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.structure_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.structure_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return {"hidden_size": self.hidden_size}
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.structure_tower.embed_dim


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