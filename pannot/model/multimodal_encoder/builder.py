import os
# from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .clip_encoder import ProtSTTower

from .esm_seq_encoder import ESMSeqTower
from .esmif_encoder import ESMIFTower

def build_seq_tower(seq_tower_cfg, **kwargs):
    seq_tower = getattr(seq_tower_cfg, 'mm_seq_tower', getattr(seq_tower_cfg, 'seq_tower', None))
    if seq_tower == 'ProtST':
        return ProtSTTower(model_name='DeepGraphLearning/ProtST', args=seq_tower_cfg, **kwargs)
    elif seq_tower == 'ESM':
        return ESMSeqTower(model_name='facebook/esm2_t33_650M_UR50D', args=seq_tower_cfg, **kwargs)

    raise ValueError(f'Unknown sequence encoder: {seq_tower}')


def build_struc_tower(struc_tower_cfg, **kwargs):
    struc_tower = getattr(struc_tower_cfg, 'mm_struc_tower', getattr(struc_tower_cfg, 'struc_tower', None))
    if struc_tower == 'ESMIF':
        return ESMIFTower(model_name='esm_if1_gvp4_t16_142M_UR50', args=struc_tower_cfg, **kwargs)
    # elif struc_tower == 'ESM':
    #     return ESMStructEncoder(model_name='facebook/esm2_t33_650M_UR50D', args=struc_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown structure encoder: {struc_tower}')
# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
#     is_absolute_path_exists = os.path.exists(vision_tower)
#     use_s2 = getattr(vision_tower_cfg, 's2', False)
#     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
#         if use_s2:
#             return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
#         else:
#             return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')

