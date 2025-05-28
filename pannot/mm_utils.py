from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from pannot.constants import  SEQ_TOKEN_INDEX, STR_TOKEN_INDEX


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit




def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


import re
import pickle
import numpy as np
from biotite.structure import AtomArray, guess_element, filter_peptide_backbone

def load_structure_from_pkl(file_path: str, chain: str = None) -> AtomArray:
    """
    Load a preprocessed protein structure from a .pkl file into a biotite AtomArray.
    
    Parameters
    ----------
    file_path : str
        Path to the .pkl file containing preprocessed structure data.
    chain : str, optional
        Chain ID (e.g., "A") to filter atoms by chain.
    
    Returns
    -------
    AtomArray
        Biotite AtomArray containing backbone atoms (N, CA, C).
    """
    # Load preprocessed data
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    coords_all_atoms = data["atom_positions"]        # (L, 37, 3)
    atom_mask = data["atom_mask"]                    # (L, 37)
    aatype = data["aatype"]                          # (L,)
    res_idx = data["residue_index"]                  # (L,)
    chain_idx = data["chain_index"]                  # (L,)
    modeled_idx = data.get("modeled_idx", np.arange(len(aatype)))

    # Fixed order of 37 standard atom names
    fixed_atom_names = [
        "N", "CA", "C", "O", "CB",
        "CG", "CG1", "CG2", "CD", "CD1", "CD2",
        "CE", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2",
        "ND1", "ND2", "NE", "NE1", "NE2", "NH1", "NH2",
        "NZ", "OD1", "OD2", "OE1", "OE2", "OG", "OG1", "OH",
        "SD", "SG"
    ]

    # Collect atom-level data
    atom_name_list = []
    coord_list = []
    res_id_list = []
    chain_id_list = []

    for i in range(coords_all_atoms.shape[0]):
        res_atoms = coords_all_atoms[i]          # (37, 3)
        res_mask = atom_mask[i] > 0.0            # (37,)

        for j in range(37):
            if not res_mask[j]:
                continue
            atom_name_list.append(fixed_atom_names[j])
            coord_list.append(res_atoms[j])
            res_id_list.append(res_idx[i])
            chain_id_list.append(chr(65 + chain_idx[i]))  # assumes A-Z chains

    # Build AtomArray
    coords = np.array(coord_list)
    atom_names = np.array(atom_name_list)
    res_ids = np.array(res_id_list)
    chain_ids = np.array(chain_id_list)

    atoms = AtomArray(len(coords))
    atoms.coord = coords
    atoms.atom_name = atom_names
    atoms.res_id = res_ids
    atoms.chain_id = chain_ids
    atoms.element = guess_element(atom_names)

    # Optional chain filtering
    if chain is not None:
        atoms = atoms[atoms.chain_id == chain]

    # Keep only backbone atoms (N, CA, C)
    backbone_mask = filter_peptide_backbone(atoms)
    return atoms[backbone_mask]

    
def tokenizer_protein_token(prompt, tokenizer, seq_token_index=SEQ_TOKEN_INDEX, str_token_index=STR_TOKEN_INDEX, return_tensors=None):
    # Split the prompt on both <seq> and <str> while preserving the split tokens
    prompt_chunks = re.split(r'(<seq>|<str>)', prompt)

    # Tokenize the chunks and replace <seq> and <str> with their respective token indices
    tokenized_input = []
    for chunk in prompt_chunks:
        if chunk == '<seq>':
            tokenized_input.append(seq_token_index)
        elif chunk == '<str>':
            tokenized_input.append(str_token_index)
        else:
            # Tokenize the chunk normally
            tokenized_input.extend(tokenizer.encode(chunk, add_special_tokens=False))

    # If return_tensors is specified, return the result as a PyTorch tensor
    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(tokenized_input, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return tokenized_input


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
