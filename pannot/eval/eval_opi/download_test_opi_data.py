from huggingface_hub import hf_hub_download
import os
import shutil

# List of relative file paths to download
file_paths = [
    "OPI_DATA/SU/EC_number/test/CLEAN_EC_number_price_test.jsonl",
    "OPI_DATA/SU/EC_number/test/CLEAN_EC_number_new_test.jsonl",
    "OPI_DATA/SU/Subcellular_localization/test/subcell_loc_test.jsonl",
    "OPI_DATA/SU/Fold_type/test/fold_type_test.jsonl",
    "OPI_DATA/AP/Function/test/UniProtSeq_function_test.jsonl",
    "OPI_DATA/AP/Function/test/IDFilterSeq_function_test.jsonl",
    "OPI_DATA/AP/Function/test/CASPSimilarSeq_function_test.jsonl",
    "OPI_DATA/AP/GO/test/UniProtSeq_go_test.jsonl",
    "OPI_DATA/AP/GO/test/IDFilterSeq_go_test.jsonl",
    "OPI_DATA/AP/GO/test/CASPSimilarSeq_go_test.jsonl",
    "OPI_DATA/AP/Keywords/test/UniProtSeq_keywords_test.jsonl",
    "OPI_DATA/AP/Keywords/test/IDFilterSeq_keywords_test.jsonl",
    "OPI_DATA/AP/Keywords/test/CASPSimilarSeq_keywords_test.jsonl",
    "OPI_DATA/KM/gName2Cancer/test/gene_name_to_cancer_test.jsonl",
    "OPI_DATA/KM/gSymbol2Cancer/test/gene_symbol_to_cancer_test.jsonl",
    "OPI_DATA/KM/gSymbol2Tissue/test/gene_symbol_to_tissue_test.jsonl"
]

# Your target directory on TAMU Grace
target_dir = "/scratch/user/yining_yang/TAMU/PhD/Pannot/data/opi/OPI_DATA"
os.makedirs(target_dir, exist_ok=True)

# Download and copy files preserving structure
for file_path in file_paths:
    # Download the file
    local_path = hf_hub_download(
        repo_id="BAAI/OPI",
        filename=file_path,
        repo_type="dataset"
    )

    # Compute destination path (preserve full relative path)
    relative_path = os.path.relpath(file_path, "OPI_DATA")
    dest_path = os.path.join(target_dir, relative_path)

    # Ensure target directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Copy the file
    shutil.copy(local_path, dest_path)
    print(f"Copied: {file_path} -> {dest_path}")