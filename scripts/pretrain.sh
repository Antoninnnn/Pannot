#!/bin/bash
#SBATCH --job-name=pannot_pretrain
#SBATCH --output=logs/pannot_pretrain_%j.out
#SBATCH --error=logs/pannot_pretrain_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --time=72:00:00




module purge
module load CUDA/11.8.0 Anaconda3
# eval "$(conda shell.bash hook)"

# Set CUDA environment variables on Grace HPRC of TAMU
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to a writable location
export HF_HOME=$SCRATCH/hf_cache

# # Create the directory if it doesn't exist
# mkdir -p $TRANSFORMERS_CACHE


source activate pannot-dev

# # The reason I deactivate and activate again is that 
# # I want to make sure the python is used in the environment,
# # not the one in the system.
# # (the problem would occur when i activate and directly call python)
conda deactivate 

source activate pannot-dev


# Example: Pannot pretraining script (multimodal: protein sequence + structure)
# Be sure to set these environment variables or modify inline:

MODEL_VERSION=Meta-Llama-3.1-8B-Instruct
PROMPT_VERSION=plain

# Customize these:
DATA_PATH=$SCRATCH/TAMU/PhD/Pannot/data/opi/OPI_full_1.61M_train_first_10000.json
OUTPUT_DIR=./checkpoints/pannot-${MODEL_VERSION}-pretrain-v00
SEQ_TOWER=ESM
STR_TOWER=ESMIF

echo "Running on $(hostname), node rank: $SLURM_NODEID, task rank: $SLURM_PROCID"
echo "Using model: ${MODEL_VERSION}, prompt: ${PROMPT_VERSION}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Using DeepSpeed config: ./scripts/zero2.json"

deepspeed pannot/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path local_pretrained_llm/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --use_seq_tower True \
    --mm_seq_tower $SEQ_TOWER \
    --mm_seq_projector_type linear \
    --mm_seq_select_layer -1 \
    --mm_seq_select_feature "cls"\
    --mm_seq_no_pooling True \
    --use_str_tower True \
    --mm_struc_tower $STR_TOWER \
    --mm_str_projector_type linear \
    --mm_str_select_layer -1 \
    --mm_str_select_feature "residue"
