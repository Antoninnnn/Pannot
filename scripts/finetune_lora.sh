#!/bin/bash
#SBATCH --job-name=pannot_finetune
#SBATCH --output=logs/pannot_finetune_%j.out
#SBATCH --error=logs/pannot_finetune_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --time=96:00:00




module purge
module load CUDA/11.8.0 Anaconda3
# eval "$(conda shell.bash hook)"
ml GCC/10.3.0 

export PATH=$SCRATCH/local/pdsh/bin:$PATH
# Set CUDA environment variables on Grace HPRC of TAMU
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to a writable location
export HF_HOME=$SCRATCH/hf_cache

#Set the Torch cache directory in the $SCRATCH

export TORCH_HOME=$SCRATCH/.cache/torch

# # Create the directory if it doesn't exist
# mkdir -p $TRANSFORMERS_CACHE


source activate pannot-dev

# # The reason I deactivate and activate again is that 
# # I want to make sure the python is used in the environment,
# # not the default python in the system.(in sw/...)
# # (the problem would occur when i activate and directly call python)
conda deactivate 

source activate pannot-dev


# Example: Pannot pretraining script (multimodal: protein sequence + structure)
# Be sure to set these environment variables or modify inline:

MODEL_VERSION=Meta-Llama-3.1-8B-Instruct
PROMPT_VERSION=plain

# Customize these:
DATA_PATH=$SCRATCH/TAMU/PhD/Pannot/data/opi/OPI_full_1.61M_train_converted.jsonl
# DATA_PATH=$SCRATCH/TAMU/PhD/Pannot/data/opi/OPI_full_1.61M_train_first_10000.json
PRET_MODEL_DIR=./checkpoints/pannot-${MODEL_VERSION}-pretrain-v00
SEQ_TOWER=ESM
STR_TOWER=ESMIF

echo "Running on $(hostname), node rank: $SLURM_NODEID, task rank: $SLURM_PROCID"
echo "Using model: ${MODEL_VERSION}, prompt: ${PROMPT_VERSION}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Using DeepSpeed config: ./scripts/zero3.json"


export WANDB_API_KEY=c6da89ba565a8b25f5b18c6fb722e7ad6637d4de  # from wandb.ai/settings
export WANDB_MODE=offline  # or remove this if online logging is available
export WANDB_DIR=$SCRATCH/wandb_logs

echo "[INFO] Writing hostfile:"
scontrol show hostnames $SLURM_NODELIST | sed 's/$/ slots=2/' > scripts/hostfile.txt

# export CUDA_LAUNCH_BLOCKING=1
deepspeed --hostfile ./scripts/hostfile.txt --num_gpus 2\
	pannot/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ${PRET_MODEL_DIR}/checkpoint-24000 \
    --pretrain_mm_mlp_adapter ${PRET_MODEL_DIR}/mm_projector.bin \
    --output_dir ./checkpoints/pannot-${MODEL_VERSION}-finetune-v00 \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
