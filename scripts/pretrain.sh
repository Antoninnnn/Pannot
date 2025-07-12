#!/bin/bash
#SBATCH --job-name=pannot_pretrain
#SBATCH --output=logs/pannot_pretrain_%j.out
#SBATCH --error=logs/pannot_pretrain_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=96G
#SBATCH --time=72:00:00




module purge
module load CUDA/11.8.0 Anaconda3
# eval "$(conda shell.bash hook)"

# For multi nodes training, you need to set the following packages :
ml GCC/10.3.0 

# # Create a build directory
# mkdir -p $SCRATCH/pdsh_build && cd $SCRATCH/pdsh_build

# # Download the latest pdsh source
# wget https://github.com/chaos/pdsh/releases/download/pdsh-2.34/pdsh-2.34.tar.gz

# # Extract it
# tar -xvzf pdsh-2.34.tar.gz
# cd pdsh-2.34

# # Configure installation prefix (choose your install path)
# ./configure --prefix=$SCRATCH/local/pdsh

# # Build and install
# make -j
# make install
# # ml OpenMPI/4.1.1

export PATH=$SCRATCH/local/pdsh/bin:$PATH

# Set CUDA environment variables on Grace HPRC of TAMU
export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to a writable location
export HF_HOME=$SCRATCH/hf_cache

#Set the Torch cache directory in the $SCRATCH

export TORCH_HOME=$SCRATCH/.cache/torch

# # # Set the Torch cache directory in the $SCRATCH for every node
# NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd, -)
# pdsh -R ssh -w $NODES "export TORCH_HOME=$SCRATCH/.cache/torch; echo \$HOSTNAME uses \$TORCH_HOME"


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
# DATA_PATH=$SCRATCH/TAMU/PhD/Pannot/data/opi/OPI_full_1.61M_train_first_10000.json
DATA_PATH=$SCRATCH/TAMU/PhD/Pannot/data/opi/OPI_full_1.61M_train_converted.jsonl
OUTPUT_DIR=./checkpoints/pannot-${MODEL_VERSION}-pretrain-v02
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
    --model_name_or_path local_pretrained_llm/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ${DATA_PATH} \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 25000 \
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
 
# export HOSTFILE=$(realpath ./scripts/hostfile.txt)

# mpirun -np 8 --hostfile $HOSTFILE \
#   --bind-to none --map-by slot \
#   -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
#   -mca pml ob1 -mca btl ^openib \
#   deepspeed --launcher=mpi \
#     --hostfile $HOSTFILE \
#     pannot/train/train_mem.py \
#     --model_name_or_path local_pretrained_llm/$MODEL_VERSION \
#     --version $PROMPT_VERSION \
#     --data_path ${DATA_PATH} \
#     --tune_mm_mlp_adapter True \
#     --bf16 True \
#     --output_dir ${OUTPUT_DIR} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --save_total_limit 1 \
#     --learning_rate 2e-3 \
#     --weight_decay 0.0 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --use_seq_tower True \
#     --mm_seq_tower $SEQ_TOWER \
#     --mm_seq_projector_type linear \
#     --mm_seq_select_layer -1 \
#     --mm_seq_select_feature "cls" \
#     --mm_seq_no_pooling True \
#     --use_str_tower True \
#     --mm_struc_tower $STR_TOWER \
#     --mm_str_projector_type linear \
#     --mm_str_select_layer -1 \
#     --mm_str_select_feature "residue"