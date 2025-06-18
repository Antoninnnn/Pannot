#!/bin/bash
#SBATCH --job-name=pannot_opi-eval
#SBATCH --output=logs/pannot_eval_opi_%j.out
#SBATCH --error=logs/pannot_eval_opi_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=96:00:00





module purge
module load CUDA/11.8.0 Anaconda3

ml GCC/10.3.0


export PATH=$SCRATCH/local/pdsh/bin:$PATH

export LD_LIBRARY_PATH=$EBROOTCUDA/lib64:$LD_LIBRARY_PATH

# Set Hugging Face cache directory to a writable location
export HF_HOME=$SCRATCH/hf_cache

#Set the Torch cache directory in the $SCRATCH

export TORCH_HOME=$SCRATCH/.cache/torch
source activate pannot-dev
conda deactivate

source activate pannot-dev



CUDA_VISIBLE_DEVICES=0 python pannot/eval/eval_opi/eval_opi_dataset_pannot.py \
   --input_file /scratch/user/yining_yang/TAMU/PhD/Pannot/data/opi/OPI_DATA/AP/Keywords/test/UniProtSeq_keywords_test.jsonl\
   --output_file /scratch/user/yining_yang/TAMU/PhD/Pannot/results/pretrained/UniProtSeq_keywords_test.json \
   --model-path /scratch/user/yining_yang/TAMU/PhD/Pannot/checkpoints/pannot-Meta-Llama-3.1-8B-Instruct-pretrain-v00 \
   --model-base /scratch/user/yining_yang/TAMU/PhD/Pannot/checkpoints/pannot-Meta-Llama-3.1-8B-Instruct-pretrain-v00/checkpoint-24000 \
   --temperature 0.2 \
   --top_p 0.9 \
   --num_beams 1 \
   --max_new_tokens 256
