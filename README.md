# Pannot : Protein Language Understanding Interface

This project aims at protein multimodal inputs and natural language output. It was adapted from the llava project (https://github.com/haotian-liu/LLaVA)

## Release


-[05/24] Pannot model and basic training module is tested

## Command to set up the environment 


### For Grace HPRC(TAMU)
```
module purge

ml CUDA/11.8.0 Anaconda3

<!-- module load GCC/12.3.0 CUDA/11.8.0 Anaconda3 --> ## if you want to use the predefined module of grace, you can use this line(caution: there would probably be version problem!)
```

### Set the conda virtual environment
```
conda create -n pannot-dev python=3.10 -y
source activate pannot-dev
pip install --upgrade pip  # enable PEP 660 support
pip install -e . # install the package defined in pyproject.toml

# You would need torch-geometric for protein structure processing(Used in GVP Module)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install torch-geometric

```

problem solving : I(Yining) met some problem for the torch with cuda 11.8. The final solution is using `pip uninstall torch torchvision`, and call `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118` separately

### Install train

```bash
pip install -e .[train]
pip install "flash-attn<=2.5.6" --no-build-isolation
```

For multi-node training you need to install `pdsh` as the default launcher for deepspeed:

✅ Fix: Rebuild pdsh with SSH support
You need to explicitly enable the ssh module during the configure step. Here's how:

🔁 Step-by-step fix:
bash
Copy code
##### Remove the old build (optional)
```bash
rm -rf $SCRATCH/pdsh_build && mkdir -p $SCRATCH/pdsh_build
cd $SCRATCH/pdsh_build
```
##### Download source
```bash
wget https://github.com/chaos/pdsh/releases/download/pdsh-2.34/pdsh-2.34.tar.gz
tar -xzf pdsh-2.34.tar.gz && cd pdsh-2.34
```
##### Configure with SSH support
```bash
./configure --prefix=$SCRATCH/local/pdsh --with-ssh
```

##### Build and install
```bash
make -j && make install
```
Then add this to your environment:

```bash
export PATH=$SCRATCH/local/pdsh/bin:$PATH
```

## Model
test code on Grace
```
srun --partition=gpu --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G --time=02:00:00 --pty bash
```

If you need to pretrain on multi-nodes cluster, you would probably need pdsh:
```
conda install -c conda-forge pdsh
```

## Training

### Stage 1

```
deepspeed --hostfile ./scripts/hostfile.txt --num_gpus 2    pannot/train/train_mem.py     --deepspeed ./scripts/zero2.json     --model_name_or_path local_pretrained_llm/$MODEL_VERSION     --version $PROMPT_VERSION     --data_path ${DATA_PATH}     --tune_mm_mlp_adapter True     --bf16 True     --output_dir ${OUTPUT_DIR}     --num_train_epochs 1     --per_device_train_batch_size 4     --per_device_eval_batch_size 4     --gradient_accumulation_steps 1     --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 24000     --save_total_limit 1     --learning_rate 2e-3     --weight_decay 0.0     --warmup_ratio 0.03     --lr_scheduler_type "cosine"     --logging_steps 1     --tf32 True     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb     --use_seq_tower True     --mm_seq_tower $SEQ_TOWER     --mm_seq_projector_type linear     --mm_seq_select_layer -1     --mm_seq_select_feature "cls"    --mm_seq_no_pooling True     --use_str_tower True     --mm_struc_tower $STR_TOWER     --mm_str_projector_type linear     --mm_str_select_layer -1     --mm_str_select_feature "residue" 

```


TO upload the local wandb record 
```
wandb sync offline-run-20250528_034900-dcplvyd3
```

### Stage2




## Data interface 

### OPI
The OPI dataset is a protein language dataset. The data is in json format. The data is stored in a json file. The original json file called `OPI_XXXXX.json` is converted to a jsonl file called `OPI_XXXXX_converted.jsonl` using the following command:
```
jq -c '.[]' $DATA_PATH > ${DATA_PATH%.json}_converted.jsonl
```
This operation removed '[]' from the json file.


For dataset used by our model, we would use `LazySupervisedProteinDataset` and `DataCollatorForSupervisedDataset` to load the data. The data is stored in a list of dictionaries. Each dictionary contains the following keys:
The data is stored in a list of dictionaries. Each dictionary contains the following keys:

- `input_ids`: A list of integers representing the input tokens.
- `labels`: A list of integers representing the labels.
- `attention_mask`: A list of integers representing the attention mask.
- `seq_input_ids`: A list of integers representing the sequence input tokens.
- `seq_attention_mask`: A list of integers representing the sequence attention mask.
- `struct_coords`: A list of lists of integers representing the coordinates of the protein structure.

The data is stored in a json file, with format as list of dictionaries. Each dictionary contains the following keys:



##### Frames For natural language:
input_ids
labels
attention_mask

##### Frames For sequence input:
seq_input_ids
seq_attention_mask

##### Frames For strucuture input:
struct_coords


#### Inference
Create a process for interactive inference
```shell
srun --partition=gpu --gres=gpu:a100:1 --nodes=1 --ntasks=2 --cpus-per-task=4 --mem=96G --time=08:00:00 --pty bash
```

```shell
module purge
module load CUDA/11.8.0 Anaconda3
# eval "$(conda shell.bash hook)"


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

```
Then call `python` to test the inference
```python
from transformers import AutoTokenizer
from pannot.model.language_model.pannot_llama import PannotLlamaForCausalLM
import torch


from pannot.mm_utils import (
    load_structure_from_pkl,

    tokenizer_protein_token,     # similar to tokenizer_image_token
    get_model_name_from_path      # if reused or overridden in pannot
)


from pannot.model.builder import load_pretrained_model  


model_path="/scratch/user/yining_yang/TAMU/PhD/Pannot/checkpoints/pannot-Meta-Llama-3.1-8B-Instruct-finetune-lora-v00"
model_base = "/scratch/user/yining_yang/TAMU/PhD/Pannot/checkpoints/pannot-Meta-Llama-3.1-8B-Instruct-pretrain-v00/checkpoint-24000"
model_name = get_model_name_from_path(model_path)
tokenizer, model, _, context_len = load_pretrained_model(
        model_path, model_base, model_name, use_flash_attn=True
    )

model.eval()
model_device = next(model.parameters()).device
model_dtype = next(model.parameters()).dtype
if hasattr(model, 'get_seq_tower'):
        model.get_seq_tower().to(model_device)


eos_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"] if tokenizer.convert_tokens_to_ids(t) is not None]
pad_token_id = tokenizer.pad_token_id or eos_token_ids[0]

stopping_keywords = ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"]

outputs = []


instruction = "What is the functional description of the protein sequence?"
sequence="MSTEGGGRRCQAQVSRRISFSASHRLYSKFLSDEENLKLFGKCNNPNGHGHNYKVVVTVHGEIDPATGMVMNLADLKKYMEEAIMQPLDHKNLDMDVPYFADVVSTTENVAVYIWDNLQKVLPVGVLYKVKVYETDNNIVVYKGE"

target = "Involved in the biosynthesis of tetrahydrobiopterin, an essential cofactor of aromatic amino acid hydroxylases. Catalyzes the transformation of 7,8-dihydroneopterin triphosphate into 6-pyruvoyl tetrahydropterin."

PROT_TOKEN_INDEX = -300
DEFAULT_PROT_TOKEN = "<prot>"
DEFAULT_PROT_PATCH_TOKEN = "<prot_patch>"
DEFAULT_PROT_START_TOKEN = "<prot_start>"
DEFAULT_PROT_END_TOKEN = "<prot_end>"
PROT_PLACEHOLDER = "<prot-placeholder>"

SEQ_TOKEN_INDEX = -330
DEFAULT_SEQ_TOKEN = "<seq>"
DEFAULT_SEQ_PATCH_TOKEN = "<seq_patch>"
DEFAULT_SEQ_START_TOKEN = "<seq_start>"
DEFAULT_SEQ_END_TOKEN = "<seq_end>"

SEQUENCE_PLACEHOLDER = '<seq-placeholder>'

seq_token_se = DEFAULT_SEQ_START_TOKEN + DEFAULT_SEQ_TOKEN + DEFAULT_SEQ_END_TOKEN

if SEQUENCE_PLACEHOLDER in instruction:
                prompt = instruction.replace(SEQUENCE_PLACEHOLDER, seq_token_se)
else:
                prompt = seq_token_se + "\n" + instruction


from pannot.conversation import conv_templates


if "llama" in model_name.lower():
        conv_mode = "pannot_llama_2"
elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
elif "v1" in model_name.lower():
        conv_mode = "pannot_v1"
else:
        conv_mode = "pannot_v0"

conv = conv_templates[conv_mode].copy()
conv.append_message(conv.roles[0], prompt)
conv.append_message(conv.roles[1], None)
full_prompt = conv.get_prompt()


tokenized_prompt = tokenizer_protein_token(full_prompt, tokenizer, return_tensors="pt").unsqueeze(0)
input_ids = tokenized_prompt.to(model.device)
attention_mask = torch.ones_like(input_ids)

if hasattr(model, 'get_seq_tower'):
        tokenized_seq = model.get_seq_tower().tokenize(sequence)
        seq_input_id = tokenized_seq["input_ids"].squeeze(0).to(model.device)
        seq_attention_mask = tokenized_seq["attention_mask"].squeeze(0).to(model.device)
        seqs = [seq_input_id]
        seq_attention_masks = [seq_attention_mask]
else:
        seqs = None
        seq_attention_masks = None


from pannot.eval.eval_opi.eval_opi_dataset_pannot import KeywordsStoppingCriteria


stopping_criteria = [KeywordsStoppingCriteria(stopping_keywords, tokenizer, input_ids)]
temperature = 0.5
top_p = 0.75
num_beams = 1
max_new_tokens = 256
with torch.no_grad():
    output_ids = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    seqs=seqs,
                    seq_attention_mask=seq_attention_masks,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    eos_token_id=eos_token_ids,
                    pad_token_id=pad_token_id,
                    stopping_criteria=stopping_criteria
                )
pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
```