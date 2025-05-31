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



```
deepspeed --hostfile ./scripts/hostfile.txt --num_gpus 2    pannot/train/train_mem.py     --deepspeed ./scripts/zero2.json     --model_name_or_path local_pretrained_llm/$MODEL_VERSION     --version $PROMPT_VERSION     --data_path ${DATA_PATH}     --tune_mm_mlp_adapter True     --bf16 True     --output_dir ${OUTPUT_DIR}     --num_train_epochs 1     --per_device_train_batch_size 4     --per_device_eval_batch_size 4     --gradient_accumulation_steps 1     --evaluation_strategy "no"     --save_strategy "steps"     --save_steps 24000     --save_total_limit 1     --learning_rate 2e-3     --weight_decay 0.0     --warmup_ratio 0.03     --lr_scheduler_type "cosine"     --logging_steps 1     --tf32 True     --model_max_length 2048     --gradient_checkpointing True     --dataloader_num_workers 4     --lazy_preprocess True     --report_to wandb     --use_seq_tower True     --mm_seq_tower $SEQ_TOWER     --mm_seq_projector_type linear     --mm_seq_select_layer -1     --mm_seq_select_feature "cls"    --mm_seq_no_pooling True     --use_str_tower True     --mm_struc_tower $STR_TOWER     --mm_str_projector_type linear     --mm_str_select_layer -1     --mm_str_select_feature "residue" 

```


TO upload the local wandb record 
```
wandb sync offline-run-20250528_034900-dcplvyd3
```


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

The data is stored in a json file. The data is stored in a list of dictionaries. Each dictionary contains the following keys:



##### Frames For natural language:
input_ids
labels
attention_mask

##### Frames For sequence input:
seq_input_ids
seq_attention_mask

##### Frames For strucuture input:
struct_coords
