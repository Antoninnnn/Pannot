# Pannot : Protein Language Understanding Interface

This project aims at protein multimodal inputs and natural language output. It was adapted from the llava project (https://github.com/haotian-liu/LLaVA)

## Release


-[05/24] Pannot model and basic training module is tested

## Command to set up the environment 


### For Grace HPRC(TAMU)
```
module purge
module load GCC/12.3.0 CUDA/11.8.0 Anaconda3
```

### Set the conda virtual environment
```
conda create -n pannot-dev python=3.10 -y
source activate pannot-dev
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

```

### Install train

```

pip install -e .[train]


```

## Model

