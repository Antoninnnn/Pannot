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
pip install -e .

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

pip install torch-geometric

```

problem solving : I(Yining) met some problem for the torch with cuda 11.8. The final solution is using `pip uninstall torch torchvision`, and call `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118` separately

### Install train

```

pip install -e .[train]
pip install "flash-attn<=2.5.6" --no-build-isolation

```

## Model

