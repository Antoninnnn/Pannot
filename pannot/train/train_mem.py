from pannot.train.train import train
import os

# Set shared cache directory across all nodes
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/scratch/user/yining_yang/.cache/torch")

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
