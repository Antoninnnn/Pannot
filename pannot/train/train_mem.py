from pannot.train.train import train
import os

# Set shared cache directory across all nodes
os.environ["TORCH_HOME"] = os.environ.get("TORCH_HOME", "/scratch/user/yining_yang/.cache/torch")

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_DIR"] = os.environ.get("SCRATCH", "/tmp") + "/wandb_logs"
# Optional: keep if you ever switch to online
os.environ["WANDB_API_KEY"] = "c6da89ba565a8b25f5b18c6fb722e7ad6637d4de"
if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
