import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ============================================================
# 1. Choose where to store model inside project
# ============================================================

# dynamically find current project root (file’s parent)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_cache", "romansetu-cpt-roman-sft-roman")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"📦 Downloading model: ai4bharat/romansetu-cpt-roman-sft-roman")
print(f"📂 Local path: {MODEL_DIR}")
print("----------------------------------------------------")

# ============================================================
# 2. Snapshot download with progress bar
# ============================================================

model_path = snapshot_download(
    repo_id="ai4bharat/romansetu-cpt-roman-sft-roman",
    local_dir=MODEL_DIR,              # save directly into project folder
    local_dir_use_symlinks=False,     # real copy, not symlink
    resume_download=True,             # resume if interrupted
    force_download=False,             # don’t overwrite if cached
    tqdm_class=None                   # ensures visible progress bar
)

print("\n✅ Model successfully downloaded into:")
print(model_path)
print("----------------------------------------------------\n")

# ============================================================
# 3. Quick test to verify model loads from disk
# ============================================================

print("🧠 Verifying model load from local directory...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,   # ✅ correct arg name
    device_map="auto"
)


device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

print(f"✅ Model loaded successfully on {device.upper()} from local path.")
print("You can now safely use this directory for all future runs.")
