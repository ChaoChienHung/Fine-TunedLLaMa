import os
import torch
from safetensors.torch import load_file
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import LORA_PATH, MODEL_DIR

print("Loading LoRA config...")
config = PeftConfig.from_pretrained(LORA_PATH)
print(f"Base model (from adapter): {config.base_model_name_or_path}")

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map=None  # no offload -> avoids 'meta' device warnings
)

adapter_path = os.path.join(LORA_PATH, "adapter_model.safetensors")
adapter_weights = None
if os.path.exists(adapter_path):
    print("Using fixed adapter weights...")
    adapter_weights = load_file(adapter_path)

print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH, adapter_state_dict=adapter_weights)

print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

print("Saving merged model...")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
save_path = os.path.join(MODEL_DIR, "merged_lora_llama_7b")
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ Merged model saved to: {save_path}")
