# generate_responses.py
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# CONFIG
MODEL_NAME = "meta-llama/Llama-2-7b-hf"   # base
LORA_DIR = "./lora_out_adapter"           # your saved adapter
PROMPT_FILE = "prompts.jsonl"             # input prompts, see format below
OUT_DIR = Path("eval_outputs")
OUT_DIR.mkdir(exist_ok=True)

# generation settings
GEN_CFG = dict(max_new_tokens=256, do_sample=False, temperature=0.2, top_p=0.95)

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Load base model (no LoRA)
print("Loading base model (may be large)...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
base_model.eval()

# Load finetuned model as PeftModel (base + LoRA)
print("Loading base + LoRA adapter (finetuned) ...")
base_for_peft = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)
finetuned = PeftModel.from_pretrained(base_for_peft, LORA_DIR)
finetuned.eval()

def generate_for_model(model, prompt_text):
    input_ids = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**input_ids, **GEN_CFG)

    return tokenizer.decode(gen[0], skip_special_tokens=True)

# Load prompts file (one JSONL line per prompt: {"id": "p1", "prompt": "..."})
prompts = []
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        prompts.append(json.loads(line))

# Prepare outputs
out_base = []
out_finetuned = []

for p in prompts:
    pid = p["id"]
    text = p["prompt"]

    print(f"Generating for prompt {pid} ...")
    # full combined prompt: many benchmarks feed instruction; if you used template, ensure prompt is ready
    base_resp = generate_for_model(base_model, text)
    ft_resp = generate_for_model(finetuned, text)

    out_base.append({"id": pid, "prompt": text, "response": base_resp})
    out_finetuned.append({"id": pid, "prompt": text, "response": ft_resp})

# Save outputs as JSONL
with open(OUT_DIR / "base_responses.jsonl","w",encoding="utf-8") as f:
    for r in out_base:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(OUT_DIR / "finetuned_responses.jsonl","w",encoding="utf-8") as f:
    for r in out_finetuned:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Generation complete. Files written to eval_outputs/")
