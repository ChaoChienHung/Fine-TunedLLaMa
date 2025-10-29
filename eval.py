from peft import PeftModel
from lm_eval import tasks, evaluator
from config import DATETIME
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "meta-llama/Llama-2-7b-hf"  # e.g., "decapoda-research/llama-7b-hf"
lora_adapter_path = f"./lora_out_adapter"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load LoRA on top
lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

task_list = ["mmlu_pro"]  # replace with any task you like

# Evaluate base model
print("Evaluating Base Model...")
base_results = evaluator.evaluate(base_model, tokenizer, task_list)

# Evaluate LoRA model
print("Evaluating LoRA Model...")
lora_results = evaluator.evaluate(lora_model, tokenizer, task_list)

# Compare
for task in task_list:
    print(f"\nTask: {task}")
    print(f"Base Model Accuracy: {base_results[task]['accuracy']}")
    print(f"LoRA Model Accuracy: {lora_results[task]['accuracy']}")
