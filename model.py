import wandb
from config import MODEL
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model(tokenizer):
    # Initialize wandb logging (optional, but useful for tracking model configs)
    wandb.config.update({
        "model_name": MODEL,
        "quantization": "8-bit",
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    })

    # Load in 8-bit mode for VRAM efficiency
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Resize embeddings if tokenizer was updated
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)

    wandb.log({
        "model_loaded": True,
        "lora_config": lora_config.to_dict()
    })

    print("Model loaded and wrapped with LoRA.")
    return model
