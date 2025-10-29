import os
import wandb
from config import MODEL, MODEL_DIR
from accelerate import PartialState
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def load_model(tokenizer, cache_dir=MODEL_DIR, device_map="auto", use_lora=True, quantize=True):
    """
    Load a causal language model, optionally applying LoRA and quantization.

    Args:
        tokenizer: The tokenizer for resizing token embeddings.
        cache_dir: The cache directory to store pre-trained models.
        device_map: The device map to determine how to distribute the model across devices.
        use_lora: Whether to apply LoRA (Low-Rank Adaptation) to the model.
        quantize: Whether to apply 8-bit quantization for memory efficiency.

    Returns:
        The loaded and optionally modified model.
    """
    
    wandb.init(project="model-training", config={
        "model_name": MODEL,
        "quantization": "8-bit" if quantize else "None",
        "lora_r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    })
    
    if device_map == 'DDP':
        device_string = PartialState().process_index
        device_map = { '': device_string }

    try:
        bnb_config = None
        if quantize:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            quantization_config=bnb_config,
            device_map=device_map,
            cache_dir=cache_dir
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    model.resize_token_embeddings(len(tokenizer))
    
    if use_lora:
        model = prepare_model_for_kbit_training(model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        wandb.log({"lora_config": lora_config.to_dict()})

    wandb.log({
        "model_loaded": True,
        "device_map": device_map,
        "quantization": quantize,
        "use_lora": use_lora
    })

    print("Model loaded and wrapped with LoRA." if use_lora else "Model loaded without LoRA.")
    return model
