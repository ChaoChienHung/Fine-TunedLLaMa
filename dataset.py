import os
from datasets import load_dataset
from transformers import AutoTokenizer
from config import DATA_DIR, MODEL, MAX_LENGTH

# Ensure cache exists
os.makedirs(DATA_DIR, exist_ok=True)

def load_and_tokenize_dataset():
    ds = load_dataset("databricks/databricks-dolly-15k", cache_dir=DATA_DIR)
    ds = ds['train'].filter(lambda x: bool(x['response'] and x['response'].strip()))
    
    def format_example(x):
        return {"text": f"### Instruction:\n{x['instruction']}\n\n"
                        f"### Context:\n{x['context']}\n\n"
                        f"### Response:\n{x['response']}"}
    ds = ds.map(format_example)
    
    ds_split = ds.train_test_split(test_size=0.2, seed=42)
    val_test_split = ds_split["test"].train_test_split(test_size=0.5, seed=42)
    train_ds = ds_split["train"]
    val_ds = val_test_split["train"]
    test_ds = val_test_split["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, cache_dir=DATA_DIR)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------
    # Tokenize
    # ----------
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    train_tokenized = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
    test_tokenized = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)
    
    print(f"Train: {len(train_tokenized)}, Val: {len(val_tokenized)}, Test: {len(test_tokenized)}")
    return train_tokenized, val_tokenized, test_tokenized, tokenizer
