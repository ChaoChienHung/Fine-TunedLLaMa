import os
from datasets import load_dataset
from transformers import AutoTokenizer
from config import TOKENIZER_DIR, MODEL, MAX_LENGTH, DATA_DIR

def load_and_tokenize_dataset():
    # Ensure cache exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load and filter dataset
    ds = load_dataset("databricks/databricks-dolly-15k", cache_dir=DATA_DIR)
    ds = ds['train'].filter(lambda x: bool(x['response'] and x['response'].strip()))

    # Format: separate input (prompt) and label (response)
    def format_example(x):
        context_str = f"\n\n### Context:\n{x['context']}" if x.get('context') else ""
        prompt = f"### Instruction:\n{x['instruction']}{context_str}\n\n### Response:\n"
        return {
            "input": prompt,
            "label": x["response"]
        }

    ds = ds.map(format_example)
    
    # Split train / val / test
    ds_split = ds.train_test_split(test_size=0.2, seed=42)
    val_test_split = ds_split["test"].train_test_split(test_size=0.5, seed=42)
    train_ds = ds_split["train"]
    val_ds = val_test_split["train"]
    test_ds = val_test_split["test"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, cache_dir=TOKENIZER_DIR)
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "</s>"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize input and label separately
    def tokenize(example):
        input_enc = tokenizer(
            example["input"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        label_enc = tokenizer(
            example["label"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length"
        )
        input_enc["labels"] = label_enc["input_ids"]
        return input_enc

    train_tokenized = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)
    test_tokenized = test_ds.map(tokenize, batched=True, remove_columns=test_ds.column_names)
    
    print(f"Train: {len(train_tokenized)}, Val: {len(val_tokenized)}, Test: {len(test_tokenized)}")
    return train_tokenized, val_tokenized, test_tokenized, tokenizer

if __name__ == "__main__":
    train_tokenized, val_tokenized, test_tokenized, tokenizer = load_and_tokenize_dataset()
