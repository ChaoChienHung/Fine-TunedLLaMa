import wandb
from model import load_model
from datetime import datetime
from huggingface_hub import login
from dataset import load_and_tokenize_dataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

# -------------------------------
# Get today's date in YYYY-MM-DD format
# -------------------------------
today_date = datetime.today().strftime('%Y-%m-%d %H-%M-%S')

# -------------------------------
# Initialize wandb for logging
# -------------------------------
wandb.init(project="Fine-Tuned LLaMa", name=f"Fine-Tuned LLaMa - {today_date}", config={
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
})

# -------------------------------
# Load tokenized dataset & tokenizer
# -------------------------------
train_tokenized, val_tokenized, test_tokenized, tokenizer = load_and_tokenize_dataset()

# -------------------------------
# Load model with LoRA
# -------------------------------
model = load_model(tokenizer)

# -------------------------------
# Data collator for causal LM
# -------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir=f"./lora_out_{today_date}",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch ~16
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    report_to="wandb",  # Only report to wandb
    load_best_model_at_end=True,
)

# -------------------------------
# Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# -------------------------------
# Train
# -------------------------------
trainer.train()

# -------------------------------
# Save LoRA adapter only with today's date
# -------------------------------
model.save_pretrained(f"./lora_out_adapter_{today_date}")
print("LoRA adapter saved!")

# Finish the wandb run
wandb.finish()
