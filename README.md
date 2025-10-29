# Fine‑TunedLLaMa

This repository is a practice project for fine‑tuning the LLaMA‑2 (7B) model using parameter‑efficient fine‑tuning (PEFT) via LoRA, trained on the databricks/dolly‑15k dataset. The goal is to demonstrate loss convergence and observable performance uplift on benchmarks such as AlpacaEval2 and MT‑Bench compared to the base model.

---

## 🚀 Highlights

- Uses PEFT (LoRA) to fine‑tune LLaMA‑2‑7B, enabling efficient adaptation.  
- Trained on the Dolly‑15K dataset from Databricks.  
- Codebase includes training scripts, dataset handling, model architecture adjustments, LoRA merging, and evaluation scaffolding.  
- Shows quantitative improvements (loss convergence and benchmark scores) over the base model.

---

## 📁 Repository Structure

- `config.py` – Configuration settings for model, training, dataset paths, hyperparameters.  
- `dataset.py` – Handles the loading, processing, and batching of the Dolly‑15K dataset.  
- `model.py` – Defines the model architecture modifications (e.g., adding LoRA layers) and how the base model is wrapped.  
- `train.py` – Training script (single GPU / default) for fine‑tuning.  
- `train_ddp.py` – Training script supporting distributed data‑parallel (DDP) training for multiple GPUs.  
- `merge_lora.py` – Utility to merge the LoRA weights back into model (for inference or deployment).  
- `requirements.txt` – Python dependencies for the project.  

---

## 🔧 Setup & Usage

### Prerequisites

- Python 3.8+ (or compatible version).  
- GPUs with sufficient memory (fine‑tuning LLaMA‑2‑7B typically requires > 20 GB VRAM per GPU, depending on batch size).  
- Access to the base model weights for LLaMA‑2‑7B from Meta (ensure you comply with licensing).  
- The Dolly‑15K dataset (via HuggingFace or Databricks).  

### Installation
```bash
git clone https://github.com/ChaoChienHung/Fine‑TunedLLaMa.git  
cd Fine‑TunedLLaMa  
pip install -r requirements.txt  
```
### Configuration

Edit `config.py` to set:  

- Paths to model checkpoint, dataset files  
- LoRA hyperparameters (rank, alpha, dropout)  
- Training hyperparameters (batch size, learning rate, epochs)  
- DDP settings if using `train_ddp.py`  

### Running Training

Single GPU:
```bash
python train.py
```

Multi‑GPU (DDP):
```bash
python train_ddp.py --num_gpus N
```
### After Training

You can merge the LoRA weights into the base model for inference using:
```bash
python -m utils.merge_lora                                              
```

Then use the merged model for inference or deployment.

---

## 📊 Results & Benchmarking

The fine‑tuned model shows:

- Loss convergence compared to base model (plots/logs included).  
- Improved scores on AlpacaEval2 and MT‑Bench vs the base LLaMA‑2‑7B model.  
- Demonstrates that parameter‑efficient fine‑tuning (LoRA) works effectively for large language models.

---

## 🧠 Why This Matters

Fine‐tuning large language models from scratch is expensive. Using PEFT approaches like LoRA enables:

- Much smaller number of trainable parameters (faster training, less memory).  
- Retaining the majority of the base model while adapting to a new dataset (like Dolly‑15K).  
- A practical workflow to bring base‑model capabilities into a more specialized setting with fewer resources.

---

## ✅ Contribution & Usage Notes

- This is a **practice** project: meant for experimentation / learning.  
- If you adapt or extend this work, please ensure license compliance for LLaMA‑2 and Dolly‑15K.  
- Contributions, issues, enhancements welcome (e.g., alternative datasets, hyperparameter tuning, inference pipelines).  
- For deployment, further steps may be needed (quantization, optimization, serving infrastructure).

---

## 📚 References

- Meta’s LLaMA‑2 model (7B variant) – check licensing.  
- Databricks Dolly‑15K dataset.  
- PEFT / LoRA methods for efficient fine‑tuning.  
- Benchmarks: AlpacaEval2, MT‑Bench.

---

## 📄 License

Specify license here (e.g., MIT, Apache 2.0) depending on your preference.  
If no license file included, consider adding one to clarify usage rights.

---

Thank you for exploring this repository! If you have questions or suggestions, feel free to open an issue or pull request.