# Fine-Tuning and Analysis of 1-Bit LLMs with QLoRa Adapters and other quantizations

This project explores and compares quantization techniques for efficiently fine-tuning large language models, with a focus on **QLoRA** and **BitNet**. The base model is **GPT-2**, and scalability is tested on a **LLaMA-7B** model. The goal is to reduce memory usage and latency without significant performance loss.

---

## 🔍 Project Objectives

- Evaluate various fine-tuning strategies:
  - Full-precision baseline
  - 8-bit and 4-bit quantization
  - QLoRA (4-bit LoRA adaptation)
  - BitNet (1.58-bit ternary compression)
  - Combined QLoRA + BitNet
- Measure:
  - Perplexity
  - Training time
  - Inference latency
  - Peak GPU memory usage
  - Trainable vs. total parameters
- Scale the best approach to a 7B-parameter model
- Deploy and benchmark on Google Cloud (T4 GPU)

---

## 📦 Dataset

- **CodeSearchNet (Python subset)**  
  Source: [Papers With Code](https://paperswithcode.com/dataset/codesearchnet)  
  Contains millions of function-documentation pairs for supervised code modeling.

---

## 🧪 Techniques Used

| Method              | Quantization | Trainable Params | Notes                        |
|---------------------|--------------|------------------|------------------------------|
| Baseline Fine-Tune  | None (FP32)  | All              | High accuracy, slow training |
| 8-bit / 4-bit       | 8 / 4-bit    | All              | Reduced memory footprint     |
| QLoRA               | 4-bit (NF4)  | LoRA adapters    | Efficient fine-tuning        |
| BitNet              | 1.58-bit     | Ternary linear   | Custom CUDA; inference only  |
| QLoRA + BitNet      | 1.58-bit     | LoRA only        | Best for deployment          |

---

## 📈 Metrics Collected

- **Perplexity**
- **Training Time**
- **GPU Peak Memory Usage**
- **Inference Latency**
- **Trainable Parameters**
- **Total Parameters**

---
## 🛠️ Setup Instructions

```bash
# Create environment
python -m  venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```