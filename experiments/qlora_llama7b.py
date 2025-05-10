import time
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import evaluate
import bitsandbytes as bnb

# ---------------------
# Load dataset
# ---------------------
dataset = load_dataset("code_search_net", "python", split="train[:1%]")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token  # LLaMA needs this

def tokenize_function(example):
    return tokenizer(example["code"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True).remove_columns(dataset.column_names)

# ---------------------
# Load 4-bit quantized LLaMA 7B with QLoRA
# ---------------------
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",
    load_in_4bit=True,
    quantization_config=bnb.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
)

# ---------------------
# Apply LoRA config
# ---------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------
# Set training arguments
# ---------------------
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=True,
    output_dir="./qlora-llama-7b-output",
    logging_dir="./logs",
    report_to="none"
)

# ---------------------
# Training with timing and memory profiling
# ---------------------
start_time = time.time()
torch.cuda.reset_peak_memory_stats()
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_args,
    tokenizer=tokenizer,
    max_seq_length=512,
)
trainer.train()
training_time = time.time() - start_time
peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)

# ---------------------
# Perplexity evaluation
# ---------------------
perplexity_metric = evaluate.load("perplexity")
val_dataset = load_dataset("code_search_net", "python", split="validation[:0.5%]")
val_tokenized = val_dataset.map(tokenize_function, batched=True).remove_columns(val_dataset.column_names)
model.eval()
with torch.no_grad():
    results = perplexity_metric.compute(model=model, predictions=val_tokenized["input_ids"], tokenizer=tokenizer)
perplexity = results["perplexities"][0]

# ---------------------
# Inference latency test
# ---------------------
import torch.cuda
input_text = "def fibonacci(n):"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
start_latency = time.time()
_ = model.generate(input_ids, max_new_tokens=256)
latency = time.time() - start_latency

# ---------------------
# Print metrics
# ---------------------
print(f"Perplexity: {perplexity:.2f}")
print(f"Training time: {training_time / 3600:.2f} hours")
print(f"Peak GPU memory: {peak_memory:.2f} GB")
print(f"Inference latency (256 tokens): {latency:.2f} sec")
