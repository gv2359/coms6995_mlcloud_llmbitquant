# utils/finetuning.py


from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset                                                       # HF Datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from data import load_codesearchnet, tokenize_dataset                        



#  Helper to build the fine-tuning dataset

def get_tokenised_dataset(
    language: str,
    split: str,
    tokenizer,
    cache_dir: Optional[str] = None,
    streaming: bool = False,
    max_length: int = 1024,
    max_samples: Optional[int] = None,
) -> Dataset:
    """
    Load CodeSearchNet, tokenise `code` field, return a HF Dataset ready for Trainer.
    """
    raw = load_codesearchnet(
        language=language,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    if max_samples and not raw.streaming:
        raw = raw.shuffle(seed=42).select(range(max_samples))

    tokenised = tokenize_dataset(
        raw,
        tokenizer=tokenizer,
        text_field="code",
        max_length=max_length,
    )
    return tokenised



#  Main fine-tuning routine

def main(args: argparse.Namespace) -> None:
    # 1) Tokeniser ---------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token   # Phi-2 has no pad token

    # 2) 4-bit quantised base model 
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", #4bit
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 3) Attach LoRA adapters
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules.split(",") if args.target_modules else None,
    )
    model = get_peft_model(model, lora_cfg)

    # 4) Dataset
    train_set = get_tokenised_dataset(
        language=args.language,
        split=args.split,
        tokenizer=tokenizer,
        cache_dir=args.cache_dir,
        streaming=args.streaming,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # # 5) Trainer setup 
    # train_args = TrainingArguments(
    #     output_dir=args.output_dir,
    #     num_train_epochs=args.epochs,
    #     per_device_train_batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.grad_accum,
    #     learning_rate=args.lr,
    #     bf16=args.bf16,
    #     fp16=not args.bf16,
    #     logging_steps=20,
    #     save_strategy="epoch",
    #     save_total_limit=3,
    #     report_to="none",
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=train_args,
    #     train_dataset=train_set,
    #     data_collator=data_collator,
    # )


    # trainer.train()
    # trainer.save_model(args.output_dir)
    # tokenizer.save_pretrained(args.output_dir)
    # print(f"\n  Model + tokenizer saved to: {args.output_dir}\n")

