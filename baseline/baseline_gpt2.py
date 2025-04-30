import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.dataloader import load_codesearchnet,tokenize_dataset
from utils.utilities import save_model
from typing import Tuple, Literal, Optional
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

DEFAULT_LANGUAGE: Literal[
    "go", "java", "javascript", "php", "python", "ruby"
] = "python"

OUTPUT_DIR = "../saved_models/baseline_gpt2"

def load_data():
    train_data = load_codesearchnet(DEFAULT_LANGUAGE, "train")
    test_data = load_codesearchnet(DEFAULT_LANGUAGE, "test")
    return train_data, test_data

def load_model(model_name, device):
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def train_model(model, tokenizer, train_data, test_data, output_dir):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    print("The fine tuned model is saved to {output_dir}")

    eval_results = trainer.evaluate()
    print(f"Loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

    return model, tokenizer


def main():

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    print(f"Using Device : {device}")

    print("Getting the model")
    model, tokenizer = load_model("gpt2", device)

    print("Getting the dataset")
    train_data = load_codesearchnet(DEFAULT_LANGUAGE, "train")
    test_data = load_codesearchnet(DEFAULT_LANGUAGE, "test")

    print("Tokenising the dataset")
    tokenised_train_data = tokenize_dataset(train_data, tokenizer)
    tokenised_test_data = tokenize_dataset(test_data, tokenizer)

    print("Training and evaluating the model")
    model, tokenizer = train_model(model, tokenizer, tokenised_train_data, tokenised_test_data, OUTPUT_DIR)

    print("Saving the model")
    save_model(model, tokenizer)

    return

if __name__ == "__main__":
    main()


