from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from utils.utilities import log_gpu_memory
import torch
import time

def train_model(model, tokenizer, train_data, test_data, output_dir, log_dir):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        eval_strategy="epoch",
        logging_steps=50,
        logging_dir=log_dir,
        save_strategy="epoch",
        remove_unused_columns=False,
        fp16=True,
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()

    print(f"The fine tuned model is saved to {output_dir}")

    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"])).item()

    print(f"Loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {perplexity}")

    metrics = {
        "eval_loss": eval_results["eval_loss"],
        "perplexity": perplexity,
        "training_time_sec": end_time - start_time,
        "gpu_memory": log_gpu_memory(),
        "output_dir": output_dir
    }

    return model, tokenizer, metrics