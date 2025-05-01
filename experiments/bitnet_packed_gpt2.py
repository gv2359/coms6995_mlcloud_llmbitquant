import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.dataloader import load_codesearchnet,tokenize_dataset
from utils.utilities import save_model,count_parameters,save_metrics
from utils.trainer import train_model
from quantizers.bitnet_packed import load_model
from typing import Tuple, Literal, Optional
import torch

DEFAULT_LANGUAGE: Literal[
    "go", "java", "javascript", "php", "python", "ruby"
] = "python"

OUTPUT_DIR = "../saved_models/bitnet_packed_gpt2"
LOG_DIR = "../logs/bitnet_packed_gpt2"
RESULTS_PATH = "../results"
EXPERIMENT_ID = "bitnet_packed_gpt2"


def main():

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")

    print(f"Using Device : {device}")

    print("Getting the model")
    model, tokenizer = load_model("gpt2", device)
    param_info = count_parameters(model)
    print(f"Trainable: {param_info['trainable']:,} / {param_info['total']:,} ({param_info['percent']:.2f}%)")
    
    print("Getting the dataset")
    train_data, test_data = load_codesearchnet(DEFAULT_LANGUAGE)

    print("Tokenising the dataset")
    tokenised_train_data = tokenize_dataset(train_data, tokenizer)
    tokenised_test_data = tokenize_dataset(test_data, tokenizer)
    
    print("Training and evaluating the model")
    model, tokenizer, metrics = train_model(model, tokenizer, tokenised_train_data, tokenised_test_data, OUTPUT_DIR, LOG_DIR)
    metrics.update(param_info)
    save_metrics(metrics, RESULTS_PATH, EXPERIMENT_ID)
    
    print("Saving the model")
    save_model(model, tokenizer, OUTPUT_DIR)

    return

if __name__ == "__main__":
    main()


