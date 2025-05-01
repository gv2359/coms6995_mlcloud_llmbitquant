import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils.dataloader import load_codesearchnet,tokenize_dataset
from utils.trainer import train_model
from utils.utilities import save_model
from quantizers.baseline import load_model
from typing import Tuple, Literal, Optional
import torch

DEFAULT_LANGUAGE: Literal[
    "go", "java", "javascript", "php", "python", "ruby"
] = "python"

OUTPUT_DIR = "../saved_models/baseline_gpt2"

def load_data():
    train_data = load_codesearchnet(DEFAULT_LANGUAGE, "train")
    test_data = load_codesearchnet(DEFAULT_LANGUAGE, "test")
    return train_data, test_data


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


