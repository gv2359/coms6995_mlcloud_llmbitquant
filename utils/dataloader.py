from typing import Tuple, Literal, Optional

import datasets
from datasets import Dataset
from transformers import PreTrainedTokenizerBase  

DEFAULT_LANGUAGE: Literal[
    "go", "java", "javascript", "php", "python", "ruby"
] = "python"

HF_DATASET_NAME = "code_search_net"

def load_codesearchnet(
    language: str = DEFAULT_LANGUAGE,
    split: str = "train"
) -> Dataset:
    """
    Load the CodeSearchNet split for `language`.
    
    """

    ds = datasets.load_dataset(
        HF_DATASET_NAME,
        language,
        split=split,
        trust_remote_code=True)

    return ds


def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_field: str = "func_code_string",
    max_length: int = 512,
) -> Dataset:
    """
    Tokenise `text_field` and keep only the tokens.
    """
    def _tokenise(batch):
        tokenized = tokenizer(
            batch[text_field],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenised = ds.map(
        _tokenise,
        batched=True,
        remove_columns=ds.column_names)
    
    tokenised.set_format(type="torch",columns=["input_ids", "attention_mask", "labels"])    
    return tokenised