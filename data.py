
# dataset for Phi-2 + QLoRA fine-tuning.




from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple, Literal, Optional

import datasets                    # Hugging Face `datasets`
from datasets import Dataset       
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
import torch
from torch.utils.data import DataLoader
try:
    import gcsfs                   # Optional, for GCS downloads
except ImportError:
    gcsfs = None                   


# Config


DEFAULT_LANGUAGE: Literal[
    "go", "java", "javascript", "php", "python", "ruby"
] = "python"

HF_DATASET_NAME = "code_search_net"
DEFAULT_TOKENIZER = "microsoft/phi-2"


# Core helpers



def _maybe_download_from_gcs(gcs_path: str, local_dir: str) -> None:
    """
    If `gcs_path` starts with 'gs://', copy its contents to `local_dir`
    (requires `pip install gcsfs` and GCP auth).
    """
    if not gcs_path.startswith("gs://"):
        return
    if gcsfs is None:
        raise ImportError("gcsfs is required for GCS paths: pip install gcsfs")

    fs = gcsfs.GCSFileSystem()
    fs.get(gcs_path, local_dir, recursive=True)


def load_codesearchnet(
    language: str = DEFAULT_LANGUAGE,
    split: str = "train",
    cache_dir: str | None = None,
    streaming: bool = False,
) -> Dataset:
    """
    Load the CodeSearchNet split for `language`.

    • If `cache_dir` points to an on-disk copy, load from disk.
    • If `cache_dir` is a GCS bucket (gs://…), download locally once.
    • Otherwise, stream / download from the HF Hub.
    """
    # 1) Resolve cache directory
    resolved_cache: Optional[Path] = None
    if cache_dir:
        resolved_cache = Path(cache_dir).expanduser()
        if cache_dir.startswith("gs://"):
            local_tmp = Path.home() / ".cache" / "codesearchnet_gcs"
            _maybe_download_from_gcs(cache_dir, str(local_tmp))
            resolved_cache = local_tmp

    # 2) Load dataset
    if resolved_cache and (resolved_cache / language / split).exists():
        return datasets.load_from_disk(resolved_cache / language / split)

    ds = datasets.load_dataset(
        HF_DATASET_NAME,
        language,
        split=split,
        streaming=streaming,
        cache_dir=str(resolved_cache) if resolved_cache else None,
        trust_remote_code=True,
    )

    # Persist to disk for reuse (only if not streaming)
    if resolved_cache and not streaming:
        save_path = resolved_cache / language / split
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(save_path)

    return ds



# Tokenisation utilities



def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    text_field: str = "code",
    max_length: int = 1024,
    num_proc: int | None = 8,
) -> Dataset:
    """
    Tokenise `text_field` and keep only the tokens.
    """
    def _tokenise(batch):
        return tokenizer(
            batch[text_field],
            truncation=True,
            max_length=max_length,
        )

    tokenised = ds.map(
        _tokenise,
        batched=True,
        remove_columns=ds.column_names if not ds.streaming else None,
        num_proc=None if ds.streaming else num_proc,
    )
    tokenised.set_format(type="torch")    
    return tokenised


def build_dataloader(
    tokenised_ds: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap the Hugging Face `Dataset` into a PyTorch `DataLoader`.
    """
    return DataLoader(
        tokenised_ds,
        batch_size=batch_size,
        shuffle=shuffle and not tokenised_ds.streaming,
        collate_fn=lambda x: {
            k: torch.stack([d[k] for d in x]) for k in x[0]
        },
    )



# Convenience one-liner



def get_training_dataloader(
    language: str = DEFAULT_LANGUAGE,
    split: str = "train",
    cache_dir: str | None = None,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    batch_size: int = 4,
    streaming: bool = False,
) -> Tuple[DataLoader, PreTrainedTokenizerBase]:
    """
    End-to-end helper: load -> tokenise -> return DataLoader & tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        use_fast=True,
    )

    ds = load_codesearchnet(
        language=language,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    tokenised = tokenize_dataset(ds, tokenizer)
    loader = build_dataloader(tokenised, batch_size=batch_size)
    return loader, tokenizer




