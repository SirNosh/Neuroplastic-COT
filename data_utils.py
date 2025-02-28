"""
Utilities for loading and preprocessing the S1.1K Chain of Thought dataset.
"""

import logging
import os
import json
import random
from typing import Dict, List, Optional, Union, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

import config

logger = logging.getLogger(__name__)


def load_s1_1k_dataset(
    dataset_name: str = "simplescaling/s1K-1.1",
    dataset_subset: Optional[str] = None,
    train_test_split: float = 0.9,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """
    Load the S1.1K Chain of Thought dataset and create train/test splits.
    
    Args:
        dataset_name: Name of the dataset to load
        dataset_subset: Subset of the dataset to load (if applicable)
        train_test_split: Fraction of data to use for training
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load the dataset
    if dataset_subset:
        dataset = load_dataset(dataset_name, dataset_subset)
    else:
        dataset = load_dataset(dataset_name)
    
    # Get the 'train' split if it exists, otherwise use the default split
    if "train" in dataset:
        full_dataset = dataset["train"]
    else:
        full_dataset = dataset[next(iter(dataset.keys()))]
    
    # Create train/test splits
    dataset_dict = full_dataset.train_test_split(
        test_size=1 - train_test_split, 
        seed=seed
    )
    
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]
    
    logger.info(f"Loaded dataset with {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    return train_dataset, test_dataset


def preprocess_s1_1k_example(
    example: Dict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    prompt_template: str,
) -> Dict:
    """
    Preprocess a single example from the S1.1K dataset.
    
    Args:
        example: The example to preprocess
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length
        prompt_template: Template for formatting the prompt
    
    Returns:
        Preprocessed example with input_ids, attention_mask, and labels
    """
    # Get the question
    question = example.get("question", "")
    
    # Format the input using the prompt template
    input_text = prompt_template.format(problem=question)
    
    # Get the target text (chain of thought)
    # Prioritize using available CoT fields in this order
    cot_fields = ["gemini_cot", "deepseek_cot", "claude_cot", "gpt4_cot"]
    target_text = None
    
    for field in cot_fields:
        if field in example and example[field]:
            target_text = example[field]
            break
    
    # If no CoT field is available, use the solution field
    if target_text is None and "solution" in example and example["solution"]:
        target_text = example["solution"]
    
    # If still no target text, use an empty string
    if target_text is None:
        target_text = ""
    
    # Tokenize input and target
    tokenized_input = tokenizer(input_text, truncation=True, max_length=max_length // 2)
    tokenized_target = tokenizer(target_text, truncation=True, max_length=max_length // 2)
    
    # Combine input and target for the labels
    input_ids = tokenized_input["input_ids"]
    target_ids = tokenized_target["input_ids"]
    
    # Create labels: -100 for input tokens (which we don't want to predict)
    # and the actual token ids for the target tokens
    labels = [-100] * len(input_ids) + target_ids
    
    # Combine input and target ids for the full sequence
    full_input_ids = input_ids + target_ids
    
    # Create attention mask
    attention_mask = [1] * len(full_input_ids)
    
    # Ensure all sequences are of the same length
    if len(full_input_ids) > max_length:
        full_input_ids = full_input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    else:
        # Pad to max_length
        padding_length = max_length - len(full_input_ids)
        full_input_ids.extend([tokenizer.pad_token_id] * padding_length)
        attention_mask.extend([0] * padding_length)
        labels.extend([-100] * padding_length)
    
    return {
        "input_ids": torch.tensor(full_input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }


def collate_fn(examples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        examples: List of examples to collate
    
    Returns:
        Batch dictionary with input_ids, attention_mask, and labels
    """
    # Filter out examples with empty input_ids
    valid_examples = [ex for ex in examples if len(ex["input_ids"]) > 0]
    
    if not valid_examples:
        # Return empty batch if no valid examples
        return {
            "input_ids": torch.empty(0, 0, dtype=torch.long),
            "attention_mask": torch.empty(0, 0, dtype=torch.long),
            "labels": torch.empty(0, 0, dtype=torch.long),
        }
    
    # Stack tensors
    batch = {
        "input_ids": torch.stack([ex["input_ids"] for ex in valid_examples]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in valid_examples]),
        "labels": torch.stack([ex["labels"] for ex in valid_examples]),
    }
    
    return batch


class S1_1KDataset(Dataset):
    """
    Dataset class for the S1.1K Chain of Thought dataset.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        prompt_template: str,
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset: The raw dataset
            tokenizer: The tokenizer to use
            max_length: Maximum sequence length
            prompt_template: Template for formatting the prompt
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
    
    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item to get
        
        Returns:
            Preprocessed example
        """
        example = self.dataset[idx]
        return preprocess_s1_1k_example(
            example,
            self.tokenizer,
            self.max_length,
            self.prompt_template,
        )


def get_dataloaders(
    tokenizer: PreTrainedTokenizer,
    dataset_name: str = None,
    dataset_subset: str = None,
    max_length: int = None,
    batch_size: int = None,
    prompt_template: str = None,
    train_test_split: float = 0.9,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation dataloaders.
    
    Args:
        tokenizer: The tokenizer to use
        dataset_name: Name of the dataset to load
        dataset_subset: Subset of the dataset to load
        max_length: Maximum sequence length
        batch_size: Batch size
        prompt_template: Template for formatting the prompt
        train_test_split: Fraction of data to use for training
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    # Use config values if not provided
    dataset_name = dataset_name or config.DATASET_NAME
    dataset_subset = dataset_subset or config.DATASET_SUBSET
    max_length = max_length or config.MAX_LENGTH
    batch_size = batch_size or config.BATCH_SIZE
    prompt_template = prompt_template or config.PROMPT_TEMPLATE
    
    # Load dataset
    train_dataset, test_dataset = load_s1_1k_dataset(
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        train_test_split=train_test_split,
        seed=seed,
    )
    
    # Create dataset objects
    train_dataset = S1_1KDataset(
        dataset=train_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_template=prompt_template,
    )
    
    eval_dataset = S1_1KDataset(
        dataset=test_dataset,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt_template=prompt_template,
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    return train_dataloader, eval_dataloader 