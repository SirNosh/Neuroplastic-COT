"""
Utilities for loading and preprocessing the S1.1K Chain of Thought dataset.
"""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from neuroplastic_cot.config import (
    BATCH_SIZE,
    MAX_INPUT_LENGTH,
    MAX_OUTPUT_LENGTH,
    DATASET_NAME,
    TRAIN_SPLIT_RATIO
)


class S11KDataset(Dataset):
    """
    Dataset class for the S1.1K Chain of Thought dataset.
    """
    
    def __init__(self, examples, tokenizer, max_input_length, max_output_length):
        """
        Initialize the dataset.
        
        Args:
            examples: List of examples from the dataset
            tokenizer: Tokenizer to use for encoding
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a preprocessed example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Preprocessed example
        """
        example = self.examples[idx]
        return preprocess_s1_1k_example(
            example, 
            self.tokenizer, 
            self.max_input_length, 
            self.max_output_length
        )


def load_s1_1k_dataset(train_split_ratio=TRAIN_SPLIT_RATIO):
    """
    Load the S1.1K Chain of Thought dataset.
    
    Args:
        train_split_ratio: Ratio of data to use for training
        
    Returns:
        Train and test datasets
    """
    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    
    # Create train/test split
    train_size = int(len(dataset["train"]) * train_split_ratio)
    test_size = len(dataset["train"]) - train_size
    
    # Split dataset
    train_test_split = dataset["train"].train_test_split(
        train_size=train_size,
        test_size=test_size,
        seed=42
    )
    
    return train_test_split["train"], train_test_split["test"]


def preprocess_s1_1k_example(example, tokenizer, max_input_length, max_output_length):
    """
    Preprocess an example from the S1.1K dataset.
    
    Args:
        example: Example from the dataset
        tokenizer: Tokenizer to use for encoding
        max_input_length: Maximum input sequence length
        max_output_length: Maximum output sequence length
        
    Returns:
        Preprocessed example
    """
    # Format input
    input_text = f"Question: {example['question']}\nAnswer with step-by-step reasoning:"
    
    # Format target (prioritize CoT fields if available)
    if "cot_gpt4" in example and example["cot_gpt4"]:
        target_text = example["cot_gpt4"]
    elif "cot_palm" in example and example["cot_palm"]:
        target_text = example["cot_palm"]
    elif "cot_claude" in example and example["cot_claude"]:
        target_text = example["cot_claude"]
    else:
        # Fallback to answer if no CoT is available
        target_text = example["answer"]
    
    # Tokenize input and target
    input_ids = tokenizer(
        input_text,
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.squeeze()
    
    target_ids = tokenizer(
        target_text,
        max_length=max_output_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids.squeeze()
    
    # Create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).float()
    
    # Create labels (set to -100 for input tokens to ignore them in loss calculation)
    labels = torch.cat([
        torch.ones_like(input_ids) * -100,
        target_ids
    ])[:max_input_length + max_output_length]
    
    # Pad or truncate labels to match expected length
    if len(labels) < max_input_length + max_output_length:
        labels = torch.cat([
            labels,
            torch.ones(max_input_length + max_output_length - len(labels), dtype=torch.long) * -100
        ])
    else:
        labels = labels[:max_input_length + max_output_length]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    
    Args:
        batch: Batch of examples
        
    Returns:
        Collated batch
    """
    # Filter out examples with empty input_ids
    batch = [example for example in batch if example["input_ids"].numel() > 0]
    
    if not batch:
        return None
    
    # Stack tensors
    input_ids = torch.stack([example["input_ids"] for example in batch])
    attention_mask = torch.stack([example["attention_mask"] for example in batch])
    labels = torch.stack([example["labels"] for example in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def get_dataloaders(tokenizer, batch_size=BATCH_SIZE):
    """
    Get train and validation dataloaders.
    
    Args:
        tokenizer: Tokenizer to use for encoding
        batch_size: Batch size for dataloaders
        
    Returns:
        Train and validation dataloaders
    """
    # Load dataset
    train_dataset, val_dataset = load_s1_1k_dataset()
    
    # Create dataset objects
    train_dataset = S11KDataset(
        train_dataset,
        tokenizer,
        MAX_INPUT_LENGTH,
        MAX_OUTPUT_LENGTH
    )
    
    val_dataset = S11KDataset(
        val_dataset,
        tokenizer,
        MAX_INPUT_LENGTH,
        MAX_OUTPUT_LENGTH
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader 