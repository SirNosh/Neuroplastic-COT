import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class CoTDataProcessor:
    """
    Processes the s1K-1.1 dataset for Chain-of-Thought fine-tuning.
    """
    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 2048,
        train_split: str = "train",
        val_split: str = "validation",
        dataset_name: str = "simplescaling/s1K-1.1",
        cot_format: str = "deepseek_thinking_trajectory",
        batch_size: int = 4,
        seed: int = 42
    ):
        self.model_name_or_path = model_name_or_path
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        self.dataset_name = dataset_name
        self.cot_format = cot_format
        self.batch_size = batch_size
        self.seed = seed

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load dataset
        self.dataset = load_dataset(dataset_name)
        logger.info(f"Loaded dataset: {dataset_name}")
        logger.info(f"Dataset splits: {self.dataset.keys()}")
        
    def prepare_datasets(self):
        """
        Prepare train and validation datasets.
        
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        # Process train split
        if self.train_split in self.dataset:
            train_dataset = self.dataset[self.train_split]
            train_dataset = CoTDataset(
                train_dataset, 
                self.tokenizer, 
                self.max_length,
                self.cot_format
            )
            logger.info(f"Prepared training dataset with {len(train_dataset)} examples")
        else:
            train_dataset = None
            logger.warning(f"Train split '{self.train_split}' not found in dataset")
            
        # Process validation split
        if self.val_split in self.dataset:
            val_dataset = self.dataset[self.val_split]
            val_dataset = CoTDataset(
                val_dataset, 
                self.tokenizer, 
                self.max_length,
                self.cot_format
            )
            logger.info(f"Prepared validation dataset with {len(val_dataset)} examples")
        else:
            val_dataset = None
            logger.warning(f"Validation split '{self.val_split}' not found in dataset")
            
        return train_dataset, val_dataset
    
    def get_dataloaders(self):
        """
        Get DataLoaders for train and validation sets with deterministic behavior.

        Returns:
            tuple: (train_dataloader, val_dataloader)
        """
        train_dataset, val_dataset = self.prepare_datasets()

        # Generator for reproducible shuffling
        def seed_worker(worker_id):
            worker_seed = self.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(self.seed)

        train_dataloader = None
        if train_dataset:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                worker_init_fn=seed_worker,
                generator=g
            )

        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )

        return train_dataloader, val_dataloader
    
    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            dict: Batched inputs
        """
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100  # -100 is ignored in loss calculation
        )
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def get_reference_dataloader(self, num_samples=100):
        """
        Get a small dataloader for computing Fisher Information Matrix.
        Uses deterministic sampling for reproducibility.

        Args:
            num_samples: Number of samples to include

        Returns:
            DataLoader: DataLoader with reference examples
        """
        train_dataset, _ = self.prepare_datasets()

        if train_dataset is None:
            logger.warning("No training dataset available for reference dataloader")
            return None

        # Take a subset of the training data with fixed seed
        g = torch.Generator()
        g.manual_seed(self.seed)
        subset_indices = torch.randperm(len(train_dataset), generator=g)[:num_samples].tolist()
        reference_dataset = torch.utils.data.Subset(train_dataset, subset_indices)

        reference_dataloader = DataLoader(
            reference_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

        return reference_dataloader


class CoTDataset(Dataset):
    """
    Dataset for Chain-of-Thought examples.
    """
    def __init__(
        self, 
        dataset, 
        tokenizer, 
        max_length=2048,
        cot_format="deepseek_thinking_trajectory"
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cot_format = cot_format
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        example = self.dataset[idx]
        
        # Format the input with question and CoT
        question = example["question"]
        
        # Get the CoT reasoning trace
        cot = example.get(self.cot_format, "")
        if not cot:
            # Fallback to solution if no CoT is available
            cot = example.get("solution", "")
            
        # Format as instruction with CoT
        prompt = f"Question: {question}\n\nLet me think through this step by step:\n"
        completion = f"{cot}\n\nAnswer: {example['solution']}"
        
        # Tokenize input and output
        tokenized_prompt = self.tokenizer(prompt, truncation=False, add_special_tokens=False)
        tokenized_completion = self.tokenizer(completion, truncation=False, add_special_tokens=False)
        
        # Combine input and output tokens
        input_ids = tokenized_prompt["input_ids"]
        labels = [-100] * len(input_ids)  # Don't compute loss for prompt
        
        completion_input_ids = tokenized_completion["input_ids"]
        input_ids.extend(completion_input_ids)
        labels.extend(completion_input_ids)  # Compute loss for completion
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        } 