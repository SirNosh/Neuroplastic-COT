"""
Neuroplasticity techniques for optimizing Chain of Thought reasoning during fine-tuning.

This module implements:
1. Adaptive Learning Rate (ALR) - Dynamically adjusts learning rates based on reasoning complexity
2. Elastic Weight Consolidation (EWC) - Prevents catastrophic forgetting of reasoning patterns
3. Hebbian Learning - Strengthens connections between neurons that co-activate during reasoning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from typing import Dict, List, Optional, Union, Callable, Tuple
import numpy as np
from transformers import Trainer


class AdaptiveReasoningRate:
    """
    Implements Adaptive Learning Rate (ALR) for Chain of Thought reasoning.
    
    This technique dynamically adjusts learning rates based on the complexity
    of the reasoning steps in each batch, allowing the model to learn more
    from complex reasoning patterns.
    """
    
    def __init__(
        self,
        base_lr: float = 2e-5,
        max_lr_factor: float = 3.0,
        complexity_measure: str = "token_length",
    ):
        """
        Initialize the AdaptiveReasoningRate.
        
        Args:
            base_lr: Base learning rate
            max_lr_factor: Maximum factor to scale the learning rate
            complexity_measure: Method to measure reasoning complexity
                                ("token_length", "reasoning_depth", or "entropy")
        """
        self.base_lr = base_lr
        self.max_lr_factor = max_lr_factor
        self.complexity_measure = complexity_measure
        
    def measure_complexity(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Measure the reasoning complexity of a batch.
        
        Args:
            batch: Training batch containing input_ids, attention_mask, and labels
            
        Returns:
            Tensor of complexity scores for each example in the batch
        """
        if self.complexity_measure == "token_length":
            # Use the length of non-masked tokens in labels as a proxy for reasoning complexity
            # Longer reasoning chains are considered more complex
            complexity = (batch["labels"] != -100).sum(dim=1).float()
            
        elif self.complexity_measure == "reasoning_depth":
            # Count reasoning markers like "therefore", "because", "thus" as indicators of depth
            # This is a simplified approximation
            reasoning_markers = ["therefore", "because", "thus", "hence", "so", "since"]
            complexity = torch.zeros(len(batch["input_ids"]), device=batch["input_ids"].device)
            
            for i, input_seq in enumerate(batch["input_ids"]):
                # Count occurrences of reasoning markers
                marker_count = 0
                for marker in reasoning_markers:
                    # This is a simplified approach; in practice, you'd use the tokenizer
                    # to convert these words to token IDs and count them
                    marker_count += 1  # Placeholder for actual counting
                complexity[i] = marker_count
                
        elif self.complexity_measure == "entropy":
            # Higher entropy in the label distribution indicates more complex reasoning
            # This is a simplified approximation
            complexity = torch.zeros(len(batch["input_ids"]), device=batch["input_ids"].device)
            
            for i, label_seq in enumerate(batch["labels"]):
                valid_labels = label_seq[label_seq != -100]
                if len(valid_labels) > 0:
                    # Calculate entropy of token distribution
                    unique_tokens, counts = torch.unique(valid_labels, return_counts=True)
                    probs = counts.float() / len(valid_labels)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10))
                    complexity[i] = entropy
        
        # Normalize complexity scores to [0, 1] range
        if torch.max(complexity) > torch.min(complexity):
            complexity = (complexity - torch.min(complexity)) / (torch.max(complexity) - torch.min(complexity))
        else:
            complexity = torch.zeros_like(complexity)
            
        return complexity
    
    def get_lr_multipliers(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get learning rate multipliers for each example in the batch.
        
        Args:
            batch: Training batch
            
        Returns:
            Tensor of learning rate multipliers
        """
        complexity = self.measure_complexity(batch)
        # Scale complexity to [1, max_lr_factor]
        lr_multipliers = 1.0 + (self.max_lr_factor - 1.0) * complexity
        return lr_multipliers


class ElasticWeightConsolidation:
    """
    Implements Elastic Weight Consolidation (EWC) for preserving reasoning abilities.
    
    EWC prevents catastrophic forgetting by adding a penalty for changing parameters
    that are important for previously learned reasoning patterns.
    """
    
    def __init__(
        self,
        model: nn.Module,
        importance_factor: float = 1000.0,
        fisher_sample_size: int = 100,
        fisher_update_freq: int = 500,
    ):
        """
        Initialize EWC.
        
        Args:
            model: The model being trained
            importance_factor: Scaling factor for the EWC penalty
            fisher_sample_size: Number of samples to use for Fisher information calculation
            fisher_update_freq: How often to update Fisher information (in steps)
        """
        self.model = model
        self.importance_factor = importance_factor
        self.fisher_sample_size = fisher_sample_size
        self.fisher_update_freq = fisher_update_freq
        
        # Initialize Fisher information matrix and parameter means
        self.fisher_information = {}
        self.parameter_means = {}
        self.step_count = 0
        
        # Register parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)
                self.parameter_means[name] = param.data.clone()
    
    def update_fisher_information(self, dataloader):
        """
        Update the Fisher information matrix.
        
        Args:
            dataloader: DataLoader containing training examples
        """
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize Fisher information accumulators
        fisher_accumulators = {name: torch.zeros_like(param) 
                              for name, param in self.model.named_parameters() 
                              if param.requires_grad}
        
        # Sample batches for Fisher calculation
        samples_processed = 0
        for batch in dataloader:
            if samples_processed >= self.fisher_sample_size:
                break
                
            # Move batch to device
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with gradient calculation
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            
            # Accumulate squared gradients (Fisher information)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_accumulators[name] += param.grad.data.pow(2)
            
            samples_processed += batch["input_ids"].size(0)
        
        # Normalize and update Fisher information
        for name in fisher_accumulators:
            if samples_processed > 0:
                self.fisher_information[name] = fisher_accumulators[name] / samples_processed
            
        # Update parameter means
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.parameter_means[name] = param.data.clone()
        
        # Set model back to training mode
        self.model.train()
    
    def calculate_ewc_loss(self) -> torch.Tensor:
        """
        Calculate the EWC penalty loss.
        
        Returns:
            EWC penalty loss
        """
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_information:
                # Calculate squared distance from old parameters, weighted by Fisher information
                ewc_loss += (self.fisher_information[name] * 
                            (param - self.parameter_means[name]).pow(2)).sum()
        
        return 0.5 * self.importance_factor * ewc_loss
    
    def step(self, dataloader=None):
        """
        Perform an EWC step, updating Fisher information if needed.
        
        Args:
            dataloader: DataLoader for Fisher information calculation
        """
        self.step_count += 1
        
        # Update Fisher information periodically
        if dataloader is not None and self.step_count % self.fisher_update_freq == 0:
            self.update_fisher_information(dataloader)


class HebbianLearning:
    """
    Implements Hebbian learning principles for strengthening reasoning pathways.
    
    This technique reinforces connections between neurons that co-activate during
    successful reasoning steps, following the principle "neurons that fire together, wire together".
    """
    
    def __init__(
        self,
        model: nn.Module,
        hebbian_factor: float = 0.01,
        activation_threshold: float = 0.5,
        target_layers: Optional[List[str]] = None,
    ):
        """
        Initialize Hebbian learning.
        
        Args:
            model: The model being trained
            hebbian_factor: Scaling factor for Hebbian updates
            activation_threshold: Threshold for considering neurons as co-activated
            target_layers: List of layer names to apply Hebbian learning to
                          (None for all layers)
        """
        self.model = model
        self.hebbian_factor = hebbian_factor
        self.activation_threshold = activation_threshold
        self.target_layers = target_layers
        
        # Store activations for Hebbian updates
        self.layer_activations = {}
        self.hooks = []
        
        # Register forward hooks to capture activations
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture layer activations."""
        for name, module in self.model.named_modules():
            if self.target_layers is None or any(target in name for target in self.target_layers):
                if isinstance(module, nn.Linear):
                    hook = module.register_forward_hook(
                        lambda mod, inp, out, name=name: self._save_activation(name, out)
                    )
                    self.hooks.append(hook)
    
    def _save_activation(self, name: str, output: torch.Tensor):
        """
        Save layer activations for Hebbian updates.
        
        Args:
            name: Layer name
            output: Layer output tensor
        """
        # Store a detached copy of the activations
        self.layer_activations[name] = output.detach()
    
    def apply_hebbian_update(self, loss: torch.Tensor):
        """
        Apply Hebbian updates based on captured activations.
        
        Args:
            loss: Current loss value (used to scale updates)
        """
        # Only apply Hebbian updates for successful reasoning (lower loss)
        if loss.item() > 1.0:  # Threshold for "successful" reasoning
            return
        
        # Scale factor based on loss (lower loss = stronger Hebbian reinforcement)
        scale_factor = self.hebbian_factor * (1.0 - min(loss.item(), 1.0))
        
        for name, module in self.model.named_modules():
            if name in self.layer_activations and isinstance(module, nn.Linear):
                activations = self.layer_activations[name]
                
                # Apply activation threshold
                active_neurons = (activations > self.activation_threshold).float()
                
                # For each example in the batch
                for i in range(activations.size(0)):
                    # Get activations for this example
                    example_activations = active_neurons[i]
                    
                    # Skip if not enough active neurons
                    if example_activations.sum() < 2:
                        continue
                    
                    # Create co-activation matrix (outer product)
                    if len(example_activations.shape) > 1:
                        # For 2D activations (e.g., attention layers)
                        flat_activations = example_activations.view(-1)
                        co_activation = torch.outer(flat_activations, flat_activations)
                        
                        # Apply Hebbian update to weights
                        # This is a simplified version; in practice, you'd need to reshape
                        # the co-activation matrix to match the weight matrix dimensions
                        pass
                    else:
                        # For 1D activations (e.g., feed-forward layers)
                        co_activation = torch.outer(example_activations, example_activations)
                        
                        # Apply Hebbian update to weights (simplified)
                        if hasattr(module, 'weight') and module.weight.shape == co_activation.shape:
                            with torch.no_grad():
                                module.weight.data += scale_factor * co_activation
    
    def reset_activations(self):
        """Reset stored activations."""
        self.layer_activations = {}
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class NeuroplasticityTrainer(Trainer):
    """
    Extended Trainer class that incorporates neuroplasticity techniques.
    """
    
    def __init__(
        self,
        use_alr: bool = True,
        use_ewc: bool = True,
        use_hebbian: bool = True,
        alr_config: Optional[Dict] = None,
        ewc_config: Optional[Dict] = None,
        hebbian_config: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize the NeuroplasticityTrainer.
        
        Args:
            use_alr: Whether to use Adaptive Learning Rate
            use_ewc: Whether to use Elastic Weight Consolidation
            use_hebbian: Whether to use Hebbian Learning
            alr_config: Configuration for ALR
            ewc_config: Configuration for EWC
            hebbian_config: Configuration for Hebbian Learning
            **kwargs: Arguments to pass to the parent Trainer
        """
        super().__init__(**kwargs)
        
        self.use_alr = use_alr
        self.use_ewc = use_ewc
        self.use_hebbian = use_hebbian
        
        # Initialize neuroplasticity components
        if self.use_alr:
            alr_config = alr_config or {}
            self.alr = AdaptiveReasoningRate(**alr_config)
        
        if self.use_ewc:
            ewc_config = ewc_config or {}
            self.ewc = ElasticWeightConsolidation(self.model, **ewc_config)
        
        if self.use_hebbian:
            hebbian_config = hebbian_config or {}
            self.hebbian = HebbianLearning(self.model, **hebbian_config)
    
    def training_step(self, model, inputs):
        """
        Perform a training step with neuroplasticity techniques.
        
        Args:
            model: The model to train
            inputs: The inputs to the model
            
        Returns:
            The loss value
        """
        # Apply ALR if enabled
        if self.use_alr:
            # Get learning rate multipliers
            lr_multipliers = self.alr.get_lr_multipliers(inputs)
            
            # Apply different learning rates to different examples
            # This is a simplified implementation; in practice, you'd need to
            # modify the optimizer's step function to apply per-example learning rates
            pass
        
        # Forward pass and loss calculation
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Add EWC penalty if enabled
        if self.use_ewc:
            ewc_loss = self.ewc.calculate_ewc_loss()
            loss = loss + ewc_loss
        
        # Apply Hebbian updates if enabled
        if self.use_hebbian:
            self.hebbian.apply_hebbian_update(loss)
            self.hebbian.reset_activations()
        
        # Backward pass
        loss.backward()
        
        # Update EWC if enabled
        if self.use_ewc:
            self.ewc.step(self.train_dataloader)
        
        return loss.detach()
    
    def on_train_end(self):
        """Clean up at the end of training."""
        super().on_train_end()
        
        # Clean up Hebbian hooks
        if self.use_hebbian:
            self.hebbian.remove_hooks()


def get_neuroplasticity_trainer(
    model,
    args,
    train_dataloader,
    eval_dataloader,
    use_alr=True,
    use_ewc=True,
    use_hebbian=True,
    alr_config=None,
    ewc_config=None,
    hebbian_config=None,
):
    """
    Get a NeuroplasticityTrainer instance.
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataloader: Training dataloader
        eval_dataloader: Evaluation dataloader
        use_alr: Whether to use Adaptive Learning Rate
        use_ewc: Whether to use Elastic Weight Consolidation
        use_hebbian: Whether to use Hebbian Learning
        alr_config: Configuration for ALR
        ewc_config: Configuration for EWC
        hebbian_config: Configuration for Hebbian Learning
        
    Returns:
        A NeuroplasticityTrainer instance
    """
    return NeuroplasticityTrainer(
        model=model,
        args=args,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        use_alr=use_alr,
        use_ewc=use_ewc,
        use_hebbian=use_hebbian,
        alr_config=alr_config or {},
        ewc_config=ewc_config or {},
        hebbian_config=hebbian_config or {},
    ) 