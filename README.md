# Neuroplastic-COT

Enhancing Chain of Thought reasoning with neuroplasticity techniques for the Qwen 2.5 3B model.

## Features

- **Adaptive Learning Rate (ALR)**: Dynamically adjusts learning rates based on reasoning complexity
- **Elastic Weight Consolidation (EWC)**: Prevents catastrophic forgetting of reasoning patterns
- **Hebbian Learning**: Strengthens connections between neurons that co-activate during reasoning
- **Visualization Tools**: Visualize the effects of neuroplasticity techniques on model training
- **Integrated with Hugging Face Transformers**: Seamless integration with the Transformers library
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation

## Installation

### From GitHub

```bash
git clone https://github.com/SirNosh/Neuroplastic-COT.git
cd Neuroplastic-COT
pip install -e .
```

### Dependencies

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- Accelerate 0.20+
- BitsAndBytes 0.39+

## Usage

### Training

Train a model with neuroplasticity techniques:

```bash
neuroplastic-cot train --model_name "Qwen/Qwen2.5-3B-Instruct" --output_dir "./output" --lora --use_alr --use_ewc --use_hebbian
```

### Evaluation

Evaluate a trained model:

```bash
neuroplastic-cot evaluate --model_path "./output" --base_model "Qwen/Qwen2.5-3B-Instruct" --output_file "results.json"
```

### Demo

Run a demonstration of neuroplasticity techniques:

```bash
neuroplastic-cot demo --use_all --num_examples 10
```

Or selectively enable techniques:

```bash
neuroplastic-cot demo --use_alr --use_ewc
```

## Configuration

Configuration parameters can be found in `neuroplastic_cot/config.py`. Key parameters include:

- Model configuration (model name, quantization)
- Dataset configuration (dataset name, input/output lengths)
- Training configuration (batch size, learning rate, epochs)
- LoRA configuration (rank, alpha, dropout)
- Neuroplasticity configuration (ALR, EWC, Hebbian learning parameters)

## Project Structure

```
neuroplastic_cot/
├── __init__.py             # Package initialization
├── __main__.py             # Main entry point
├── config.py               # Configuration parameters
├── data_utils.py           # Dataset loading and processing
├── evaluate.py             # Model evaluation
├── model_utils.py          # Model loading and configuration
├── neuroplasticity.py      # Neuroplasticity techniques implementation
├── neuroplasticity_demo.py # Demo script for visualization
└── train.py                # Training script
```

## How It Works

### Adaptive Learning Rate (ALR)

ALR dynamically adjusts learning rates based on the complexity of reasoning steps in each batch. This allows the model to learn more from complex reasoning patterns and less from simpler ones.

### Elastic Weight Consolidation (EWC)

EWC prevents catastrophic forgetting by adding a penalty for changing parameters that are important for previously learned reasoning patterns. This helps the model retain knowledge while learning new patterns.

### Hebbian Learning

Hebbian learning reinforces connections between neurons that co-activate during successful reasoning steps, following the principle "neurons that fire together, wire together". This strengthens reasoning pathways in the model.

## License

MIT 