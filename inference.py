import os
import argparse
import torch
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)
from peft import PeftModel
from neuroplasticity import DynamicTemperatureScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-3B")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-3B", help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained adapter")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of new tokens to generate")
    parser.add_argument("--use_dynamic_temp", action="store_true", help="Use dynamic temperature scheduling")
    parser.add_argument("--base_temp", type=float, default=0.7, help="Base temperature for generation")
    parser.add_argument("--min_temp", type=float, default=0.5, help="Minimum temperature")
    parser.add_argument("--max_temp", type=float, default=1.2, help="Maximum temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Input/output arguments
    parser.add_argument("--input_file", type=str, help="Path to file with input prompts (one per line)")
    parser.add_argument("--output_file", type=str, help="Path to output file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load model and tokenizer with quantization"""
    logger.info(f"Loading base model: {args.base_model}")
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load adapter
    logger.info(f"Loading adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def format_prompt(question):
    """Format the input prompt for CoT reasoning"""
    return f"Question: {question}\n\nLet me think through this step by step:\n"

def generate_response(model, tokenizer, prompt, args, temp_scheduler=None):
    """Generate a response with optional dynamic temperature scheduling"""
    # Determine temperature
    temperature = args.base_temp
    if args.use_dynamic_temp and temp_scheduler:
        temperature = temp_scheduler.get_temperature(prompt)
        logger.info(f"Using dynamic temperature: {temperature:.2f}")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Configure generation parameters
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=temperature,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
        )
    
    # Decode and return response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the prompt)
    response = response[len(prompt):]
    
    return response

def interactive_mode(model, tokenizer, args, temp_scheduler):
    """Run in interactive mode"""
    logger.info("Running in interactive mode. Type 'exit' to quit.")
    
    while True:
        question = input("\nEnter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
            
        prompt = format_prompt(question)
        logger.info("Generating response...")
        
        response = generate_response(model, tokenizer, prompt, args, temp_scheduler)
        
        print("\n" + "="*50)
        print("RESPONSE:")
        print(response)
        print("="*50)

def process_file(model, tokenizer, args, temp_scheduler):
    """Process inputs from a file"""
    logger.info(f"Processing inputs from: {args.input_file}")
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]
    
    responses = []
    for i, question in enumerate(questions):
        logger.info(f"Processing question {i+1}/{len(questions)}")
        
        prompt = format_prompt(question)
        response = generate_response(model, tokenizer, prompt, args, temp_scheduler)
        
        responses.append({
            "question": question,
            "response": response
        })
    
    # Write responses to output file
    if args.output_file:
        logger.info(f"Writing responses to: {args.output_file}")
        with open(args.output_file, "w", encoding="utf-8") as f:
            for item in responses:
                f.write(f"QUESTION: {item['question']}\n\n")
                f.write(f"RESPONSE: {item['response']}\n\n")
                f.write("="*50 + "\n\n")
    
    return responses

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Initialize temperature scheduler if needed
    temp_scheduler = None
    if args.use_dynamic_temp:
        temp_scheduler = DynamicTemperatureScheduler(
            base_temp=args.base_temp,
            min_temp=args.min_temp,
            max_temp=args.max_temp
        )
        logger.info("Using dynamic temperature scheduling")
    
    # Run in interactive mode or process file
    if args.interactive:
        interactive_mode(model, tokenizer, args, temp_scheduler)
    elif args.input_file:
        process_file(model, tokenizer, args, temp_scheduler)
    else:
        logger.error("Either --interactive or --input_file must be specified")

if __name__ == "__main__":
    main() 