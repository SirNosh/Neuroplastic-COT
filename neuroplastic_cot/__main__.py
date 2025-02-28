"""
Main entry point for the neuroplastic-cot package.
"""

import argparse
import sys
from neuroplastic_cot.train import main as train_main
from neuroplastic_cot.evaluate import main as evaluate_main
from neuroplastic_cot.neuroplasticity_demo import train_with_neuroplasticity


def main():
    """Main entry point for the package."""
    parser = argparse.ArgumentParser(
        description="Neuroplastic Chain of Thought - Enhancing reasoning with neuroplasticity techniques",
        usage="""neuroplastic-cot <command> [<args>]

Commands:
   train       Train a model with neuroplasticity techniques
   evaluate    Evaluate a trained model
   demo        Run a demonstration of neuroplasticity techniques
"""
    )
    parser.add_argument("command", help="Command to run")
    
    # Parse the command
    args = parser.parse_args(sys.argv[1:2])
    
    # Run the appropriate command
    if args.command == "train":
        # Remove the command from sys.argv and run train
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        train_main()
    elif args.command == "evaluate":
        # Remove the command from sys.argv and run evaluate
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        evaluate_main()
    elif args.command == "demo":
        # Remove the command from sys.argv and run demo
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        # Parse demo arguments
        demo_parser = argparse.ArgumentParser(description="Demo for neuroplasticity techniques")
        demo_parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct", 
                            help="Model to use for the demo")
        demo_parser.add_argument("--use_alr", action="store_true", help="Use Adaptive Learning Rate")
        demo_parser.add_argument("--use_ewc", action="store_true", help="Use Elastic Weight Consolidation")
        demo_parser.add_argument("--use_hebbian", action="store_true", help="Use Hebbian Learning")
        demo_parser.add_argument("--use_all", action="store_true", help="Use all neuroplasticity techniques")
        demo_parser.add_argument("--num_examples", type=int, default=10, 
                            help="Number of examples to use for the demo")
        demo_parser.add_argument("--output_dir", type=str, default="neuroplasticity_demo_results",
                            help="Directory to save demo results")
        demo_parser.add_argument("--seed", type=int, default=42, help="Random seed")
        
        demo_args = demo_parser.parse_args(sys.argv[1:])
        train_with_neuroplasticity(demo_args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main() 