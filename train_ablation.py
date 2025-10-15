"""
Unified ablation training script for neuroplasticity mechanisms.
Supports training with individual mechanisms: EWC only, ALR only, or Hebbian only.
"""
import os
import sys

# Import the main train function and arg parser
from train import train, parse_args, logger

def main():
    args = parse_args()

    # Check which ablation mode is specified
    ablation_mode = os.environ.get('ABLATION_MODE', None)

    if ablation_mode:
        logger.info(f"Running ablation study: {ablation_mode} only")

        # Reset all neuroplasticity flags
        args.use_alr = False
        args.use_ewc = False
        args.use_hebbian = False

        # Enable only the specified mechanism
        if ablation_mode == "alr":
            args.use_alr = True
            args.output_dir = args.output_dir.replace("output", "output_alr")
        elif ablation_mode == "ewc":
            args.use_ewc = True
            args.output_dir = args.output_dir.replace("output", "output_ewc")
        elif ablation_mode == "hebbian":
            args.use_hebbian = True
            args.output_dir = args.output_dir.replace("output", "output_hebbian")
        else:
            logger.error(f"Unknown ablation mode: {ablation_mode}")
            logger.error("Valid modes: alr, ewc, hebbian")
            sys.exit(1)

    # Run training with modified args
    train(args)

if __name__ == "__main__":
    main()
