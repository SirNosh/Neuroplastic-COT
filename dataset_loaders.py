"""
Dataset loaders for continual learning experiments.

Supports three mathematical domains:
1. Arithmetic (MAWPS/GSM8K filtered)
2. Algebra (MATH dataset)
3. Geometry (MATH dataset)
"""

import logging
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContinualDatasetLoader:
    """Loads datasets for continual learning across mathematical domains."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)

    def load_arithmetic_data(self, max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Load arithmetic dataset (GSM8K filtered for basic arithmetic).

        Returns:
            train_data, test_data
        """
        logger.info("Loading arithmetic dataset (GSM8K - basic operations)...")

        try:
            # Load GSM8K
            dataset = load_dataset("openai/gsm8k", "main")

            # Filter for basic arithmetic problems (heuristic: shorter solutions, basic ops)
            def is_basic_arithmetic(example):
                answer = example.get("answer", "")
                # Basic arithmetic usually has simpler solutions
                # Look for problems with fewer steps
                steps = answer.count("\n")
                return steps <= 3  # Simple problems typically have 1-3 steps

            train_arithmetic = [ex for ex in dataset["train"] if is_basic_arithmetic(ex)]
            test_arithmetic = [ex for ex in dataset["test"] if is_basic_arithmetic(ex)]

            if max_samples:
                train_arithmetic = train_arithmetic[:max_samples]
                test_arithmetic = test_arithmetic[:min(max_samples // 4, len(test_arithmetic))]

            logger.info(f"Loaded {len(train_arithmetic)} train, {len(test_arithmetic)} test arithmetic problems")

            return train_arithmetic, test_arithmetic

        except Exception as e:
            logger.error(f"Error loading arithmetic data: {e}")
            # Fallback: use all GSM8K
            logger.info("Falling back to full GSM8K dataset...")
            dataset = load_dataset("openai/gsm8k", "main")
            train_data = list(dataset["train"])
            test_data = list(dataset["test"])

            if max_samples:
                train_data = train_data[:max_samples]
                test_data = test_data[:min(max_samples // 4, len(test_data))]

            return train_data, test_data

    def load_algebra_data(self, max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Load algebra dataset (MATH dataset - Algebra section).

        Returns:
            train_data, test_data
        """
        logger.info("Loading algebra dataset (MATH - Algebra)...")

        try:
            # Load MATH dataset
            dataset = load_dataset("lighteval/MATH", "all")

            # Filter for algebra problems
            def is_algebra(example):
                subject = example.get("type", "").lower()
                return "algebra" in subject

            # Split into train/test (80/20)
            algebra_problems = [ex for ex in dataset["train"] if is_algebra(ex)]
            random.shuffle(algebra_problems)

            split_idx = int(len(algebra_problems) * 0.8)
            train_algebra = algebra_problems[:split_idx]
            test_algebra = algebra_problems[split_idx:]

            if max_samples:
                train_algebra = train_algebra[:max_samples]
                test_algebra = test_algebra[:min(max_samples // 4, len(test_algebra))]

            logger.info(f"Loaded {len(train_algebra)} train, {len(test_algebra)} test algebra problems")

            return train_algebra, test_algebra

        except Exception as e:
            logger.error(f"Error loading algebra data: {e}")
            logger.warning("MATH dataset not available. Using GSM8K as fallback for algebra.")
            # Fallback to GSM8K filtered differently
            return self._fallback_algebra()

    def _fallback_algebra(self) -> Tuple[List[Dict], List[Dict]]:
        """Fallback algebra data using GSM8K problems with equations."""
        dataset = load_dataset("openai/gsm8k", "main")

        def has_equations(example):
            text = example.get("question", "") + example.get("answer", "")
            # Look for algebraic indicators
            return any(indicator in text.lower() for indicator in ["equation", "solve for", "find x", "variable"])

        train_data = [ex for ex in dataset["train"] if has_equations(ex)][:1000]
        test_data = [ex for ex in dataset["test"] if has_equations(ex)][:250]

        logger.info(f"Fallback: {len(train_data)} train, {len(test_data)} test algebra-like problems")
        return train_data, test_data

    def load_geometry_data(self, max_samples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Load geometry dataset (MATH dataset - Geometry section).

        Returns:
            train_data, test_data
        """
        logger.info("Loading geometry dataset (MATH - Geometry)...")

        try:
            # Load MATH dataset
            dataset = load_dataset("lighteval/MATH", "all")

            # Filter for geometry problems
            def is_geometry(example):
                subject = example.get("type", "").lower()
                return "geometry" in subject

            # Split into train/test (80/20)
            geometry_problems = [ex for ex in dataset["train"] if is_geometry(ex)]
            random.shuffle(geometry_problems)

            split_idx = int(len(geometry_problems) * 0.8)
            train_geometry = geometry_problems[:split_idx]
            test_geometry = geometry_problems[split_idx:]

            if max_samples:
                train_geometry = train_geometry[:max_samples]
                test_geometry = test_geometry[:min(max_samples // 4, len(test_geometry))]

            logger.info(f"Loaded {len(train_geometry)} train, {len(test_geometry)} test geometry problems")

            return train_geometry, test_geometry

        except Exception as e:
            logger.error(f"Error loading geometry data: {e}")
            logger.warning("MATH dataset not available. Using GSM8K as fallback for geometry.")
            return self._fallback_geometry()

    def _fallback_geometry(self) -> Tuple[List[Dict], List[Dict]]:
        """Fallback geometry data using GSM8K problems with geometric terms."""
        dataset = load_dataset("openai/gsm8k", "main")

        def has_geometry(example):
            text = example.get("question", "") + example.get("answer", "")
            # Look for geometric indicators
            return any(indicator in text.lower() for indicator in
                      ["angle", "triangle", "circle", "square", "rectangle", "area", "perimeter", "radius"])

        train_data = [ex for ex in dataset["train"] if has_geometry(ex)][:800]
        test_data = [ex for ex in dataset["test"] if has_geometry(ex)][:200]

        logger.info(f"Fallback: {len(train_data)} train, {len(test_data)} test geometry-like problems")
        return train_data, test_data

    def load_all_phases(self, max_samples_per_phase: Optional[int] = None) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
        """
        Load all three phases of continual learning data.

        Returns:
            Dictionary mapping phase names to (train_data, test_data) tuples
        """
        logger.info("Loading all continual learning phases...")

        phases = {
            "arithmetic": self.load_arithmetic_data(max_samples_per_phase),
            "algebra": self.load_algebra_data(max_samples_per_phase),
            "geometry": self.load_geometry_data(max_samples_per_phase)
        }

        # Log summary
        for phase_name, (train, test) in phases.items():
            logger.info(f"{phase_name.capitalize()}: {len(train)} train, {len(test)} test samples")

        return phases

    def format_for_training(self, data: List[Dict], domain: str = "arithmetic") -> List[Dict]:
        """
        Format dataset for training with CoT prompts.

        Args:
            data: Raw dataset
            domain: Domain name for context

        Returns:
            Formatted data with prompts
        """
        formatted = []

        for item in data:
            # Extract question and answer based on dataset format
            if "question" in item and "answer" in item:
                # GSM8K format
                question = item["question"]
                answer = item["answer"]
            elif "problem" in item and "solution" in item:
                # MATH format
                question = item["problem"]
                answer = item["solution"]
            else:
                logger.warning(f"Unknown format for item: {item.keys()}")
                continue

            # Create CoT prompt
            prompt = f"Solve this {domain} problem step by step:\n\nQuestion: {question}\n\nLet me think through this step by step:\n"

            formatted.append({
                "prompt": prompt,
                "completion": answer,
                "question": question,
                "answer": answer,
                "domain": domain
            })

        return formatted


if __name__ == "__main__":
    # Test the loader
    loader = ContinualDatasetLoader(seed=42)

    print("Testing dataset loaders...")
    print("=" * 60)

    # Test each phase
    arithmetic_train, arithmetic_test = loader.load_arithmetic_data(max_samples=100)
    print(f"✓ Arithmetic: {len(arithmetic_train)} train, {len(arithmetic_test)} test")

    algebra_train, algebra_test = loader.load_algebra_data(max_samples=100)
    print(f"✓ Algebra: {len(algebra_train)} train, {len(algebra_test)} test")

    geometry_train, geometry_test = loader.load_geometry_data(max_samples=100)
    print(f"✓ Geometry: {len(geometry_train)} train, {len(geometry_test)} test")

    print("=" * 60)
    print("All dataset loaders working correctly!")

    # Show example
    print("\nExample formatted data (Arithmetic):")
    formatted = loader.format_for_training(arithmetic_train[:1], "arithmetic")
    print(formatted[0]["prompt"][:200] + "...")
