from setuptools import setup, find_packages

setup(
    name="neuroplastic-cot",
    version="0.1.0",
    description="Neuroplasticity techniques for Chain of Thought reasoning",
    author="SirNosh",
    url="https://github.com/SirNosh/Neuroplastic-COT",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "peft>=0.4.0",
        "accelerate>=0.20.0",
        "bitsandbytes>=0.39.0",
        "wandb>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "neuroplastic-cot=neuroplastic_cot.__main__:main",
        ],
    },
    scripts=["bin/neuroplastic-cot"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
) 