# pocket-narrator : Efficient Story Generation with Mamba

> **Master Project: Efficient Methods in Machine Learning** > *Exploring the capabilities of Small Language Models (SLMs) on the TinyStories dataset.*

## 📖 Overview
This repository contains an end-to-end implementation of the **Mamba** state-space model architecture designed for efficient language modeling. The project investigates how well small Mamba models can learn to generate coherent English stories when trained on the **TinyStories** dataset.

## Repository Structure: 

pocket-narrator/
├── configs/                          # Configuration files (YAML)
│   ├── mamba_tinystories_2k/
│   │   ├── model.yaml
│   │   ├── tokenizer.yaml
│   │   └── training.yaml
│   ├── mamba_tinystories_1M/
│   │   ├── model.yaml
│   │   └── ...
│   └── train_tokenizer_and_lm_dataset.yaml
├── pocket_narrator/                  # Source code package
│   ├── __init__.py
│   └── models/
│       ├── __init__.py
│       └── mamba/
│           ├── __init__.py
│           ├── config_utils.py       # YAML loading utilities
│           ├── mamba_evaluation.py   # PPL calculation & metrics
│           ├── mamba_generate.py     # Story generation script
│           ├── mamba_main.py         # Main training entry point
│           ├── mamba_model.py        # Mamba architecture definition
│           ├── mamba_trainer.py      # Training loop & logic
│           └── train_tokenizer_and_lm_dataset.py # Data preprocessing
├── results/                          # Checkpoints (add to .gitignore)
├── tokenizers/                       # Saved tokenizers (add to .gitignore)
├── .gitignore
├── requirements.txt
└── README.md

