# pocket-narrator : Efficient Story Generation with Mamba

> **Master Project: Efficient Methods in Machine Learning** > *Exploring the capabilities of Small Language Models (SLMs) on the TinyStories dataset.*

## 📖 Overview
This repository contains an end-to-end implementation of the **Mamba** state-space model architecture designed for efficient language modeling. The project investigates how well small Mamba models can learn to generate coherent English stories when trained on the **TinyStories** dataset.



Key features:
- **Custom Mamba Implementation**: A PyTorch implementation of the Mamba block with selective state spaces.
- **End-to-End Pipeline**: Scripts for tokenizer training, dataset chunking, model training, and evaluation.
- **Efficiency Focus**: Linear scaling with sequence length ($O(T)$) and constant-time inference.

## 📂 Project Structure

- `pocket_narrator/models/mamba/`: Core source code.
  - `mamba_model.py`: Mamba architecture implementation (SSM, Convolution, Gating).
  - `mamba_trainer.py`: Custom training loop with gradient accumulation.
  - `mamba_evaluation.py`: Perplexity calculation and diversity metrics (Distinct-n).
- `configs/`: YAML configuration files for different dataset sizes (2k, 4k, 8k, 1M).

## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install torch transformers datasets wandb pyyaml tqdm
