# Pocket Narrator 2 : Efficient Story Generation with Mamba

> **Master Project: Efficient Methods in Machine Learning** > *Exploring the capabilities of Small Language Models (SLMs) on the TinyStories dataset.*

##  Overview
This repository contains an end-to-end implementation of the **Mamba** state-space model architecture designed for efficient language modeling. The project investigates how well small Mamba models can learn to generate coherent English stories when trained on the **TinyStories** dataset.



Key features:
- **Custom Mamba Implementation**: A PyTorch implementation of the Mamba block with selective state spaces.
- **End-to-End Pipeline**: Scripts for tokenizer training, dataset chunking, model training, and evaluation.
- **Efficiency Focus**: Linear scaling with sequence length ($O(T)$) and constant-time inference.

## Project Structure

- `pocket_narrator/models/mamba/`: Core source code.
  - `mamba_model.py`: Mamba architecture implementation (SSM, Convolution, Gating).
  - `mamba_trainer.py`: Custom training loop with gradient accumulation.
  - `mamba_evaluation.py`: Perplexity calculation and diversity metrics (Distinct-n).
- `configs/`: YAML configuration files for different dataset sizes (2k, 4k, 8k, 10k, 1M).

```Plaintext
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
```
#### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
# Or manually:
pip install torch transformers datasets wandb pyyaml tqdm
```

### 1. Data Preprocessing

Train a BPE tokenizer and create a sliding-window dataset from TinyStories.

```bash
python -m pocket_narrator.models.mamba.train_tokenizer_and_lm_dataset \
        --config configs/train_tokenizer_and_lm_dataset.yaml

```
### 2. Training

Train the Mamba model. This script logs metrics to Weights & Biases automatically.
```bash
python -m pocket_narrator.models.mamba.mamba_main \
  --model_config configs/mamba_tinystories_1M/model.yaml \
  --tokenizer_config configs/mamba_tinystories_1M/tokenizer.yaml \
  --training_config configs/mamba_tinystories_1M/training.yaml \
  --num_workers 0

```

### 3. Evaluation

Evaluate a trained checkpoint for Perplexity (PPL) and generation quality.
```bash
python -m pocket_narrator.models.mamba.mamba_evaluation \
  --checkpoint results/mamba_tinystories_1M/mamba_best.pt \
  --model_config configs/mamba_tinystories_1M/model.yaml \
  --tokenizer_dir tokenizers/tinystories_1M \
  --max_examples 256

```
### 4. Text Generation

Generate stories from a custom prompt.
```bash
python -m pocket_narrator.models.mamba.mamba_generate \
  --checkpoint results/mamba_tinystories_1M/mamba_best.pt \
  --model_config configs/mamba_tinystories_1M/model.yaml \
  --tokenizer_dir tokenizers/tinystories_1M \
  --prompt "Once upon a time there was a little dragon" \
  --temperature 0.8 \
  --max_new_tokens 200

```

## Result: 
We observed a clear scaling law where increasing dataset size dramatically reduces perplexity.

-  View Full Experiment [Evaluation as .csv]([https://github.com/KosarHazrati/pocket-narrator/blob/main/wandb_export_2026-02-18T14_36_48.314%2B01_00.csv]) -  Generated [_stories 1M ](/Users/kosaralehosseini/Desktop/mambaFiles/evaluation.json)

## References
1.  **Mamba:** Gu, A., & Dao, T. (2023). [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).
2.  **TinyStories:** Eldan, R., & Li, Y. (2023). [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759).





