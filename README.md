# RoBERTa-Based AI-Generated Text Detector

## Table of Contents
- [Overview](#overview)  
- [Features](#features)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Repository Structure](#repository-structure)  
- [Usage](#usage)  
  - [Training](#training)  
  - [Evaluation](#evaluation)  
  - [Inference](#inference)  
- [Configuration](#configuration)  
- [License](#license)  

## Overview
This project implements a binary classifier to distinguish **human-written** text from **AI-generated** text.  
It uses a custom RoBERTa model that dynamically weights the representations of the *prompt* (question) and *text* (answer) segments.

## Features
- **Custom RoBERTa architecture** with weighted pooling  
- **Two inference modes**  
  - **QA mode**: 75 % question, 25 % answer  
  - **Text mode**: 100 % answer (pure-text classification)  
- **Training utilities**: length-penalty loss, gradient accumulation, checkpointing  
- **Batch evaluation** across multiple domains with precision/recall/F1/accuracy  
- **Interactive CLI** for quick, ad-hoc predictions  

## Requirements
```
python==3.11
pandas>=2.2.1
torch>=2.2.2
transformers>=4.31.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

## Installation
```bash
git clone https://github.com/yourusername/roberta-ai-detector.git
cd roberta-ai-detector

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Structure
```
.
├── load.py          # Inference CLI (QA / text modes)
├── train.py         # Training loop, data loading, checkpointing
├── val.py           # Batch evaluation script
├── requirements.txt # Python package requirements
└── README.md        # This documentation
```

## Usage
### Training
1. Create `data.csv` with columns  
   `problem`, `answer`, `label` (0 = human, 1 = AI).
2. Run:
   ```bash
   python train.py \
     --data-path data.csv \
     --output-dir ./model \
     --epochs 3 \
     --batch-size 16
   ```
   The best checkpoint and tokenizer are saved to **`./model`**.

### Evaluation
```bash
python val.py \
  --model-path ./model \
  --data-dir ./validation_data \
  --results-file result.csv
```
- Expects one or more `.jsonl` files in `validation_data`, each with  
  `prompt`, `human_text`, `machine_text`, `label`.
- Appends metrics for every file and mode to `result.csv`.

### Inference
```bash
python load.py
```
- Choose **QA** or **text** mode when prompted.  
- Enter your input; the script prints **AI** or **Human**.

## Configuration
Key hyperparameters can be set via CLI flags or env vars:
- `--max-len` (max token length)  
- `--learning-rate`  
- `--accumulation-steps`  
- `CUDA_VISIBLE_DEVICES` for multi-GPU selection  

See each script header for full argument lists and defaults.

## License
MIT License – use, modify, and distribute freely for research or production.
