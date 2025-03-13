# Fault Tolerant Training

A distributed training framework with fault tolerance capabilities for transformer models.

## Features
- Distributed training using PyTorch DDP
- Fault tolerance with checkpoint management
- Training monitoring with Prometheus
- Support for BERT-based models

## Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Start training with 2 GPUs
torchrun --nproc_per_node=2 src/train.py
```

## Project Structure
- `src/`: Source code directory
  - `models/`: Model architecture definitions
  - `utils/`: Utility functions and classes
  - `train.py`: Main training script
- `config/`: Configuration files
- `requirements.txt`: Project dependencies

## Configuration
Update `config/config.yaml` to modify:
- Model parameters
- Training settings
- Data configurations