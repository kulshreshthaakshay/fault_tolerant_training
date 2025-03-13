# Distributed Training System For LLM Dataset w/ Fault-Tolerant Enabled
The goal of this project was to train BERT-base-uncased on the IMDB dataset in a distributed multi-GPU setting while handling failures gracefully and optimizing performance.
Training large language models (LLMs) like BERT at scale presents several challenges, including hardware failures, unstable gradients, and inefficient resource utilization. To address these issues, I built a fault-tolerant, checkpoint-optimized, and anomaly-detection enabled distributed training system for LLMs. This setup ensures efficient, scalable, and robust training across multiple GPUs while minimizing potential disruptions.

## Features
✅ Distributed Training using PyTorch DDP with NCCL backend.
✅ Fault Tolerance with automatic checkpointing and recovery.
✅ Anomaly Detection to catch NaNs and unstable gradients.
✅ Performance Optimizations using AMP, gradient accumulation, and delayed gradient synchronization.
✅ Monitoring & Logging via Prometheus and structured logging.

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
- `requirements.txt`: Project dependencies
