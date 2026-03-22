import logging
import json
import os
import torch.distributed as dist
from datetime import datetime

class DistributedLogger:
    def __init__(self, name, log_dir='logs'):
        self.name = name
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"{name}_rank_{self.rank}")
        self.logger.setLevel(logging.INFO)
        if self.rank == 0:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        file_handler = logging.FileHandler(f'{log_dir}/rank_{self.rank}.log')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        # Setup metrics logger properly
        self.metrics_logger = logging.getLogger(f"{name}_metrics_rank_{self.rank}")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.propagate = False
        
        metrics_handler = logging.FileHandler(f'{log_dir}/metrics_rank_{self.rank}.json')
        metrics_handler.setLevel(logging.INFO)
        self.metrics_logger.addHandler(metrics_handler)
        
    def log_metrics(self, metrics_dict):
        metrics_dict.update({
            'timestamp': datetime.now().isoformat(),
            'rank': self.rank,
            'world_size': self.world_size
        })
        self.metrics_logger.info(json.dumps(metrics_dict))
    def info(self, message):
        self.logger.info(f"[Rank {self.rank}] {message}")
    def warning(self, message):
        self.logger.warning(f"[Rank {self.rank}] {message}")
    def error(self, message):
        self.logger.error(f"[Rank {self.rank}] {message}")