import os
import torch
import torch.distributed as dist
import torch.nn as nn
from prometheus_client import start_http_server, Summary, Counter, Histogram, Gauge
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from models.model import TransformerModel
from utils.data_loader import TextDataset
from utils.logger import DistributedLogger
from datasets import load_dataset

logger = DistributedLogger('train')

# Create a metric to track training time
TRAINING_TIME = Summary('training_time_seconds', 'Time spent training')
BATCH_PROCESSED = Counter('batches_processed_total', 'Total processed batches')
LOSS_HISTOGRAM = Histogram('training_loss_distribution', 'Distribution of training losses')
GPU_MEMORY_USAGE = Gauge('gpu_memory_bytes', 'GPU memory usage in bytes')
GRADIENT_NORM = Gauge('gradient_norm', 'Gradient norm')

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}_batch_{batch_idx}.pth')
    torch.save({
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, checkpoint_path)
    logger.info(f'Checkpoint saved: {checkpoint_path}')

def save_checkpoint_atomic(model, optimizer, epoch, batch_idx, loss, checkpoint_dir='checkpoints'):
    import tempfile
    import time
    import os
    import torch
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    if dist.get_rank() != 0:
        return
    checkpoint_data = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'world_size': dist.get_world_size(),
        'timestamp': time.time()
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}_batch_{batch_idx}.pth')
    with tempfile.NamedTemporaryFile(dir=checkpoint_dir, delete=False) as tmp_file:
        torch.save(checkpoint_data, tmp_file.name)
        try:
            test_checkpoint = torch.load(tmp_file.name, map_location='cpu')
            assert 'model_state_dict' in test_checkpoint
            assert 'optimizer_state_dict' in test_checkpoint
        except Exception as e:
            os.unlink(tmp_file.name)
            raise RuntimeError(f"Checkpoint validation failed: {e}")
        os.rename(tmp_file.name, checkpoint_path)
    cleanup_old_checkpoints(checkpoint_dir, keep_last=3)
    logger.info(f'Checkpoint saved atomically: {checkpoint_path}')

def cleanup_old_checkpoints(checkpoint_dir, keep_last=3):
    import glob
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pth')), key=os.path.getmtime, reverse=True)
    for ckpt in checkpoints[keep_last:]:
        try:
            os.remove(ckpt)
        except Exception as e:
            logger.warning(f'Failed to remove old checkpoint {ckpt}: {e}')

@TRAINING_TIME.time()
def train_distributed(model, train_loader, criterion, optimizer, device, epoch, checkpoint_interval=100):
    class TrainingAnomalyHandler:
        def __init__(self, max_anomalies=10, recovery_strategies=None):
            self.max_anomalies = max_anomalies
            self.anomaly_count = 0
            self.recovery_strategies = recovery_strategies or ['reduce_lr', 'reload_checkpoint']
        def handle_anomaly(self, anomaly_type, model, optimizer, epoch, batch_idx):
            self.anomaly_count += 1
            if self.anomaly_count > self.max_anomalies:
                raise RuntimeError(f"Too many anomalies ({self.anomaly_count}). Stopping training.")
            if anomaly_type == 'gradient_explosion':
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                logger.warning(f"Reduced learning rate to {param_group['lr']}")
            elif anomaly_type == 'nan_loss':
                logger.warning("Attempting to reload last checkpoint due to NaN loss")
                # Optionally reload last checkpoint here
            return True  # Continue training
    model.train()
    running_loss = 0.0
    anomaly_count = 0
    dist.barrier()
    anomaly_handler = TrainingAnomalyHandler()
    scaler = torch.cuda.amp.GradScaler()
    try:
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                with torch.autograd.detect_anomaly():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Invalid loss detected: {loss}")
                anomaly_handler.handle_anomaly('nan_loss', model, optimizer, epoch, batch_idx)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Enhanced Prometheus metrics
            BATCH_PROCESSED.inc()
            LOSS_HISTOGRAM.observe(loss.item())
            GPU_MEMORY_USAGE.set(torch.cuda.memory_allocated())
            GRADIENT_NORM.set(grad_norm)
            if grad_norm > 10.0:
                logger.warning(f"Large gradient norm detected: {grad_norm}")
                anomaly_handler.handle_anomaly('gradient_explosion', model, optimizer, epoch, batch_idx)
                anomaly_count += 1
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            if batch_idx % checkpoint_interval == 0:
                dist.barrier()
                save_checkpoint_atomic(model, optimizer, epoch, batch_idx, loss.item())
            if batch_idx % 100 == 0 and dist.get_rank() == 0:
                logger.info(f'Epoch {epoch}, Batch {batch_idx}: Loss {loss.item():.6f}, Grad Norm {grad_norm:.4f}')
    except RuntimeError as e:
        logger.error(f'Training error: {e}')
        dist.barrier()
        raise
    dist.barrier()
    total_loss = torch.tensor(running_loss, device=device)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    avg_loss = total_loss.item() / (len(train_loader) * dist.get_world_size())
    if dist.get_rank() == 0:
        logger.info(f'Epoch {epoch}: Average Loss {avg_loss:.6f}, Anomalies: {anomaly_count}')
    return avg_loss

def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints'):
    import glob
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pth'))
    if not checkpoints:
        return 0, 0  # No checkpoint found, start from epoch 0 and batch 0
    # Get the checkpoint with the highest epoch and batch number
    latest_checkpoint = max(checkpoints, key=lambda path: (int(path.split('_')[2]), int(path.split('_')[4].split('.')[0])))
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    logger.info(f"Loaded checkpoint '{latest_checkpoint}' (epoch {checkpoint['epoch']}, batch {checkpoint['batch_idx']})")
    return checkpoint['epoch'], checkpoint['batch_idx']

def create_optimized_dataloader(dataset, batch_size, world_size, rank):
    """Create optimized DataLoader for distributed training"""
    effective_batch_size = batch_size * world_size
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True
    )
    num_workers = min(8, os.cpu_count() // world_size)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else 2,
        drop_last=True
    )
    return train_loader

def main():
    # Initialize the distributed environment
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    logger.info(f"Initializing process {local_rank}/{world_size-1}")
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(local_rank)}")
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # Start Prometheus metrics server
    if local_rank == 0:
        start_http_server(8000)
    
    logger.info(f'Using device: {device}')
    
    # Model parameters
    model_name = 'bert-base-uncased'  # or any other transformer model
    num_classes = 2  # adjust based on your task
    
    # Create model
    model = TransformerModel(model_name=model_name, num_classes=num_classes).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Try to load an existing checkpoint
    start_epoch, start_batch = load_checkpoint(model, optimizer) if local_rank == 0 else (0, 0)

    # Load IMDb dataset
    dataset = load_dataset('imdb')
    texts = dataset['train']['text']
    labels = dataset['train']['label']
    
    # Create data loader
    batch_size = 16  # Per GPU
    gradient_accumulation_steps = 4  # Effective batch size = 16 * 4 * world_size
    dataset = TextDataset(texts, labels, model_name=model_name)
    train_loader = create_optimized_dataloader(dataset, batch_size, world_size, local_rank)
    
    # Training
    num_epochs = 3
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        avg_loss = train_distributed(model, train_loader, criterion, optimizer, device, epoch+1, checkpoint_interval=100)
        if local_rank == 0:
            save_checkpoint(model, optimizer, epoch+1, 0, avg_loss)

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    main()