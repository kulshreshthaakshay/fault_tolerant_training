import os
import torch
import torch.distributed as dist
import torch.nn as nn
from prometheus_client import start_http_server, Summary
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from models.model import TransformerModel
from utils.data_loader import TextDataset
from utils.logger import setup_logger
from datasets import load_dataset

logger = setup_logger('train')

# Create a metric to track training time
TRAINING_TIME = Summary('training_time_seconds', 'Time spent training')

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

@TRAINING_TIME.time()
def train(model, train_loader, criterion, optimizer, device, epoch, checkpoint_interval=100):
    model.train()
    running_loss = 0.0
    anomaly_count = 0
    
    try:
        with torch.autograd.detect_anomaly():
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                running_loss += loss.item()
                
                if batch_idx % checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, epoch, batch_idx, loss.item())
                
                if batch_idx % 100 == 0:
                    logger.info(f'Batch {batch_idx}: Loss {loss.item():.6f}')
    
    except RuntimeError as e:
        logger.error(f'Anomaly detected during training: {e}')
        anomaly_count += 1
        raise
        
    avg_loss = running_loss / len(train_loader)
    logger.info(f'Epoch {epoch}: Average Loss {avg_loss:.6f}')
    logger.info(f'Epoch {epoch}: Anomalies detected: {anomaly_count}')
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
    dataset = TextDataset(texts, labels, model_name=model_name)
    train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=8,  # Smaller batch size for transformers
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True
    )
    
    # Training
    num_epochs = 3
    for epoch in range(start_epoch, num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        avg_loss = train(model, train_loader, criterion, optimizer, device, epoch+1, checkpoint_interval=100)
        if local_rank == 0:
            save_checkpoint(model, optimizer, epoch+1, 0, avg_loss)

    # Clean up
    dist.destroy_process_group()

if __name__ == '__main__':
    main()