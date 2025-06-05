import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.logger import DistributedLogger

logger = DistributedLogger('data_loader')

class TextDataset(Dataset):
    def __init__(self, texts, labels, model_name='bert-base-uncased', max_length=512):
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info("Pre-tokenizing dataset...")
        self.encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        logger.info(f"Tokenized {len(texts)} samples")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }