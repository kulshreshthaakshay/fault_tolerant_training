import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransformerModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        logits = self.classifier(pooled_output)
        return logits