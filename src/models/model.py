import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class TransformerModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_classes=2, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Use proper pooling
        self.pooler = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.Tanh()
        )
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        # Initialize classifier weights
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # Use pooler output (already Linear+Tanh in BERT) or mean pooling with custom pooler
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Mean pooling with attention mask + custom pooler for non-BERT models
            hidden_states = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            pooled_output = torch.sum(hidden_states * attention_mask_expanded, 1) / torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            pooled_output = self.pooler(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits