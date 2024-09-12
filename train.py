import torch
from torch import nn
from transformers import AutoModel

class IntentClassificationModel(nn.Module):
    def __init__(self, intent_num_labels=None, model_name="bert-base-cased", dropout_prob=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, intent_num_labels)

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits

# Assuming intent_map is defined
intent_model = IntentClassificationModel(intent_num_labels=len(intent_map))

# Example of how to set up optimizer and loss function
optimizer = torch.optim.Adam(intent_model.parameters())
loss_fn = nn.CrossEntropyLoss()

# Example of how to compute accuracy
def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    return (predictions == labels).float().mean()