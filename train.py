import torch
from torch import nn
from transformers import AutoModel, BertModel, BertTokenizer
import json
import yaml

model_name = "bert-base-cased"
with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)


class IntentClassificationDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = 25
        self.data = []
        self.intents = []
        self.intent_index = []
        for item in data:
            for example in item["examples"]:
                if item["intent"] not in self.intent_index:
                    self.intent_index.append(item["intent"])
                self.data.append(example)
                self.intents.append(self.intent_index.index(item["intent"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        intent = self.intents[idx]
        encoded = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "intent": torch.tensor(intent, dtype=torch.long),
        }


class IntentClassificationModel(nn.Module):
    def __init__(
        self, intent_num_labels=None, model_name="bert-base-cased", dropout_prob=0.1
    ):
        super().__init__()
        self.bert: BertModel = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(
            self.bert.config.hidden_size, intent_num_labels
        )

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)
        return intent_logits


dataset = IntentClassificationDataset()
for item in dataset:
    print(item)
