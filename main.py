from transformers import BertTokenizer
import json
import numpy as np
from typing import List, Dict
from transformers import AutoModel, BertModel
import torch


with open("sentences.json", "r") as f:
    data = json.load(f)

with open("intents.json", "r") as f:
    intents = json.load(f)

model_name = "bert-base-cased"
tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)

sentence = "jester set tacan to channel 12345678911"
tokenized = tokenizer.tokenize(sentence)
print(tokenized)

encoded = tokenizer.encode(sentence)
print(encoded)

print(tokenizer.vocab_size)
print(max([len(tokenizer.tokenize(x)) for x in data]))


def encode_dataset(
    tokenizer: BertTokenizer, text_sequences: List[str], max_length: int
) -> Dict[str, torch.Tensor]:
    token_ids = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence, max_length=max_length, truncation=True, padding='max_length')
        token_ids[i] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {
        "input_ids": torch.tensor(token_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long)
    }

encoded_train = encode_dataset(tokenizer, data, 128)
print(encoded_train["input_ids"].shape)
print(encoded_train["attention_mask"].shape)

print(intents)

base_bert_model: BertModel = AutoModel.from_pretrained("bert-base-cased")
print(base_bert_model.config)

# Move the tensors to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_bert_model = base_bert_model.to(device)
encoded_train = {k: v.to(device) for k, v in encoded_train.items()}

outputs = base_bert_model(**encoded_train)

# Computes features for each token in input sequence
print(outputs[0].shape) # batch size, sequence length, output size

# Vector representation of the special token [CLS], typically used as a pooled
# representation for classification tasks. This will be used as the features
# for the intent classifier
print(outputs[1].shape) # batch size, output size

print(outputs[0])
print(outputs[1])