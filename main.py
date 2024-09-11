from transformers import BertTokenizer
import json
import numpy as np
from typing import List, Dict


with open("sentences.json", "r") as f:
    data = json.load(f)

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
) -> Dict:
    token_ids = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0 : len(encoded)] = encoded
    attention_masks = (token_ids != 0).astype(np.int32)
    return {"input_ids": token_ids, "attention_masks": attention_masks}


encoded_train = encode_dataset(tokenizer, data, 128)
print(encoded_train["input_ids"])
print(encoded_train["attention_masks"])