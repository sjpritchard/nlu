import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, BertModel, BertTokenizerFast, AdamW
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict
import re
from itertools import product

model_name = "bert-base-cased"

# Load data.yaml
with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

# Load entities.yaml
with open("entities.yaml", "r") as f:
    entities_data = yaml.safe_load(f)


class IntentClassificationDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name)
        self.max_length = 25
        self.data = []
        self.intents = []
        self.entities = []
        self.intent_index = ['unknown']  # Add 'unknown' as the first intent
        self.intent_to_examples = defaultdict(list)
        self.entity_synonyms = {}
        self.entity_label_set = set(['O'])

        # Process entities
        for entity in entities_data:
            entity_name = entity['entity']
            synonyms = []
            for value in entity['values']:
                synonyms.extend(value['synonyms'])
            self.entity_synonyms[entity_name] = synonyms
            # Add entity labels
            self.entity_label_set.add(f'B-{entity_name}')
            self.entity_label_set.add(f'I-{entity_name}')

        # Create entity label mappings
        self.entity_label_to_id = {label: idx for idx, label in enumerate(sorted(self.entity_label_set))}
        self.id_to_entity_label = {idx: label for label, idx in self.entity_label_to_id.items()}

        # Pattern to find entity placeholders
        pattern = re.compile(r'\[(.*?)\]')

        for item in data:
            if item["intent"] not in self.intent_index:
                self.intent_index.append(item["intent"])
            for example in item["examples"]:
                entities_in_example = pattern.findall(example)
                if entities_in_example:
                    # The example contains entities
                    entity_synonym_lists = []
                    for entity_name in entities_in_example:
                        synonyms = self.entity_synonyms[entity_name]
                        entity_synonym_lists.append(synonyms)
                    # Generate all combinations
                    for synonym_combination in product(*entity_synonym_lists):
                        new_example = example
                        entity_positions = []
                        for placeholder, synonym in zip(entities_in_example, synonym_combination):
                            new_example = new_example.replace(f'[{placeholder}]', synonym, 1)
                            entity_positions.append((synonym, placeholder))
                        self.data.append(new_example)
                        intent_id = self.intent_index.index(item["intent"])
                        self.intents.append(intent_id)
                        self.intent_to_examples[intent_id].append(len(self.data) - 1)
                        self.entities.append(entity_positions)
                else:
                    # No entities
                    self.data.append(example)
                    intent_id = self.intent_index.index(item["intent"])
                    self.intents.append(intent_id)
                    self.intent_to_examples[intent_id].append(len(self.data) - 1)
                    self.entities.append([])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        intent = self.intents[idx]
        entity_positions = self.entities[idx]
        encoded = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        offset_mapping = encoded["offset_mapping"].squeeze(0)

        # Initialize labels
        labels = []
        for idx_token, (start, end) in enumerate(offset_mapping.tolist()):
            if attention_mask[idx_token].item() == 0:
                # Padding token
                labels.append(-100)
                continue
            if start == 0 and end == 0:
                # Special token like [CLS] or [SEP]
                labels.append(-100)
                continue
            label = 'O'
            for entity_text, entity_name in entity_positions:
                entity_start = sentence.lower().find(entity_text.lower())
                entity_end = entity_start + len(entity_text)
                token_start = start
                token_end = end
                if token_start >= entity_start and token_end <= entity_end:
                    if token_start == entity_start:
                        label = f'B-{entity_name}'
                    else:
                        label = f'I-{entity_name}'
                    break
            label_id = self.entity_label_to_id.get(label, 0)
            labels.append(label_id)

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "intent": torch.tensor(intent, dtype=torch.long),
            "labels": labels,
        }


class IntentEntityClassificationModel(nn.Module):
    def __init__(
        self, intent_num_labels=None, num_entity_labels=None, model_name="bert-base-cased", dropout_prob=0.1
    ):
        super().__init__()
        self.bert: BertModel = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.intent_classifier = nn.Linear(
            self.bert.config.hidden_size, intent_num_labels
        )
        self.entity_classifier = nn.Linear(
            self.bert.config.hidden_size, num_entity_labels
        )

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        intent_logits = self.intent_classifier(pooled_output)

        sequence_output = self.dropout(sequence_output)
        entity_logits = self.entity_classifier(sequence_output)  # Shape: (batch_size, seq_length, num_entity_labels)

        return intent_logits, entity_logits


# Create dataset
dataset = IntentClassificationDataset()

# Perform stratified split
train_indices = []
val_indices = []
test_size = 0.2

for intent, example_indices in dataset.intent_to_examples.items():
    intent_train_indices, intent_val_indices = train_test_split(
        example_indices, test_size=test_size, random_state=42
    )
    train_indices.extend(intent_train_indices)
    val_indices.extend(intent_val_indices)

# Create Subset datasets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = IntentEntityClassificationModel(
    intent_num_labels=len(dataset.intent_index),
    num_entity_labels=len(dataset.entity_label_to_id),
    model_name=model_name
)
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()
entity_criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Early stopping parameters
patience = 20
best_val_loss = float('inf')
counter = 0
best_model = None

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent = batch["intent"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        intent_logits, entity_logits = model({"input_ids": input_ids, "attention_mask": attention_mask})
        intent_loss = criterion(intent_logits, intent)
        entity_loss = entity_criterion(entity_logits.view(-1, len(dataset.entity_label_to_id)), labels.view(-1))
        loss = intent_loss + entity_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    intent_correct = 0
    intent_total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent = batch["intent"].to(device)
            labels = batch["labels"].to(device)

            intent_logits, entity_logits = model({"input_ids": input_ids, "attention_mask": attention_mask})
            intent_loss = criterion(intent_logits, intent)
            entity_loss = entity_criterion(entity_logits.view(-1, len(dataset.entity_label_to_id)), labels.view(-1))
            loss = intent_loss + entity_loss
            val_loss += loss.item()

            _, intent_predicted = torch.max(intent_logits, 1)
            intent_total += intent.size(0)
            intent_correct += (intent_predicted == intent).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    intent_accuracy = 100 * intent_correct / intent_total

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Validation Loss: {avg_val_loss:.4f}")
    print(f"Intent Validation Accuracy: {intent_accuracy:.2f}%")
    print()

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
        best_model = model.state_dict()
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

# Load the best model
model.load_state_dict(best_model)

# Save the best model
torch.save(best_model, "intent_entity_classification_model.pth")
print("Best model saved successfully.")

# Test inferences with confidence threshold
def predict_intent_and_entities(model, tokenizer, text, confidence_threshold=0.7):
    model.eval()
    encoded = tokenizer.encode_plus(
        text,
        max_length=25,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offset_mapping = encoded["offset_mapping"].squeeze(0).tolist()
    
    with torch.no_grad():
        intent_logits, entity_logits = model({"input_ids": input_ids, "attention_mask": attention_mask})
        intent_probabilities = torch.nn.functional.softmax(intent_logits, dim=1)
        intent_confidence, intent_predicted = torch.max(intent_probabilities, 1)
        entity_predictions = torch.argmax(entity_logits, dim=2).squeeze(0).tolist()
    
    if intent_confidence.item() >= confidence_threshold:
        predicted_intent = dataset.intent_index[intent_predicted.item()]
    else:
        predicted_intent = dataset.intent_index[0]  # Return 'unknown' for low confidence

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0))
    entities = []
    current_entity = ""
    current_entity_label = ""
    for idx, label_id in enumerate(entity_predictions):
        label = dataset.id_to_entity_label.get(label_id, 'O')
        token = tokens[idx]
        offset = offset_mapping[idx]
        if attention_mask.squeeze(0)[idx] == 0 or label_id == -100 or offset == [0, 0]:
            continue  # Skip padding and special tokens
        if label == 'O':
            if current_entity:
                entities.append((current_entity, current_entity_label))
                current_entity = ""
                current_entity_label = ""
        else:
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_entity, current_entity_label))
                current_entity = token.replace('##', '')
                current_entity_label = label[2:]
            elif label.startswith('I-') and current_entity_label == label[2:]:
                current_entity += token.replace('##', '')
            else:
                if current_entity:
                    entities.append((current_entity, current_entity_label))
                current_entity = ""
                current_entity_label = ""
    if current_entity:
        entities.append((current_entity, current_entity_label))

    return predicted_intent, intent_confidence.item(), entities

# Load the saved model
loaded_model = IntentEntityClassificationModel(
    intent_num_labels=len(dataset.intent_index),
    num_entity_labels=len(dataset.entity_label_to_id),
    model_name=model_name
)
loaded_model.load_state_dict(torch.load("intent_entity_classification_model.pth"))
loaded_model.to(device)
loaded_model.eval()

# Test sentences
test_sentences = [
    "Close the canopy now",
    "Open the canopy please",
    "Switch to the external view",
    "canopy open",
    "canopy close",
    "external view",
    "put the canopy down",
    "baro altimeter on",
    "radar altimeter off",
    "turn on barometric altimeter",
    "enable arresting hook",
    "enable anti skid",
    "The quick brown fox jumps over the lazy dog",
]

print("\nTest Inferences (with 70% confidence threshold):")
for sentence in test_sentences:
    predicted_intent, confidence, entities = predict_intent_and_entities(
        loaded_model, dataset.tokenizer, sentence
    )
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Intent: {predicted_intent}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Entities: {entities}\n")
