import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModel, BertModel, BertTokenizer, AdamW
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import defaultdict

model_name = "bert-base-cased"
with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

class IntentClassificationDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = 25
        self.data = []
        self.intents = []
        self.intent_index = ['unknown']  # Add 'unknown' as the first intent
        self.intent_to_examples = defaultdict(list)
        for item in data:
            if item["intent"] not in self.intent_index:
                self.intent_index.append(item["intent"])
            for example in item["examples"]:
                self.data.append(example)
                intent_id = self.intent_index.index(item["intent"])
                self.intents.append(intent_id)
                self.intent_to_examples[intent_id].append(len(self.data) - 1)

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
model = IntentClassificationModel(intent_num_labels=len(dataset.intent_index), model_name=model_name)
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent = batch["intent"].to(device)

        optimizer.zero_grad()
        outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
        loss = criterion(outputs, intent)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            intent = batch["intent"].to(device)

            outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
            loss = criterion(outputs, intent)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += intent.size(0)
            correct += (predicted == intent).sum().item()

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {100 * correct / total:.2f}%")
    print()

# Save the model
torch.save(model.state_dict(), "intent_classification_model.pth")
print("Model saved successfully.")

# Test inferences with confidence threshold
def predict_intent(model, tokenizer, text, confidence_threshold=0.7):
    model.eval()
    encoded = tokenizer.encode_plus(
        text,
        max_length=25,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model({"input_ids": input_ids, "attention_mask": attention_mask})
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    if confidence.item() >= confidence_threshold:
        return dataset.intent_index[predicted.item()], confidence.item()
    else:
        return dataset.intent_index[0], confidence.item()  # Return 'unknown' for low confidence

# Load the saved model
loaded_model = IntentClassificationModel(intent_num_labels=len(dataset.intent_index), model_name=model_name)
loaded_model.load_state_dict(torch.load("intent_classification_model.pth"))
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
    "The quick brown fox jumps over the lazy dog",
]

print("\nTest Inferences (with 70% confidence threshold):")
for sentence in test_sentences:
    predicted_intent, confidence = predict_intent(loaded_model, dataset.tokenizer, sentence)
    print(f"Sentence: '{sentence}'")
    print(f"Predicted Intent: {predicted_intent}")
    print(f"Confidence: {confidence:.2%}\n")