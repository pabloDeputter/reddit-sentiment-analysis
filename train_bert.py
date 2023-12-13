import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch

# Load DataFrame
df = pd.read_pickle('merged_training.pkl')

# Split DataFrame into features and labels
X = df.drop('emotions', axis=1)
y = df['emotions']

# Split the data into train and test sets (10% for testing)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Load tokenizer and model
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Tokenize the test data
tokenized_test = tokenizer(list(X_test['text']), padding=True, truncation=True, return_tensors="pt")


# Create a PyTorch Dataset
class TestDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


test_dataset = TestDataset(tokenized_test)
test_loader = DataLoader(test_dataset, batch_size=128)  # Adjust batch size as needed

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Prediction
model.eval()
y_pred = []
labels = list(model.config.id2label.values())
with torch.no_grad():
    for batch in tqdm(test_loader, total=len(test_loader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        y_pred.extend(map(lambda x: labels[x], predictions.cpu().numpy()))

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')  # Use 'binary' for binary classification
recall = recall_score(y_test, y_pred, average='macro')  # Use 'binary' for binary classification

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
