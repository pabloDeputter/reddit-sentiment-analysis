from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_pickle("merged_training.pkl")

# Map emotions to numeric labels
label_dict = {label: index for index, label in enumerate(df['emotions'].unique())}
df['labels'] = df['emotions'].replace(label_dict)


# Custom dataset class
class EmotionDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # Make sure to access the text as a single string, not a Series
        text = str(self.data.iloc[index]['text'])
        label = self.data.iloc[index]['labels']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=200,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len



# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_dict)).to(device)

# Create dataset
dataset = EmotionDataset(df, tokenizer)
loader = DataLoader(dataset, batch_size=2)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Fine-tuning loop
for epoch in range(3):  # Number of training epochs
    print(epoch)
    for batch in loader:
        # Extract inputs
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].to(device)

        # Zero gradients
        model.zero_grad()

        # Forward pass
        outputs = model(ids, attention_mask=mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

# Save the model
model.to('cpu').save_pretrained("./my_finetuned_model")
