import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch

# 1. Load dataset (Ensure the dataset has columns 'complaint' and 'category')
df = pd.read_csv('customer_complaints.csv')  # Your dataset path

# 2. Define categories and map them to integers
categories = {'Slowness': 0, 'CCCare': 1, 'Connectivity': 2, 'Coverage': 3, 'Other': 4}
df['category'] = df['category'].map(categories)

# 3. Split the data into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(df['complaint'].tolist(), df['category'].tolist(), test_size=0.2)

# 4. Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-uncased')
model = BertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-uncased', num_labels=len(categories))

# 5. Tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# 6. Create custom dataset class for PyTorch
class ComplaintDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ComplaintDataset(train_encodings, train_labels)
test_dataset = ComplaintDataset(test_encodings, test_labels)

# 7. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=16,  # batch size for training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# 8. Define evaluation metric (accuracy)
def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    return {'accuracy': accuracy_score(p.label_ids, preds)}

# 9. Initialize Trainer
trainer = Trainer(
    model=model,                         # the model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,     # compute accuracy metric
)

# 10. Train the model
trainer.train()

# 11. Save the trained model
model.save_pretrained('customer_complaint_model')
tokenizer.save_pretrained('customer_complaint_model')

# 12. Evaluate the model
results = trainer.evaluate()

print("Evaluation Results: ", results)

# 13. Prediction example
model.eval()
inputs = tokenizer("Complaint about slow service", return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)
predicted_category = list(categories.keys())[predictions.item()]
print(f"Predicted Category: {predicted_category}")
