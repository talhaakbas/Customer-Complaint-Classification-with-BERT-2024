# Customer-Complaint-Classification-with-BERT-2024
This project uses the BERT (Bidirectional Encoder Representations from Transformers) model to automatically classify customer complaints into predefined categories. By processing customer complaints in text format, the system categorizes them into one of the following categories: Slowness, CCCare, Connectivity, Coverage, or Other.

Requirements
Python 3.x

Transformers: For accessing the BERT model.

pandas: For handling data manipulation (CSV/Excel files).

scikit-learn: For data preprocessing and evaluation metrics.

torch: For running the BERT model in PyTorch.

You can install the required dependencies using pip:

pip install transformers pandas scikit-learn torch

How it Works
Data Preparation:
The dataset consists of customer complaints (text) and their corresponding categories. The dataset is read into the program and preprocessed (tokenization, padding) to be used by the BERT model.

Model Selection:
A pre-trained BERT model (dbmdz/bert-base-turkish-uncased) is used for text classification. Transfer learning is applied to fine-tune the model on the specific task of customer complaint classification.

Training the Model:
The model is fine-tuned on the labeled dataset using the Huggingface Trainer API. The model learns to classify complaints into the predefined categories based on the training data.

Prediction:
After training, the model predicts the category for new customer complaints. The output is a category label corresponding to one of the predefined complaint types.

Evaluation:
The model’s performance is evaluated based on metrics like accuracy, precision, recall, and F1-score. The evaluation helps in determining how well the model is classifying the complaints.

How to Run
Ensure all dependencies are installed.

Prepare your dataset as a CSV or Excel file with two columns: complaint (text of the customer complaint) and categories (the category label).

Train the model on your data:

python train_complaint_classifier.py

After training, use the trained model to predict the category of new customer complaints:

python predict_complaint.py

Key Features:
BERT for Text Classification: The project leverages BERT for understanding and classifying text data.

Customer Complaint Categorization: The system classifies customer complaints into categories like slowness, coverage, connectivity, etc.

Excel/CSV Output: The model's predictions are saved in Excel/CSV format for easy analysis.

Evaluation Metrics: Accuracy, precision, recall, and F1-score are calculated to assess the model's performance.

Project Files
train_complaint_classifier.py: Script for training the BERT model on the complaint dataset.

predict_complaint.py: Script for predicting the categories of new customer complaints.

model.pkl: Saved trained BERT model.

dataset.csv: Example dataset of customer complaints and their corresponding categories.

License
This project is open-source and can be modified or used for commercial purposes.

Author
Talha Akbas

