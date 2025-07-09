import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
#import google.generativeai as genai
#genai.configure(api_key="AIzaSyBam890jDotYjjR6k7qJjQ8y_4Q_TEX-Fs")
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import joblib
import torch
import ipywidgets as widgets
from IPython.display import display
from sklearn.model_selection import train_test_split
from torch import device
from transformers import RobertaTokenizer, RobertaForSequenceClassification


df = pd.read_csv("fraud_call_sample_1000.csv", encoding='latin-1')[['label','transcription']]

# Convert labels to binary (if model expects 0/1)
df['label'] = df['label'].map({'fraud': 1, 'normal': 0})

X_train, X_test, y_train, y_test = train_test_split(df['transcription'], df['label'], test_size=0.2, random_state=42)


tokenizer_roberta = RobertaTokenizer.from_pretrained('roberta-spam-detector')
model_roberta = RobertaForSequenceClassification.from_pretrained('roberta-spam-detector')

texts = X_test.tolist()
inputs = tokenizer_roberta(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

# Inference
with torch.no_grad():
    outputs = model_roberta(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)

# Convert predictions to list
predicted_labels = predictions.cpu().numpy().tolist()

print("Confusion Matrix :\n", confusion_matrix(y_test, predicted_labels))
print("\nClassification Report of RoBERTa model trained and fine-tuned using Kaggle audio spam detection datasets:\n", classification_report(y_test, predicted_labels))
print("Accuracy Score:", accuracy_score(y_test, predicted_labels))