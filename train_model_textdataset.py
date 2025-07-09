import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load dataset safely
with open('fraud_call.file', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Step 2: Clean and split lines
data = []
for line in lines:
    parts = line.strip().split(maxsplit=1)  # Split only once
    if len(parts) == 2:
        target, call_content = parts[0], parts[1]
        data.append((call_content, target))


# Step 3: Create DataFrame
df = pd.DataFrame(data, columns=['call_content', 'target'])
print(df.head())

# Step 4: Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Step 5: Apply cleaning
df['cleaned_text'] = df['call_content'].apply(clean_text)

# 🛠️ Step 6: Encode labels properly (fixed!)
df['target'] = df['target'].map({'fraud': 1, 'normal': 0})


# Step 7: Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'], df['target'], test_size=0.2, random_state=42)

# Step 8: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 9: Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Step 10: Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 11: Save model and vectorizer
with open('spam_detector_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully.")
