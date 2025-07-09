import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC
import joblib

# Load the dataset
df = pd.read_csv("spamdatasetgenerated.csv", encoding='latin-1')[['type','transcript','category']]
df.columns = ['type','transcript', 'category']

# Check for missing values
df.dropna(inplace=True)

# Preprocessing function (basic clean)
def clean_text(text):
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text

# Clean the messages
df['transcript'] = df['transcript'].apply(clean_text)

# Encode the labels
df['category'] = df['category'].map({'normal': 0, 'fraud': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['transcript'], df['category'], test_size=0.2, random_state=42)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=3)  # 1 to 3-grams, ignore very rare words
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Random Forest Classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the model and vectorizer
with open('spam_classifier1.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer1.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
#joblib.dump(model, "spam_classifier1.pkl")
#joblib.dump(vectorizer, "vectorizer1.pkl")

print("\n✅ Model training and saving done successfully!")
