"""
train_model.py - Train and save Spam SMS Detection model
CodSoft ML Internship - Task 4
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# ─── Text Preprocessing ────────────────────────────────────────────────────────

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess SMS text."""
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    words = text.split()
    # Remove stopwords and apply stemming
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)


# ─── Load Dataset ──────────────────────────────────────────────────────────────

print("=" * 60)
print("  CodSoft ML Internship - Spam SMS Detection (Task 4)")
print("=" * 60)

# Adjust this path to where your CSV actually lives
CSV_PATH = os.path.join(os.path.dirname(__file__), 'spam_2.csv')
if not os.path.exists(CSV_PATH):
    # Try uploads folder (when running locally after copying)
    CSV_PATH = '/mnt/user-data/uploads/spam_2.csv'

df = pd.read_csv(CSV_PATH, encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df.dropna(inplace=True)

print(f"\n[Dataset] Total samples: {len(df)}")
print(df['label'].value_counts())

# ─── Preprocessing ─────────────────────────────────────────────────────────────

print("\n[Step 1] Preprocessing text...")
df['clean_text'] = df['message'].apply(preprocess_text)

# Encode labels: ham=0, spam=1
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})

X = df['clean_text']
y = df['label_enc']

# ─── Train / Test Split ────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"[Step 2] Train size: {len(X_train)}, Test size: {len(X_test)}")

# ─── TF-IDF Vectorization ──────────────────────────────────────────────────────

print("[Step 3] TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ─── Train Multiple Models ─────────────────────────────────────────────────────

models = {
    'Naive Bayes':        MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (LinearSVC)':    LinearSVC(random_state=42),
}

results = {}
print("\n[Step 4] Training & Evaluating Models:\n")
print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("-" * 65)

best_model_name = None
best_accuracy   = 0
best_clf        = None

for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    y_pred   = clf.predict(X_test_tfidf)
    acc      = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)
    spam_r   = report.get('1', {})
    prec     = spam_r.get('precision', 0)
    rec      = spam_r.get('recall', 0)
    f1       = spam_r.get('f1-score', 0)
    results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'model': clf}
    print(f"{name:<25} {acc*100:>9.2f}% {prec*100:>9.2f}% {rec*100:>7.2f}% {f1*100:>7.2f}%")

    if acc > best_accuracy:
        best_accuracy   = acc
        best_model_name = name
        best_clf        = clf

print("-" * 65)
print(f"\n[Best Model] {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)\n")

# Full report for best model
y_pred_best = best_clf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_best, target_names=['Ham', 'Spam']))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:")
print(f"  TP(Ham)={cm[0][0]}  FP={cm[0][1]}")
print(f"  FN={cm[1][0]}      TP(Spam)={cm[1][1]}")

# ─── Save Model & Vectorizer ───────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, 'model.pkl'),  'wb') as f:
    pickle.dump(best_clf, f)
with open(os.path.join(MODEL_DIR, 'tfidf.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)

# Save model metadata
import json
meta = {
    'model_name':  best_model_name,
    'accuracy':    round(best_accuracy * 100, 2),
    'total_train': len(X_train),
    'total_test':  len(X_test),
    'features':    5000,
    'all_results': {k: {m: round(v*100,2) for m,v in vd.items() if m != 'model'}
                    for k, vd in results.items()}
}
with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n[Saved] model/model.pkl, model/tfidf.pkl, model/meta.json")
print("\nRun app.py to start the web app.")
