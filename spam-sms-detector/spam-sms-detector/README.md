# 📩 Spam SMS Detection — CodSoft ML Internship (Task 4)

A machine learning web application that classifies SMS messages as **Spam** or **Ham (Legitimate)** using NLP and multiple ML classifiers.

---

## 🚀 Demo Features

- **Single message** detection with confidence score
- **Batch detection** (up to 20 messages at once)
- Model comparison dashboard (Naive Bayes, Logistic Regression, SVM)
- Dark-themed responsive web UI (Flask + Vanilla JS)

---

## 🗂 Project Structure

```
spam-sms-detector/
├── train_model.py        # Train & save model
├── app.py                # Flask web server
├── requirements.txt
├── spam_2.csv            # Dataset (place here)
├── model/
│   ├── model.pkl         # Saved best classifier
│   ├── tfidf.pkl         # Saved TF-IDF vectorizer
│   └── meta.json         # Model metadata & accuracy
├── templates/
│   └── index.html        # Web UI
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## ⚙️ Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/CODSOFT.git
cd CODSOFT/spam-sms-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place the dataset
Put `spam_2.csv` in the project root (same folder as `train_model.py`).

### 4. Train the model
```bash
python train_model.py
```
This will:
- Preprocess the SMS text (lowercase, stopword removal, stemming)
- Extract TF-IDF features (5000 features, unigram + bigram)
- Train and compare Naive Bayes, Logistic Regression, and SVM
- Save the best model to `model/`

### 5. Run the web app
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🤖 ML Pipeline

| Step | Detail |
|------|--------|
| Text Cleaning | Lowercase, remove punctuation & numbers |
| Stopword Removal | NLTK English stopwords |
| Stemming | Porter Stemmer |
| Vectorization | TF-IDF (max_features=5000, ngram_range=(1,2)) |
| Models | Naive Bayes · Logistic Regression · SVM (LinearSVC) |
| Best Model | Selected automatically by test accuracy |

---

## 📊 Dataset

- **Source:** UCI SMS Spam Collection  
- **Records:** 5,572 messages (4,825 ham + 747 spam)
- **Columns used:** `v1` (label) and `v2` (message text)

---

## 🏆 Results

| Model | Accuracy |
|-------|----------|
| Naive Bayes | ~97% |
| Logistic Regression | ~98% |
| SVM (LinearSVC) | ~98% |

*(Exact numbers depend on random seed / split)*

---

## 📌 CodSoft Internship

- **Task:** 4 — Spam SMS Detection
- **Techniques:** TF-IDF, Naive Bayes, Logistic Regression, SVM
- **Language Composition:**
  - CSS: 33%
  - Python: 29%
  - HTML: 19.5%
  - JavaScript: 18.5%
- Hashtags: `#codsoft` `#internship` `#machinelearning`

---

## 👤 Author

**Asif** — B.Tech CSE, NIET Greater Noida  
GitHub: [github.com/YOUR_USERNAME/CODSOFT](https://github.com/YOUR_USERNAME/CODSOFT)
