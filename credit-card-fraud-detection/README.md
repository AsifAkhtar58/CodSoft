# 💳 Credit Card Fraud Detection — CodSoft ML Internship (Task 2)

A machine learning web application that detects fraudulent credit card transactions in real time using Random Forest, Decision Tree, and Logistic Regression classifiers.

---

## 🚀 Features

- Real-time fraud detection with risk level (LOW / MEDIUM / HIGH)
- Fraud probability score with visual bar
- Model comparison dashboard (3 algorithms)
- Dark-themed responsive web UI (Flask + Vanilla JS)

---

## 🗂 Project Structure

```
credit-card-fraud/
├── train_model.py        # Train & save model
├── app.py                # Flask web server
├── requirements.txt
├── fraudTrain.csv        # Training dataset (place here)
├── fraudTest.csv         # Test dataset (place here)
├── model/
│   ├── model.pkl         # Saved Random Forest model
│   ├── scaler.pkl        # Standard scaler
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
git clone https://github.com/AsifAkhtar58/CodSoft.git
cd CodSoft/credit-card-fraud
```

### 2. Install dependencies
```bash
pip3 install -r requirements.txt
```

### 3. Place datasets
Put `fraudTrain.csv` and `fraudTest.csv` in the project root.

### 4. Train the model
```bash
python3 train_model.py
```

### 5. Run the web app
```bash
python3 app.py
```

Open [http://localhost:5000](http://localhost:5000)

---

## 🤖 ML Pipeline

| Step | Detail |
|------|--------|
| Feature Engineering | Hour, Day, Age, Distance to merchant, Category |
| Class Imbalance | Handled via class_weight='balanced' |
| Models | Logistic Regression · Decision Tree · Random Forest |
| Best Model | Random Forest (ROC-AUC: 99.39%) |

---

## 📊 Dataset

- Source: Credit Card Fraud Dataset
- Training records: 1,296,675 transactions
- Fraud rate: ~0.58% (highly imbalanced)
- Features: 23 columns

---

## 🏆 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 94.45% | 85.08% |
| Decision Tree | 97.11% | 98.40% |
| Random Forest | 98.18% | 99.39% ⭐ |

---

## 📌 CodSoft Internship

- **Task:** 2 — Credit Card Fraud Detection
- **Algorithms:** Logistic Regression, Decision Trees, Random Forests
- Hashtags: `#codsoft` `#internship` `#machinelearning`

---

## 👤 Author

**Asif Akhtar** — B.Tech CSE, NIET Greater Noida
GitHub: [github.com/AsifAkhtar58/CodSoft](https://github.com/AsifAkhtar58/CodSoft)
