"""
train_model.py - Credit Card Fraud Detection
CodSoft ML Internship - Task 2
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, f1_score)

print("=" * 60)
print("  CodSoft ML Internship - Credit Card Fraud Detection (Task 2)")
print("=" * 60)

# ─── Load Dataset ──────────────────────────────────────────────

TRAIN_PATH = os.path.join(os.path.dirname(__file__), 'fraudTrain.csv')
TEST_PATH  = os.path.join(os.path.dirname(__file__), 'fraudTest.csv')

if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = '/mnt/user-data/uploads/fraudTrain.csv'
    TEST_PATH  = '/mnt/user-data/uploads/fraudTest.csv'

print("\n[Step 1] Loading dataset...")
# Use sample for speed (large dataset: 1.3M rows)
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

print(f"  Train size: {len(train_df):,} transactions")
print(f"  Test size:  {len(test_df):,} transactions")
print(f"  Train fraud rate: {train_df['is_fraud'].mean()*100:.2f}%")
print(f"  Test fraud rate:  {test_df['is_fraud'].mean()*100:.2f}%")

# Sample for training speed (keep all fraud + sample legitimate)
fraud_train    = train_df[train_df['is_fraud'] == 1]
legit_train    = train_df[train_df['is_fraud'] == 0].sample(n=50000, random_state=42)
train_df       = pd.concat([fraud_train, legit_train]).sample(frac=1, random_state=42)

fraud_test     = test_df[test_df['is_fraud'] == 1]
legit_test     = test_df[test_df['is_fraud'] == 0].sample(n=20000, random_state=42)
test_df        = pd.concat([fraud_test, legit_test]).sample(frac=1, random_state=42)

print(f"\n  Sampled Train: {len(train_df):,} | Sampled Test: {len(test_df):,}")

# ─── Feature Engineering ───────────────────────────────────────

print("\n[Step 2] Feature Engineering...")

def engineer_features(df):
    df = df.copy()
    # Extract time features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour']      = df['trans_date_trans_time'].dt.hour
    df['day']       = df['trans_date_trans_time'].dt.day
    df['month']     = df['trans_date_trans_time'].dt.month
    df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek

    # Age from DOB
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (pd.Timestamp('2020-01-01') - df['dob']).dt.days // 365

    # Distance between customer and merchant
    df['distance'] = np.sqrt(
        (df['lat'] - df['merch_lat'])**2 +
        (df['long'] - df['merch_long'])**2
    )

    # Encode gender
    df['gender_enc'] = (df['gender'] == 'M').astype(int)

    # Encode category
    le_cat = LabelEncoder()
    df['category_enc'] = le_cat.fit_transform(df['category'].astype(str))

    return df

train_df = engineer_features(train_df)
test_df  = engineer_features(test_df)

# ─── Select Features ───────────────────────────────────────────

FEATURES = ['amt', 'hour', 'day', 'month', 'dayofweek',
            'age', 'distance', 'city_pop', 'gender_enc', 'category_enc']

X_train = train_df[FEATURES]
y_train = train_df['is_fraud']
X_test  = test_df[FEATURES]
y_test  = test_df['is_fraud']

# Scale features
scaler  = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"  Features used: {FEATURES}")

# ─── Train Models ──────────────────────────────────────────────

print("\n[Step 3] Training & Evaluating Models:\n")
print(f"{'Model':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
print("-" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                                   class_weight='balanced', n_jobs=-1),
}

results     = {}
best_name   = None
best_auc    = 0
best_model  = None

for name, clf in models.items():
    if name == 'Logistic Regression':
        clf.fit(X_train_sc, y_train)
        y_pred = clf.predict(X_test_sc)
        y_prob = clf.predict_proba(X_test_sc)[:, 1]
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = {'accuracy': round(acc*100, 2),
                     'f1': round(f1*100, 2),
                     'roc_auc': round(auc*100, 2),
                     'model': clf}

    print(f"{name:<25} {acc*100:>9.2f}% {f1*100:>9.2f}% {auc*100:>9.2f}%")

    if auc > best_auc:
        best_auc   = auc
        best_name  = name
        best_model = clf

print("-" * 60)
print(f"\n[Best Model] {best_name} (ROC-AUC: {best_auc*100:.2f}%)\n")

# Full report
if best_name == 'Logistic Regression':
    y_pred_best = best_model.predict(X_test_sc)
else:
    y_pred_best = best_model.predict(X_test)

print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Fraud']))

cm = confusion_matrix(y_test, y_pred_best)
print("Confusion Matrix:")
print(f"  True Legitimate:  {cm[0][0]}  |  False Fraud: {cm[0][1]}")
print(f"  Missed Fraud:     {cm[1][0]}  |  True Fraud:  {cm[1][1]}")

# ─── Save Model ────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, 'model.pkl'),  'wb') as f:
    pickle.dump(best_model, f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

meta = {
    'model_name':    best_name,
    'roc_auc':       round(best_auc * 100, 2),
    'features':      FEATURES,
    'train_samples': len(train_df),
    'test_samples':  len(test_df),
    'fraud_train':   int(y_train.sum()),
    'fraud_test':    int(y_test.sum()),
    'all_results':   {k: {m: v for m, v in vd.items() if m != 'model'}
                      for k, vd in results.items()}
}

with open(os.path.join(MODEL_DIR, 'meta.json'), 'w') as f:
    json.dump(meta, f, indent=2)

print(f"\n[Saved] model/model.pkl, model/scaler.pkl, model/meta.json")
print("\nRun: python3 app.py")
