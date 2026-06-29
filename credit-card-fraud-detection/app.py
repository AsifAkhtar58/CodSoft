"""
app.py - Flask Web App for Credit Card Fraud Detection
CodSoft ML Internship - Task 2
"""

from flask import Flask, render_template, request, jsonify
import pickle, json, os, numpy as np, pandas as pd
from datetime import datetime

app = Flask(__name__)

# ─── Load Model ────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

with open(os.path.join(MODEL_DIR, 'model.pkl'),  'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'meta.json'), 'r') as f:
    meta = json.load(f)

CATEGORIES = ['entertainment', 'food_dining', 'gas_transport', 'grocery_net',
              'grocery_pos', 'health_fitness', 'home', 'kids_pets',
              'misc_net', 'misc_pos', 'personal_care', 'shopping_net',
              'shopping_pos', 'travel']

CAT_MAP = {c: i for i, c in enumerate(sorted(CATEGORIES))}

# ─── Prediction Helper ─────────────────────────────────────────

def predict_transaction(data: dict) -> dict:
    try:
        trans_time = datetime.strptime(data['trans_time'], '%Y-%m-%dT%H:%M')
    except Exception:
        trans_time = datetime.now()

    try:
        dob = datetime.strptime(data['dob'], '%Y-%m-%d')
    except Exception:
        dob = datetime(1990, 1, 1)

    age      = (datetime(2020, 1, 1) - dob).days // 365
    distance = ((float(data['lat']) - float(data['merch_lat']))**2 +
                (float(data['long']) - float(data['merch_long']))**2) ** 0.5

    FEATURES = ['amt', 'hour', 'day', 'month', 'dayofweek',
                'age', 'distance', 'city_pop', 'gender_enc', 'category_enc']

    features = pd.DataFrame([[
        float(data['amt']),
        trans_time.hour,
        trans_time.day,
        trans_time.month,
        trans_time.weekday(),
        age,
        distance,
        float(data.get('city_pop', 50000)),
        1 if data.get('gender') == 'M' else 0,
        CAT_MAP.get(data.get('category', 'misc_pos'), 9),
    ]], columns=FEATURES)

    proba   = model.predict_proba(features)[0]
    pred    = model.predict(features)[0]
    fraud_p = float(proba[1])

    risk_level = 'LOW' if fraud_p < 0.3 else 'MEDIUM' if fraud_p < 0.7 else 'HIGH'

    return {
        'is_fraud':    bool(pred == 1),
        'fraud_prob':  round(fraud_p * 100, 2),
        'legit_prob':  round(float(proba[0]) * 100, 2),
        'risk_level':  risk_level,
        'amount':      float(data['amt']),
        'category':    data.get('category', 'unknown'),
    }

# ─── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', meta=meta, categories=CATEGORIES)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    try:
        result = predict_transaction(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    return jsonify(meta)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
