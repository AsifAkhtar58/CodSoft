"""
app.py - Flask Web Application for Spam SMS Detection
CodSoft ML Internship - Task 4
"""

from flask import Flask, render_template, request, jsonify
import pickle, json, os, re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# ─── Load Model & Vectorizer ───────────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

with open(os.path.join(MODEL_DIR, 'model.pkl'),  'rb') as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'tfidf.pkl'), 'rb') as f:
    tfidf = pickle.load(f)
with open(os.path.join(MODEL_DIR, 'meta.json'), 'r') as f:
    meta = json.load(f)

ps         = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ─── Helper Functions ──────────────────────────────────────────────────────────

def preprocess_text(text):
    text  = text.lower()
    text  = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_sms(message: str) -> dict:
    cleaned  = preprocess_text(message)
    vec      = tfidf.transform([cleaned])
    pred     = model.predict(vec)[0]
    label    = 'spam' if pred == 1 else 'ham'

    # Confidence via decision function (SVM) or predict_proba
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 2)
    elif hasattr(model, 'decision_function'):
        score      = model.decision_function(vec)[0]
        # Sigmoid-like normalization
        import math
        confidence = round(100 / (1 + math.exp(-abs(score))), 2)

    return {
        'label':      label,
        'is_spam':    bool(pred == 1),
        'confidence': confidence,
        'word_count': len(message.split()),
        'char_count': len(message)
    }

# ─── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', meta=meta)

@app.route('/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400
    result = predict_sms(message)
    return jsonify(result)

@app.route('/api/stats')
def stats():
    return jsonify(meta)

@app.route('/batch', methods=['POST'])
def batch_predict():
    data     = request.get_json()
    messages = data.get('messages', [])
    if not messages or not isinstance(messages, list):
        return jsonify({'error': 'No messages provided'}), 400
    results = [{'message': m, **predict_sms(m)} for m in messages[:20]]  # cap at 20
    return jsonify({'results': results, 'total': len(results),
                    'spam_count': sum(1 for r in results if r['is_spam'])})

# ─── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, port=5000)
