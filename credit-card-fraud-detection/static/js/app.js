/* ─────────────────────────────────────────────────────────────
   FraudShield – Frontend JS
   CodSoft ML Internship | Task 2
───────────────────────────────────────────────────────────── */

// ── Tab Switching ─────────────────────────────────────────────

function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  btn.classList.add('active');
}

// ── Example Fillers ───────────────────────────────────────────

const examples = {
  fraud: {
    amt:        '1289.50',
    category:   'shopping_net',
    trans_time: '2020-06-15T02:30',
    dob:        '1985-03-22',
    gender:     'M',
    city_pop:   '149',
    lat:        '36.0788',
    long:       '-81.1781',
    merch_lat:  '37.1234',
    merch_long: '-120.5678',
  },
  legit: {
    amt:        '24.99',
    category:   'grocery_pos',
    trans_time: '2020-06-15T14:00',
    dob:        '1990-08-10',
    gender:     'F',
    city_pop:   '45000',
    lat:        '36.0788',
    long:       '-81.1781',
    merch_lat:  '36.1000',
    merch_long: '-81.2000',
  }
};

function fillExample(type) {
  const e = examples[type];
  document.getElementById('amt').value        = e.amt;
  document.getElementById('category').value   = e.category;
  document.getElementById('trans_time').value = e.trans_time;
  document.getElementById('dob').value        = e.dob;
  document.getElementById('gender').value     = e.gender;
  document.getElementById('city_pop').value   = e.city_pop;
  document.getElementById('lat').value        = e.lat;
  document.getElementById('long').value       = e.long;
  document.getElementById('merch_lat').value  = e.merch_lat;
  document.getElementById('merch_long').value = e.merch_long;
  clearResult();
}

function clearForm() {
  ['amt','city_pop','lat','long','merch_lat','merch_long'].forEach(id => {
    document.getElementById(id).value = '';
  });
  clearResult();
}

function clearResult() {
  const box = document.getElementById('resultBox');
  box.className = 'result-box hidden';
  box.innerHTML = '';
}

// ── Analyse Transaction ───────────────────────────────────────

async function analyseTransaction() {
  const amt = document.getElementById('amt').value;
  if (!amt) { alert('Please enter a transaction amount.'); return; }

  const payload = {
    amt:        document.getElementById('amt').value,
    category:   document.getElementById('category').value,
    trans_time: document.getElementById('trans_time').value,
    dob:        document.getElementById('dob').value,
    gender:     document.getElementById('gender').value,
    city_pop:   document.getElementById('city_pop').value || '50000',
    lat:        document.getElementById('lat').value || '36.0',
    long:       document.getElementById('long').value || '-81.0',
    merch_lat:  document.getElementById('merch_lat').value || '36.0',
    merch_long: document.getElementById('merch_long').value || '-81.0',
  };

  const btn = document.querySelector('.btn-primary');
  btn.disabled = true;
  btn.textContent = '⏳ Analysing...';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    renderResult(data);
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Analyse Transaction';
  }
}

// ── Render Result ─────────────────────────────────────────────

function renderResult(data) {
  const box      = document.getElementById('resultBox');
  const isF      = data.is_fraud;
  const risk     = data.risk_level;

  let cls, icon, labelText, labelCls;

  if (isF) {
    cls      = 'fraud-result';
    icon     = '🚨';
    labelText = 'FRAUDULENT TRANSACTION';
    labelCls  = 'fraud';
  } else if (risk === 'MEDIUM') {
    cls      = 'warn-result';
    icon     = '⚠️';
    labelText = 'SUSPICIOUS — VERIFY';
    labelCls  = 'warn';
  } else {
    cls      = 'legit-result';
    icon     = '✅';
    labelText = 'LEGITIMATE TRANSACTION';
    labelCls  = 'legit';
  }

  box.className = `result-box ${cls}`;
  box.innerHTML = `
    <div class="result-header">
      <div class="result-label ${labelCls}">${icon} ${labelText}</div>
      <span class="risk-badge risk-${risk}">${risk} RISK</span>
    </div>

    <div class="prob-bars">
      <div class="prob-row">
        <span class="prob-label">Fraud Prob.</span>
        <div class="prob-bar-wrap">
          <div class="prob-fill fraud" style="width: ${data.fraud_prob}%"></div>
        </div>
        <span class="prob-pct" style="color:#ef4444">${data.fraud_prob}%</span>
      </div>
      <div class="prob-row">
        <span class="prob-label">Legit Prob.</span>
        <div class="prob-bar-wrap">
          <div class="prob-fill legit" style="width: ${data.legit_prob}%"></div>
        </div>
        <span class="prob-pct" style="color:#22c55e">${data.legit_prob}%</span>
      </div>
    </div>

    <div class="result-meta">
      <span>Amount: <strong>$${parseFloat(data.amount).toFixed(2)}</strong></span>
      <span>Category: <strong>${data.category.replace('_', ' ')}</strong></span>
      <span>Risk Level: <strong>${risk}</strong></span>
    </div>
  `;
}
