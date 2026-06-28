/* ─────────────────────────────────────────────────────────────
   SpamShield – Frontend JS
   CodSoft ML Internship | Task 4
───────────────────────────────────────────────────────────── */

// ── Tab Switching ─────────────────────────────────────────────

function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + tabId).classList.add('active');
  btn.classList.add('active');
}

// ── Character Counter ─────────────────────────────────────────

const smsInput = document.getElementById('smsInput');
smsInput.addEventListener('input', () => {
  const len = smsInput.value.length;
  document.getElementById('charCount').textContent = len;
  if (len > 450) document.getElementById('charCount').style.color = '#ef4444';
  else           document.getElementById('charCount').style.color = '';
});

// ── Example Filler ────────────────────────────────────────────

const examples = {
  spam1: "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
  ham1:  "Hey, are you coming for lunch today? I was thinking we could go to that new place near the library."
};

function fillExample(key) {
  smsInput.value = examples[key];
  smsInput.dispatchEvent(new Event('input'));
  clearResult();
}

function clearInput() {
  smsInput.value = '';
  smsInput.dispatchEvent(new Event('input'));
  clearResult();
}

function clearResult() {
  const box = document.getElementById('resultBox');
  box.classList.add('hidden');
  box.className = 'result-box hidden';
  box.innerHTML = '';
}

// ── Single Predict ────────────────────────────────────────────

async function predictSingle() {
  const message = smsInput.value.trim();
  if (!message) { alert('Please enter a message.'); return; }

  const btn = document.querySelector('#tab-single .btn-primary');
  btn.disabled = true;
  btn.innerHTML = '<span>⏳ Analysing...</span>';

  try {
    const res  = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message })
    });
    const data = await res.json();
    renderResult(data);
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>🔍 Analyse Message</span>';
  }
}

function renderResult(data) {
  const box  = document.getElementById('resultBox');
  const isSpam = data.is_spam;
  const icon   = isSpam ? '🚨' : '✅';
  const label  = isSpam ? 'SPAM'  : 'HAM (Legitimate)';
  const cls    = isSpam ? 'spam'  : 'ham';
  const conf   = data.confidence !== null ? data.confidence : '—';

  box.className = `result-box ${cls}-result`;
  box.innerHTML = `
    <div class="result-label ${cls}">${icon} ${label}</div>
    ${data.confidence !== null ? `
    <div class="confidence-bar ${cls}">
      <div class="confidence-fill" style="width: ${data.confidence}%"></div>
    </div>` : ''}
    <div class="result-meta">
      <span>Confidence: <strong>${conf}%</strong></span>
      <span>Words: <strong>${data.word_count}</strong></span>
      <span>Characters: <strong>${data.char_count}</strong></span>
    </div>
  `;
  box.classList.remove('hidden');
}

// ── Batch Predict ─────────────────────────────────────────────

async function runBatch() {
  const raw      = document.getElementById('batchInput').value.trim();
  const messages = raw.split('\n').map(m => m.trim()).filter(m => m.length > 0);
  if (messages.length === 0) { alert('Enter at least one message.'); return; }
  if (messages.length > 20)  { alert('Max 20 messages at a time.'); return; }

  const btn = document.querySelector('#tab-batch .btn-primary');
  btn.disabled = true;
  btn.innerHTML = '⏳ Analysing...';

  try {
    const res  = await fetch('/batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages })
    });
    const data = await res.json();
    renderBatch(data);
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = '🔍 Analyse All';
  }
}

function renderBatch(data) {
  const box = document.getElementById('batchResult');
  const hamCount = data.total - data.spam_count;
  box.innerHTML = `
    <div class="batch-summary">
      📊 <strong>${data.total}</strong> messages analysed &nbsp;|&nbsp;
      🚨 <strong>${data.spam_count}</strong> spam &nbsp;|&nbsp;
      ✅ <strong>${hamCount}</strong> legitimate
    </div>
    ${data.results.map((r, i) => `
      <div class="batch-item ${r.is_spam ? 'spam' : 'ham'}">
        <span class="batch-badge ${r.is_spam ? 'spam' : 'ham'}">${r.is_spam ? 'SPAM' : 'HAM'}</span>
        <span class="batch-msg">${escapeHtml(r.message)}</span>
      </div>
    `).join('')}
  `;
  box.classList.remove('hidden');
}

// ── Utility ───────────────────────────────────────────────────

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// Enter key shortcut
smsInput.addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') predictSingle();
});
