"""
Customer Churn Prediction — Flask app for Render deployment.
Loads a pre-trained Keras ANN (ann.pkl) and serves a single-page,
animated, glassmorphic/3D UI with an analytics dashboard.

Run locally:  python app.py
Deploy on Render:  gunicorn app:app
"""

import os
import pickle
import numpy as np

# Keep TensorFlow's internal thread pools small so it fits comfortably in
# low-memory environments like Render's free tier.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from flask import Flask, request, jsonify, render_template_string

# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ann.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Feature order the model expects (10 inputs)
FEATURES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# If you have the ORIGINAL scaler used at training time, drop a file named
# `scaler.pkl` (a fitted sklearn StandardScaler) next to this app.py and it
# will be used automatically for exact-accuracy scaling.
scaler = None
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

# Warm the model up at import time (not on the first request). Keras/TF
# lazily builds its computation graph on the very first call, which can
# take long enough to trip a gunicorn worker timeout. Doing it once here,
# during startup, means real requests are always fast.
try:
    model.predict(np.zeros((1, 10), dtype=float), verbose=0)
except Exception as _warm_err:
    print(f"Warm-up prediction failed (non-fatal): {_warm_err}")

# Fallback: public summary statistics of the standard "Churn_Modelling.csv"
# dataset (10,000 rows) this architecture is trained on, used only if no
# scaler.pkl is supplied. Geography encoded France=0, Germany=1, Spain=2.
# Gender encoded Female=0, Male=1.
FALLBACK_MEAN = np.array([650.5288, 0.7462, 0.5457, 38.9218, 5.0128,
                           76485.8893, 1.5302, 0.7055, 0.5151, 100090.2399])
FALLBACK_STD = np.array([96.6533, 0.8281, 0.4979, 10.4878, 2.8921,
                          62397.4052, 0.5817, 0.4558, 0.4998, 57510.4928])


def scale(raw_vector: np.ndarray) -> np.ndarray:
    if scaler is not None:
        return scaler.transform(raw_vector)
    return (raw_vector - FALLBACK_MEAN) / FALLBACK_STD


def risk_level(prob: float) -> str:
    if prob >= 0.66:
        return "High"
    if prob >= 0.33:
        return "Medium"
    return "Low"


def top_factors(raw_vector: np.ndarray, prob: float):
    """Very lightweight explainability: rank features by how far they sit
    from the population mean (z-score), weighted by a plausible churn
    direction, so the UI can show 'why' behind a prediction."""
    z = (raw_vector[0] - FALLBACK_MEAN) / FALLBACK_STD
    # direction: +1 means "higher value -> more likely to push toward churn"
    direction = np.array([-1, 1, 0, 1, -1, 1, -1, -1, -1, 0])
    score = z * direction
    order = np.argsort(-score)
    labels = {
        "CreditScore": "Credit score",
        "Geography": "Geography",
        "Gender": "Gender",
        "Age": "Age",
        "Tenure": "Tenure (years with bank)",
        "Balance": "Account balance",
        "NumOfProducts": "Number of products",
        "HasCrCard": "Has credit card",
        "IsActiveMember": "Active membership",
        "EstimatedSalary": "Estimated salary",
    }
    out = []
    for idx in order[:4]:
        feat = FEATURES[idx]
        contributes_up = score[idx] > 0
        out.append({
            "feature": labels[feat],
            "z": round(float(z[idx]), 2),
            "impact": "increases risk" if contributes_up else "lowers risk",
        })
    return out


def encode_form(payload: dict) -> np.ndarray:
    geo_map = {"France": 0, "Germany": 1, "Spain": 2}
    gender_map = {"Female": 0, "Male": 1}
    row = [
        float(payload["CreditScore"]),
        geo_map[payload["Geography"]],
        gender_map[payload["Gender"]],
        float(payload["Age"]),
        float(payload["Tenure"]),
        float(payload["Balance"]),
        float(payload["NumOfProducts"]),
        1.0 if payload["HasCrCard"] == "Yes" else 0.0,
        1.0 if payload["IsActiveMember"] == "Yes" else 0.0,
        float(payload["EstimatedSalary"]),
    ]
    return np.array([row], dtype=float)


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(PAGE_TEMPLATE, scaler_note=("custom scaler.pkl" if scaler else "built-in dataset statistics"))


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        raw = encode_form(payload)
        scaled = scale(raw)
        prob = float(model.predict(scaled, verbose=0)[0][0])
        result = {
            "probability": round(prob * 100, 2),
            "prediction": "Churn" if prob >= 0.5 else "Stay",
            "risk": risk_level(prob),
            "factors": top_factors(raw, prob),
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# --------------------------------------------------------------------------
# Front-end (single template: HTML + CSS + JS)
# --------------------------------------------------------------------------
PAGE_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Churn Predictor · ANN</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.4/chart.umd.min.js"></script>
<style>
  :root{
    --bg-1:#0b0f1e; --bg-2:#141a34; --accent:#7c5cff; --accent-2:#00d4c7;
    --danger:#ff5673; --warn:#ffb84d; --good:#3ddc97;
    --card:rgba(255,255,255,0.06); --card-border:rgba(255,255,255,0.14);
    --text:#eef0ff; --muted:#9aa2c4;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  html,body{height:100%;}
  body{
    font-family:'Segoe UI', system-ui, -apple-system, sans-serif;
    color:var(--text); min-height:100vh; overflow-x:hidden;
    background:linear-gradient(125deg, var(--bg-1), var(--bg-2), var(--bg-1));
    background-size:400% 400%;
    animation:gradientShift 18s ease infinite;
    position:relative;
  }
  @keyframes gradientShift{
    0%{background-position:0% 50%;} 50%{background-position:100% 50%;} 100%{background-position:0% 50%;}
  }
  .orb{position:fixed; border-radius:50%; filter:blur(60px); opacity:.35; z-index:0; pointer-events:none;}
  .orb1{width:420px; height:420px; background:var(--accent); top:-120px; left:-120px; animation:float1 14s ease-in-out infinite;}
  .orb2{width:360px; height:360px; background:var(--accent-2); bottom:-100px; right:-100px; animation:float2 16s ease-in-out infinite;}
  @keyframes float1{0%,100%{transform:translate(0,0);} 50%{transform:translate(60px,80px);}}
  @keyframes float2{0%,100%{transform:translate(0,0);} 50%{transform:translate(-70px,-50px);}}

  .wrap{position:relative; z-index:1; max-width:1200px; margin:0 auto; padding:40px 20px 80px;}
  header{text-align:center; margin-bottom:36px; animation:fadeDown .8s ease;}
  header h1{
    font-size:2.4rem; font-weight:800; letter-spacing:.5px;
    background:linear-gradient(90deg, var(--accent), var(--accent-2));
    -webkit-background-clip:text; background-clip:text; color:transparent;
  }
  header p{color:var(--muted); margin-top:8px; font-size:1rem;}
  @keyframes fadeDown{from{opacity:0; transform:translateY(-16px);} to{opacity:1; transform:translateY(0);}}

  .grid{display:grid; grid-template-columns:1.1fr 0.9fr; gap:28px;}
  @media (max-width:960px){ .grid{grid-template-columns:1fr;} }

  .card{
    background:var(--card); border:1px solid var(--card-border);
    border-radius:20px; padding:28px; backdrop-filter:blur(16px);
    box-shadow:0 20px 45px rgba(0,0,0,.35);
    transform-style:preserve-3d; transition:transform .15s ease, box-shadow .3s ease;
    animation:fadeUp .7s ease both;
  }
  .card:hover{ box-shadow:0 30px 60px rgba(124,92,255,.25); }
  @keyframes fadeUp{from{opacity:0; transform:translateY(24px);} to{opacity:1; transform:translateY(0);}}

  .card h2{font-size:1.15rem; margin-bottom:18px; color:var(--text); display:flex; align-items:center; gap:8px;}
  .badge{font-size:.7rem; padding:3px 9px; border-radius:20px; background:rgba(124,92,255,.2); color:var(--accent-2); border:1px solid rgba(124,92,255,.4);}

  .field{margin-bottom:16px;}
  .field label{display:flex; justify-content:space-between; font-size:.8rem; color:var(--muted); margin-bottom:6px;}
  .field input[type=number], .field select{
    width:100%; padding:11px 13px; border-radius:10px; border:1px solid var(--card-border);
    background:rgba(255,255,255,.05); color:var(--text); font-size:.92rem; outline:none;
    transition:border-color .2s ease, background .2s ease;
  }
  .field input[type=number]:focus, .field select:focus{ border-color:var(--accent); background:rgba(124,92,255,.08); }
  .field input[type=range]{ width:100%; accent-color:var(--accent); }
  .row2{display:grid; grid-template-columns:1fr 1fr; gap:14px;}

  .btn{
    width:100%; padding:14px; border:none; border-radius:12px; cursor:pointer;
    font-weight:700; font-size:.95rem; letter-spacing:.3px; color:#0b0f1e;
    background:linear-gradient(90deg, var(--accent-2), var(--accent));
    position:relative; overflow:hidden; margin-top:6px;
    transition:transform .15s ease, box-shadow .2s ease;
  }
  .btn:hover{ transform:translateY(-2px); box-shadow:0 12px 24px rgba(124,92,255,.35); }
  .btn:active{ transform:translateY(0); }
  .btn .ripple{position:absolute; border-radius:50%; background:rgba(255,255,255,.5); transform:scale(0); animation:ripple .6s linear; pointer-events:none;}
  @keyframes ripple{ to{ transform:scale(4); opacity:0; } }

  .result{ display:flex; flex-direction:column; align-items:center; gap:10px; }
  .gauge-wrap{ position:relative; width:200px; height:200px; margin:6px auto 10px; }
  .gauge-wrap svg{ transform:rotate(-90deg); }
  .gauge-bg{ fill:none; stroke:rgba(255,255,255,.08); stroke-width:14; }
  .gauge-fg{ fill:none; stroke-width:14; stroke-linecap:round; transition:stroke-dashoffset 1s cubic-bezier(.65,0,.35,1), stroke .4s ease; }
  .gauge-center{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
  .gauge-center .pct{ font-size:2.1rem; font-weight:800; }
  .gauge-center .lbl{ font-size:.75rem; color:var(--muted); margin-top:2px; }

  .risk-pill{ padding:6px 18px; border-radius:20px; font-weight:700; font-size:.85rem; letter-spacing:.4px; }
  .risk-Low{ background:rgba(61,220,151,.15); color:var(--good); border:1px solid rgba(61,220,151,.4); }
  .risk-Medium{ background:rgba(255,184,77,.15); color:var(--warn); border:1px solid rgba(255,184,77,.4); }
  .risk-High{ background:rgba(255,86,115,.15); color:var(--danger); border:1px solid rgba(255,86,115,.4); animation:pulse 1.4s ease-in-out infinite; }
  @keyframes pulse{ 0%,100%{ box-shadow:0 0 0 0 rgba(255,86,115,.4);} 50%{ box-shadow:0 0 0 8px rgba(255,86,115,0);} }

  .factors{ width:100%; margin-top:14px; display:flex; flex-direction:column; gap:8px; }
  .factor{ display:flex; justify-content:space-between; align-items:center; background:rgba(255,255,255,.04); padding:9px 12px; border-radius:10px; font-size:.82rem; border:1px solid rgba(255,255,255,.06); }
  .factor .up{ color:var(--danger); } .factor .down{ color:var(--good); }

  .stats{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-top:22px; }
  .stat{ background:rgba(255,255,255,.04); border:1px solid var(--card-border); border-radius:14px; padding:16px; text-align:center; }
  .stat .num{ font-size:1.6rem; font-weight:800; color:var(--accent-2); }
  .stat .lbl{ font-size:.72rem; color:var(--muted); margin-top:4px; }

  canvas{ max-height:230px; }
  .chart-row{ display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:18px; }
  @media (max-width:960px){ .chart-row{grid-template-columns:1fr;} }

  .empty{ text-align:center; color:var(--muted); padding:30px 10px; font-size:.9rem; }
  .note{ font-size:.72rem; color:var(--muted); margin-top:18px; text-align:center; line-height:1.5; }
  footer{ text-align:center; color:var(--muted); font-size:.75rem; margin-top:50px; }
</style>
</head>
<body>
<div class="orb orb1"></div>
<div class="orb orb2"></div>

<div class="wrap">
  <header>
    <h1>Customer Churn Predictor</h1>
    <p>Artificial Neural Network · real-time inference · analytics dashboard</p>
  </header>

  <div class="grid">
    <!-- FORM -->
    <div class="card" id="formCard">
      <h2>Customer profile <span class="badge">10 features</span></h2>
      <div class="row2">
        <div class="field">
          <label>Credit Score <span id="csv">650</span></label>
          <input type="range" id="CreditScore" min="300" max="900" value="650">
        </div>
        <div class="field">
          <label>Age <span id="agev">35</span></label>
          <input type="range" id="Age" min="18" max="92" value="35">
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Geography</label>
          <select id="Geography">
            <option>France</option><option>Germany</option><option>Spain</option>
          </select>
        </div>
        <div class="field">
          <label>Gender</label>
          <select id="Gender"><option>Female</option><option>Male</option></select>
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Tenure (years) <span id="tenv">5</span></label>
          <input type="range" id="Tenure" min="0" max="10" value="5">
        </div>
        <div class="field">
          <label>Number of Products <span id="nopv">1</span></label>
          <input type="range" id="NumOfProducts" min="1" max="4" value="1">
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Balance ($)</label>
          <input type="number" id="Balance" value="76485" step="100">
        </div>
        <div class="field">
          <label>Estimated Salary ($)</label>
          <input type="number" id="EstimatedSalary" value="100000" step="100">
        </div>
      </div>
      <div class="row2">
        <div class="field">
          <label>Has Credit Card</label>
          <select id="HasCrCard"><option>Yes</option><option>No</option></select>
        </div>
        <div class="field">
          <label>Active Member</label>
          <select id="IsActiveMember"><option>Yes</option><option>No</option></select>
        </div>
      </div>
      <button class="btn" id="predictBtn">Predict Churn Risk</button>
      <p class="note">Scaling source: <strong>{{ scaler_note }}</strong>. Drop your own <code>scaler.pkl</code> next to app.py for exact training-time accuracy.</p>
    </div>

    <!-- RESULT -->
    <div class="card" id="resultCard">
      <h2>Prediction</h2>
      <div class="result">
        <div class="gauge-wrap">
          <svg width="200" height="200" viewBox="0 0 200 200">
            <circle class="gauge-bg" cx="100" cy="100" r="86"></circle>
            <circle class="gauge-fg" id="gaugeFg" cx="100" cy="100" r="86"
              stroke-dasharray="540.35" stroke-dashoffset="540.35" stroke="#7c5cff"></circle>
          </svg>
          <div class="gauge-center">
            <div class="pct" id="pctText">--%</div>
            <div class="lbl">churn probability</div>
          </div>
        </div>
        <div id="riskPill" class="risk-pill" style="display:none;">--</div>
        <div class="factors" id="factors"></div>
      </div>
    </div>
  </div>

  <!-- ANALYTICS -->
  <div class="card" style="margin-top:28px;" id="analyticsCard">
    <h2>Session Analytics <span class="badge">local history</span></h2>
    <div class="stats">
      <div class="stat"><div class="num" id="statTotal">0</div><div class="lbl">Predictions made</div></div>
      <div class="stat"><div class="num" id="statAvg">0%</div><div class="lbl">Avg. churn probability</div></div>
      <div class="stat"><div class="num" id="statHigh">0</div><div class="lbl">High-risk customers</div></div>
    </div>
    <div class="chart-row">
      <div><canvas id="trendChart"></canvas></div>
      <div><canvas id="riskChart"></canvas></div>
    </div>
    <p class="note" id="emptyNote">Run a prediction to start building your session analytics.</p>
  </div>

  <footer>Built with a Keras ANN · Flask · Chart.js</footer>
</div>

<script>
// ---------- live slider labels ----------
const bind = (id,out)=>{ const el=document.getElementById(id); el.addEventListener('input',()=>{document.getElementById(out).textContent=el.value;});};
bind('CreditScore','csv'); bind('Age','agev'); bind('Tenure','tenv'); bind('NumOfProducts','nopv');

// ---------- 3D tilt effect ----------
document.querySelectorAll('.card').forEach(card=>{
  card.addEventListener('mousemove', e=>{
    const r = card.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    const rx = ((y / r.height) - 0.5) * -6;
    const ry = ((x / r.width) - 0.5) * 6;
    card.style.transform = `perspective(900px) rotateX(${rx}deg) rotateY(${ry}deg) translateZ(0)`;
  });
  card.addEventListener('mouseleave', ()=>{ card.style.transform = 'perspective(900px) rotateX(0) rotateY(0)'; });
});

// ---------- ripple ----------
document.getElementById('predictBtn').addEventListener('click', function(e){
  const btn = e.currentTarget;
  const circle = document.createElement('span');
  circle.className = 'ripple';
  const rect = btn.getBoundingClientRect();
  circle.style.left = (e.clientX - rect.left) + 'px';
  circle.style.top = (e.clientY - rect.top) + 'px';
  circle.style.width = circle.style.height = '20px';
  btn.appendChild(circle);
  setTimeout(()=>circle.remove(), 650);
  runPrediction();
});

// ---------- charts ----------
const trendCtx = document.getElementById('trendChart').getContext('2d');
const riskCtx = document.getElementById('riskChart').getContext('2d');
let trendChart = new Chart(trendCtx, {
  type:'line',
  data:{ labels:[], datasets:[{ label:'Churn probability %', data:[], borderColor:'#7c5cff', backgroundColor:'rgba(124,92,255,.15)', tension:.35, fill:true, pointRadius:3 }]},
  options:{ responsive:true, plugins:{legend:{labels:{color:'#9aa2c4'}}}, scales:{ x:{ticks:{color:'#9aa2c4'}, grid:{color:'rgba(255,255,255,.06)'}}, y:{min:0,max:100,ticks:{color:'#9aa2c4'}, grid:{color:'rgba(255,255,255,.06)'}} } }
});
let riskChart = new Chart(riskCtx, {
  type:'doughnut',
  data:{ labels:['Low','Medium','High'], datasets:[{ data:[0,0,0], backgroundColor:['#3ddc97','#ffb84d','#ff5673'], borderWidth:0 }]},
  options:{ responsive:true, plugins:{legend:{position:'bottom', labels:{color:'#9aa2c4'}}} }
});

// ---------- local history ----------
function loadHistory(){ return JSON.parse(localStorage.getItem('churn_history')||'[]'); }
function saveHistory(h){ localStorage.setItem('churn_history', JSON.stringify(h)); }

function refreshAnalytics(){
  const hist = loadHistory();
  document.getElementById('emptyNote').style.display = hist.length ? 'none' : 'block';
  document.getElementById('statTotal').textContent = hist.length;
  const avg = hist.length ? (hist.reduce((a,b)=>a+b.probability,0)/hist.length) : 0;
  document.getElementById('statAvg').textContent = avg.toFixed(1)+'%';
  const high = hist.filter(h=>h.risk==='High').length;
  document.getElementById('statHigh').textContent = high;

  const last = hist.slice(-12);
  trendChart.data.labels = last.map((_,i)=>'#'+(hist.length-last.length+i+1));
  trendChart.data.datasets[0].data = last.map(h=>h.probability);
  trendChart.update();

  const counts = {Low:0, Medium:0, High:0};
  hist.forEach(h=>counts[h.risk]++);
  riskChart.data.datasets[0].data = [counts.Low, counts.Medium, counts.High];
  riskChart.update();
}

// ---------- gauge + prediction ----------
function setGauge(pct, risk){
  const circumference = 2*Math.PI*86;
  const offset = circumference - (pct/100)*circumference;
  const fg = document.getElementById('gaugeFg');
  fg.style.strokeDasharray = circumference;
  fg.style.strokeDashoffset = offset;
  const color = risk==='High' ? '#ff5673' : risk==='Medium' ? '#ffb84d' : '#3ddc97';
  fg.style.stroke = color;
}

function animateCount(el, target){
  let start = 0; const dur = 900; const t0 = performance.now();
  function step(t){
    const p = Math.min((t-t0)/dur, 1);
    const val = (start + (target-start)*p).toFixed(1);
    el.textContent = val + '%';
    if(p<1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

async function runPrediction(){
  const payload = {
    CreditScore: document.getElementById('CreditScore').value,
    Geography: document.getElementById('Geography').value,
    Gender: document.getElementById('Gender').value,
    Age: document.getElementById('Age').value,
    Tenure: document.getElementById('Tenure').value,
    Balance: document.getElementById('Balance').value,
    NumOfProducts: document.getElementById('NumOfProducts').value,
    HasCrCard: document.getElementById('HasCrCard').value,
    IsActiveMember: document.getElementById('IsActiveMember').value,
    EstimatedSalary: document.getElementById('EstimatedSalary').value,
  };
  const btn = document.getElementById('predictBtn');
  btn.textContent = 'Predicting…'; btn.disabled = true;
  try{
    const res = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const data = await res.json();
    if(data.error){ alert('Error: '+data.error); return; }

    setGauge(data.probability, data.risk);
    animateCount(document.getElementById('pctText'), data.probability);

    const pill = document.getElementById('riskPill');
    pill.style.display = 'inline-block';
    pill.textContent = data.risk + ' risk · ' + data.prediction;
    pill.className = 'risk-pill risk-' + data.risk;

    const factorsEl = document.getElementById('factors');
    factorsEl.innerHTML = '';
    data.factors.forEach(f=>{
      const div = document.createElement('div');
      div.className = 'factor';
      div.innerHTML = `<span>${f.feature}</span><span class="${f.impact.includes('increase')?'up':'down'}">${f.impact} (z=${f.z})</span>`;
      factorsEl.appendChild(div);
    });

    const hist = loadHistory();
    hist.push({ probability:data.probability, risk:data.risk, ts: Date.now() });
    saveHistory(hist);
    refreshAnalytics();
  } catch(err){
    alert('Request failed: ' + err.message);
  } finally {
    btn.textContent = 'Predict Churn Risk'; btn.disabled = false;
  }
}

refreshAnalytics();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
