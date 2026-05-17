from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import sqlite3
from contextlib import contextmanager
import pandas as pd
import numpy as np
import joblib
import os
import json
import uuid
import hashlib
from datetime import datetime
import random
import time
import urllib.request
import urllib.parse
import warnings
warnings.filterwarnings('ignore')
from authlib.integrations.flask_client import OAuth
import google.generativeai as genai
import shap
import os
from flask import send_from_directory



def _format_chat_context_explanation(pred):
    import html as _html
    if not pred:
        return (
            "<b>No prediction context found.</b><br>"
            "Open a prediction page, submit the form, then ask me to explain your result."
        )

    disease = _html.escape(str(pred.get('disease') or pred.get('page') or 'Current prediction'))
    status = _html.escape(str(pred.get('status') or pred.get('risk') or pred.get('prediction') or 'Unknown'))
    score = pred.get('score') or pred.get('confidence') or pred.get('probability') or ''
    score_html = _html.escape(str(score)) if score != '' else 'Not available'
    probabilities = pred.get('probabilities') or {}
    reasons = pred.get('reasons') or pred.get('clinical_flags') or []

    prob_lines = []
    if isinstance(probabilities, dict):
        for name, value in list(probabilities.items())[:6]:
            prob_lines.append(f"• {_html.escape(str(name).replace('_', ' '))}: <b>{_html.escape(str(value))}%</b>")

    reason_lines = []
    for item in reasons[:8]:
        reason_lines.append(f"• {_html.escape(str(item))}")

    html = (
        f"<b>Detailed Result Explanation</b><br>"
        f"Disease/Page: <b>{disease}</b><br>"
        f"Result: <b>{status}</b><br>"
        f"Confidence / risk score: <b>{score_html}%</b><br><br>"
        "<b>What this means:</b><br>"
        "This is a screening-style result. It suggests the risk level from the values entered in the VitalsAI form; it is not a confirmed medical diagnosis.<br><br>"
    )

    if prob_lines:
        html += "<b>Probability breakdown:</b><br>" + "<br>".join(prob_lines) + "<br><br>"

    if reason_lines:
        html += "<b>Main factors behind the result:</b><br>" + "<br>".join(reason_lines) + "<br><br>"
    else:
        html += (
            "<b>Main factors to review:</b><br>"
            "• Compare your BP, glucose, cholesterol, BMI and symptom values with the normal range.<br>"
            "• If the result is high risk, repeat the test/checkup and consult a specialist.<br><br>"
        )

    html += (
        "<b>Next steps:</b><br>"
        "• Save/download the report from the prediction page.<br>"
        "• Recheck abnormal values with a qualified doctor or lab test.<br>"
        "• Seek urgent care if you have chest pain, severe breathlessness, fainting, stroke signs, very low urine output, or confusion.<br><br>"
        "<i>Please consult a doctor for personalized advice.</i>"
    )
    return html

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
print("Loaded KEY:", API_KEY)
# ── Load .env file ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env loaded")
except ImportError:
    print("[INFO] python-dotenv not installed")

app = Flask(__name__)
app.secret_key = 'vitalsai-secret-2024'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour

# reCAPTCHA — replace with real keys from https://www.google.com/recaptcha/admin
RECAPTCHA_SECRET = os.environ.get('RECAPTCHA_SECRET', '6LdoPNMsAAAAAGRGG6G41pbS5bKrrGzT6Cbwwwtf')
IDLE_TIMEOUT = 3600  # 1 hour idle auto-logout

GOOGLE_CLIENT_ID     = os.environ.get('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.environ.get('GOOGLE_CLIENT_SECRET', '')

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODELS  = {}
GEMINI_API_KEY  = os.environ.get('GEMINI_API_KEY')
OPENROUTER_KEY  = os.environ.get('OPENROUTER_API_KEY', '')

# ── In-memory OTP store (no persistence needed) ───────────
OTP_STORE = {}   # email -> {otp, expires, attempts}

# ══════════════════════════════════════════════════════════════
#  SQLite DATABASE SETUP
#  File: vitalsai.db  (auto-created in project folder)
# ══════════════════════════════════════════════════════════════
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vitalsai.db')

@contextmanager
def get_db():
    """Thread-safe SQLite connection context manager"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Better concurrency
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """Create all tables if they don't exist"""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                email       TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                pwd         TEXT,
                created_at  TEXT DEFAULT (datetime('now')),
                verified    INTEGER DEFAULT 0,
                google      INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id          TEXT PRIMARY KEY,
                user_email  TEXT NOT NULL,
                disease     TEXT NOT NULL,
                inputs      TEXT,
                result      TEXT,
                risk        TEXT,
                probability REAL,
                created_at  TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_email) REFERENCES users(email)
            );
        """)
    print("[DB] SQLite database initialised ✅")

init_db()

# ── DB helper functions (drop-in replacements for dict ops) ──

def db_get_user(email):
    """Return user dict or None"""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None

def db_email_exists(email):
    with get_db() as conn:
        return conn.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone() is not None

def db_create_user(email, name, pwd_hash, verified=True, google=False):
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO users (email,name,pwd,verified,google) VALUES (?,?,?,?,?)",
            (email, name, pwd_hash, int(verified), int(google))
        )

def db_save_prediction(user_email, disease, inputs, result):
    """Save prediction to DB (linked to logged-in user)"""
    rec_id   = str(uuid.uuid4())[:8]
    ts       = datetime.now().strftime('%Y-%m-%d %H:%M')
    risk     = result.get('risk') or result.get('prediction') or result.get('stage','')
    prob_raw = result.get('probability') or result.get('survival_probability') or 0
    try:
        prob = float(str(prob_raw).replace('%',''))
    except Exception:
        prob = 0.0
    with get_db() as conn:
        conn.execute(
            """INSERT INTO predictions (id,user_email,disease,inputs,result,risk,probability,created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (rec_id, user_email, disease,
             json.dumps(inputs,  ensure_ascii=False),
             json.dumps(result,  ensure_ascii=False),
             str(risk), prob, ts)
        )
    return rec_id

def db_get_history(user_email, limit=50):
    """Get prediction history for a user"""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id,disease,inputs,result,risk,probability,created_at
               FROM predictions WHERE user_email=?
               ORDER BY created_at DESC LIMIT ?""",
            (user_email, limit)
        ).fetchall()
    history = []
    for r in rows:
        try:   inp = json.loads(r['inputs'])
        except: inp = {}
        try:   res = json.loads(r['result'])
        except: res = {}
        history.append({
            'id':        r['id'],
            'timestamp': r['created_at'],
            'disease':   r['disease'],
            'inputs':    inp,
            'result':    res,
            'risk':      r['risk'],
            'probability': r['probability'],
        })
    return history

def db_delete_prediction(user_email, record_id):
    with get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE id=? AND user_email=?", (record_id, user_email))

def db_clear_history(user_email):
    with get_db() as conn:
        conn.execute("DELETE FROM predictions WHERE user_email=?", (user_email,))

def hash_pwd(p):
    return hashlib.sha256(p.encode()).hexdigest()

# ── Load All Models ────────────────────────────────────────
def load_models():

    # ── HEART ──────────────────────────────────────────────
    try:
        MODELS['heart'] = {
            'model': joblib.load('models/final_stacking_model.pkl'),
            'features': ["Age","BMI","HighBP","HighChol","Diabetes",
                         "Smoker","PhysActivity","GenHlth","Sex"],
            'threshold': 0.30
        }
        print("[OK] Heart model loaded")
    except Exception as e:
        print(f"[WARN] Heart: {e}")

    # ── BRAIN ──────────────────────────────────────────────
    try:
        MODELS['brain'] = {
            'model':    joblib.load(os.path.join(MODELS_DIR, 'brain_ml_model.pkl')),
            'features': joblib.load(os.path.join(MODELS_DIR, 'brain_features.pkl')),
            'selector': joblib.load(os.path.join(MODELS_DIR, 'brain_selector.pkl')),
            'scaler':   joblib.load(os.path.join(MODELS_DIR, 'brain_scaler.pkl')),
            'threshold': 0.15
        }
        print("[OK] Brain model loaded")
    except Exception as e:
        print(f"[WARN] Brain: {e}")

    # ── DIABETES ───────────────────────────────────────────
    try:
        MODELS['diabetes'] = {
            'model':   joblib.load(os.path.join(MODELS_DIR, 'diabetes_stack_model.pkl')),
            'scaler':  joblib.load(os.path.join(MODELS_DIR, 'diabetes_scaler.pkl')),
            'features': ["Pregnancies","Glucose","BloodPressure",
                         "SkinThickness","Insulin","BMI",
                         "DiabetesPedigreeFunction","Age"],
            'threshold': 0.40
        }
        print("[OK] Diabetes model loaded")
    except Exception as e:
        print(f"[WARN] Diabetes: {e}")

    # ── KIDNEY ─────────────────────────────────────────────
    try:
        MODELS['kidney'] = {
            'model':      joblib.load(os.path.join(MODELS_DIR, 'kidney_pipeline.pkl')),
            'features':   joblib.load(os.path.join(MODELS_DIR, 'kidney_features.pkl')),
            'target_map': joblib.load(os.path.join(MODELS_DIR, 'kidney_target_map.pkl')),
        }
        print("[OK] Kidney model loaded")
    except Exception as e:
        print(f"[WARN] Kidney: {e}")

    # ── EYE ────────────────────────────────────────────────
    try:
        import tensorflow as tf

        # Fix: quantization_config issue — patch Dense layer
        from tensorflow.keras.layers import Dense as _OrigDense
        class _PatchedDense(_OrigDense):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        MODELS['eye'] = {
            'model': tf.keras.models.load_model(
                os.path.join(MODELS_DIR, 'eye_cnn_model.h5'),
                custom_objects={'Dense': _PatchedDense},
                compile=False
            ),
        }
        with open(os.path.join(MODELS_DIR, 'eye_class_indices.json')) as f:
            ci = json.load(f)
        MODELS['eye']['reverse'] = {v: k for k, v in ci.items()}
        print("[OK] Eye CNN loaded")
        print("[OK] Eye classes:", list(MODELS['eye']['reverse'].values()))
    except Exception as e:
        print(f"[WARN] Eye: {e}")

      # ── LUNG ───────────────────────────────────────────────
    try:
        MODELS['lung'] = {
            'model':      joblib.load(os.path.join(MODELS_DIR, 'lung_stacking_model.pkl')),
            'le_gender':  joblib.load(os.path.join(MODELS_DIR, 'lung_le_gender.pkl')),
            'le_stage':   joblib.load(os.path.join(MODELS_DIR, 'lung_le_stage.pkl')),
            'le_family':  joblib.load(os.path.join(MODELS_DIR, 'lung_le_family.pkl')),
            'le_smoking': joblib.load(os.path.join(MODELS_DIR, 'lung_le_smoking.pkl')),
            'le_treat':   joblib.load(os.path.join(MODELS_DIR, 'lung_le_treatment.pkl')),
        }
        with open(os.path.join(MODELS_DIR, 'lung_metadata.json')) as f:
            MODELS['lung']['meta'] = json.load(f)
        print("[OK] Lung model loaded")
    except Exception as e:
        print(f"[WARN] Lung: {e}")
load_models()

# ── Helper: Save to history (DB-backed) ─────────────────────
def save_to_history(session_id, disease, inputs, result):
    """Save prediction — uses logged-in user email if available"""
    user = session.get('user')
    if user and user.get('email'):
        db_save_prediction(user['email'], disease, inputs, result)

# ── Helper: Doctor recommendations ────────────────────────
DOCTOR_MAP = {
    'heart':    {'HIGH RISK': 'Cardiologist (હૃદય રોગ નિષ્ણાત)',   'MEDIUM RISK': 'General Physician', 'LOW RISK': 'Annual Checkup'},
    'brain':    {'HIGH RISK': 'Neurologist (મગજ નિષ્ણાત)',          'MEDIUM RISK': 'General Physician', 'LOW RISK': 'Lifestyle Check'},
    'diabetes': {'HIGH RISK': 'Endocrinologist (ડાયાબિટીસ નિષ્ણાત)','MEDIUM RISK': 'General Physician', 'LOW RISK': 'Diet Consultation'},
    'kidney':   {
        'Severe_Disease': 'Nephrologist (કિડની નિષ્ણાત) — Urgent',
        'High_Risk':      'Nephrologist (કિડની નિષ્ણાત)',
        'Moderate_Risk':  'General Physician + Nephrology Referral',
        'Low_Risk':       'General Physician',
        'No_Disease':     'Annual Kidney Function Test'
    },
    'eye': {
        'Glaucoma':      'Ophthalmologist (આંખ નિષ્ણાત) — Urgent',
        'Cataracts':     'Ophthalmologist (આંખ નિષ્ણાત)',
        'Bulging_Eyes':  'Ophthalmologist + Endocrinologist',
        'Crossed_Eyes':  'Ophthalmologist',
        'Uveitis':       'Ophthalmologist — Urgent',
        'Eye_diseases':  'Ophthalmologist (આંખ નિષ્ણાત) — Checkup Needed'
    },
    'lung': {
        'HIGH RISK': 'Oncologist (Lung Cancer Specialist) → Urgent!',
        'MEDIUM RISK': 'Pulmonologist → Oncology Referral',
        'LOW RISK': 'Pulmonologist → Regular Monitoring',
    }
}

# ── Pages ──────────────────────────────────────────────────
@app.route('/')
def index():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    if 'user' not in session:
        return redirect(url_for('login_page'))
    # Update last_active on home visit
    session['last_active'] = time.time()
    session.modified = True
    return render_template('index.html')

@app.route('/login')
def login_page():     return render_template('login.html')

# ── Google OAuth Routes ────────────────────────────────────
@app.route('/auth/google')
def google_login():
    redirect_uri = url_for('google_callback', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/auth/google/callback')
def google_callback():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info:
            user_info = google.get('https://www.googleapis.com/oauth2/v3/userinfo').json()
        email = user_info.get('email', '').lower()
        name  = user_info.get('name', email.split('@')[0])
        if not db_email_exists(email):
            db_create_user(email, name, None, verified=True, google=True)
        if 'sid' not in session:
            session['sid'] = str(uuid.uuid4())
        session.permanent      = True
        session['user']        = {'email': email, 'name': name}
        session['last_active'] = time.time()
        return redirect(url_for('index'))
    except Exception as e:
        print(f"[Google OAuth Error] {e}")
        return redirect(url_for('login_page') + '?error=google_failed')

def login_required(f):
    """Decorator: redirect to login if not logged in or session idle > 1hr"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login_page'))
        if 'IDLE_TIMEOUT' in dir() or 'IDLE_TIMEOUT' in globals():
            if is_session_idle():
                session.pop('user', None)
                session.pop('last_active', None)
                print("[SESSION] Idle timeout — auto logout")
                return redirect(url_for('login_page') + '?timeout=1')
        session['last_active'] = time.time()
        session.modified = True
        return f(*args, **kwargs)
    return decorated

@app.route('/heart')
@login_required
def heart_page():     return render_template('heart.html')
@app.route('/brain')
@login_required
def brain_page():     return render_template('brain.html')
@app.route('/diabetes')
@login_required
def diabetes_page():  return render_template('diabetes.html')
@app.route('/kidney')
@login_required
def kidney_page():    return render_template('kidney.html')
@app.route('/eye')
@login_required
def eye_page():       return render_template('eye.html')
@app.route('/lung')
@login_required
def lung_page():  return render_template('lung.html')
@app.route('/assistant')
@login_required
def assistant_page(): return render_template('assistant.html')
@app.route('/history')
@login_required
def history_page():   return render_template('history.html')
@app.route('/bmi')
@login_required
def bmi_page():       return render_template('bmi.html')
@app.route('/lifestyle')
@login_required
def lifestyle_page(): return render_template('lifestyle.html')

# ── Auth APIs ──────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def api_login():
    d             = request.json
    email         = d.get('email', '').strip().lower()
    pwd           = d.get('password', '')
    captcha_token = d.get('captcha_token', '')

    # ── 1. reCAPTCHA verification ─────────────────────────────
    if not captcha_token:
        return jsonify({'success': False, 'error': 'CAPTCHA verification required'})
    try:
        verify_data = urllib.parse.urlencode({
            'secret':   RECAPTCHA_SECRET,
            'response': captcha_token
        }).encode()
        req    = urllib.request.Request(
            'https://www.google.com/recaptcha/api/siteverify',
            data=verify_data
        )
        resp   = urllib.request.urlopen(req, timeout=5)
        result = json.loads(resp.read().decode())
        if not result.get('success'):
            return jsonify({'success': False, 'error': 'CAPTCHA failed. Please try again.'})
    except Exception as e:
        print(f"[reCAPTCHA Error] {e}")
        if RECAPTCHA_SECRET != '6LeIxAcTAAAAAGG-vFI1TnRWxMZNFuojJ4WifJWe':
            return jsonify({'success': False, 'error': 'CAPTCHA check failed.'})

    # ── 2. Credentials check with DEBUGGING (SQLite) ────────────
    print(f"\n--- Login Attempt for: {email} ---") # Debug Log
    u = db_get_user(email)
    
    if not u:
        print(f"[DEBUG] User NOT found in database for email: {email}") 
        # જો અહીં "User NOT found" આવે, તો તમારો ડેટાબેઝ રીસેટ થઈ રહ્યો છે.
        return jsonify({'success': False, 'error': 'Invalid email or password'})

    print(f"[DEBUG] User found: {u['name']}") # Debug Log
    
    # Check if password exists and matches
    stored_pwd = u.get('pwd')
    if not stored_pwd:
        print(f"[DEBUG] No password stored for user: {email}")
        return jsonify({'success': False, 'error': 'Account password not set'})

    if stored_pwd != hash_pwd(pwd):
        print(f"[DEBUG] Password Mismatch for user: {email}") # Debug Log
        return jsonify({'success': False, 'error': 'Invalid email or password'})

    # ── 3. Set session ────────────────────────────────────────
    print(f"[DEBUG] Password Match! Logging in {email}...") 
    session.permanent      = True
    session['user']        = {'email': email, 'name': u['name']}
    session['last_active'] = time.time()
    
    print(f"[LOGIN SUCCESS] {email} logged in successfully ✅")
    return jsonify({'success': True, 'name': u['name']})


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop('user', None)
    session.pop('last_active', None)
    return jsonify({'success': True})

# ── Ping — extend session from frontend ───────────────────
@app.route('/api/ping', methods=['POST'])
def api_ping():
    if 'user' in session:
        session['last_active'] = time.time()
        session.modified = True
        return jsonify({'success': True})
    return jsonify({'success': False})

# ── Idle check helper ──────────────────────────────────────
def is_session_idle():
    last = session.get('last_active')
    return bool(last and (time.time() - last) > IDLE_TIMEOUT)

@app.route('/api/me')
def api_me():
    u = session.get('user')
    if not u:
        return jsonify({'logged_in': False})
    email    = u.get('email', '')
    name     = u.get('name', 'User')
    userdata = db_get_user(email) or {}
    initials = ''.join(w[0] for w in name.split() if w)[:2].upper()
    return jsonify({
        'logged_in': True,
        'name':      name,
        'email':     email,
        'initials':  initials,
        'created':   userdata.get('created_at', ''),
        'google':    bool(userdata.get('google', False)),
        'verified':  bool(userdata.get('verified', False)),
    })


# ── Check if email already exists ─────────────────────────────────
@app.route('/api/check-email', methods=['POST'])
def api_check_email():
    email = request.json.get('email', '').strip().lower()
    return jsonify({'exists': db_email_exists(email)})


# ── Send OTP via Email (Gmail SMTP) ──────────────────────
@app.route('/api/send-otp', methods=['POST'])
def api_send_otp():
    d     = request.json
    email = d.get('email', '').strip().lower()

    if not email:
        return jsonify({'success': False, 'error': 'Email required'})

    # ── Duplicate email check (DB) ─────────────────────────
    if db_email_exists(email):
        return jsonify({'success': False, 'error': 'Email already registered. Please login.'})

    # ── Generate 6-digit OTP ───────────────────────────────
    otp     = str(random.randint(100000, 999999))
    expires = time.time() + 120  # 2 minutes

    OTP_STORE[email] = {
        'otp':      otp,
        'expires':  expires,
        'attempts': 0
    }

    # ── Send via Gmail SMTP ────────────────────────────────
    gmail_user = os.environ.get('GMAIL_USER', '')
    gmail_pass = os.environ.get('GMAIL_APP_PASSWORD', '')
    otp_sent   = False

    if gmail_user and gmail_pass:
        try:
            import smtplib
            # SMTP સર્વર સાથે કનેક્ટ થતી વખતે ટાઈમઆઉટ સેટ કરો જેથી સર્વર હેંગ ન થાય
           

            from email.mime.multipart import MIMEMultipart
            from email.mime.text      import MIMEText

            html_body = f"""
            <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;
                        background:#0a0e27;color:#f0f4ff;border-radius:16px;padding:32px;">
              <h2 style="color:#00d9ff;margin:0 0 4px;font-size:28px;letter-spacing:-1px">VitalsAI</h2>
              <p style="color:#9ca3b5;margin:0 0 28px;font-size:12px;letter-spacing:1px">
                INTELLIGENT HEALTH PREDICTION SYSTEM
              </p>
              <p style="margin:0 0 16px;font-size:15px">Your One-Time Password (OTP) for account verification:</p>
              <div style="background:#1a1f3a;border:2px solid #00d9ff;border-radius:12px;
                          padding:24px;text-align:center;margin:0 0 24px">
                <span style="font-size:42px;font-weight:800;letter-spacing:14px;
                             color:#00d9ff;font-family:monospace">{otp}</span>
              </div>
              <table style="width:100%;margin:0 0 20px">
                <tr>
                  <td style="color:#9ca3b5;font-size:13px">⏱️ Valid for</td>
                  <td style="color:#f0f4ff;font-size:13px;font-weight:700;text-align:right">2 minutes only</td>
                </tr>
                <tr>
                  <td style="color:#9ca3b5;font-size:13px">📧 Sent to</td>
                  <td style="color:#00d9ff;font-size:13px;text-align:right">{email}</td>
                </tr>
              </table>
              <p style="color:#ff6b9d;font-size:12px;margin:0 0 20px;
                        background:rgba(255,22,84,.1);border:1px solid #ff1654;
                        border-radius:8px;padding:10px 14px;">
                🔒 Never share this OTP with anyone. VitalsAI will never ask for your OTP.
              </p>
              <hr style="border:none;border-top:1px solid #2a2a50;margin:20px 0">
              <p style="color:#6b7280;font-size:11px;margin:0">
                If you did not request this, please ignore this email.
              </p>
            </div>
            """

            msg            = MIMEMultipart('alternative')
            msg['Subject'] = f'VitalsAI — Your OTP is {otp}'
            msg['From']    = f'VitalsAI <{gmail_user}>'
            msg['To']      = email
            msg.attach(MIMEText(f'Your VitalsAI OTP is: {otp}. Valid for 2 minutes. Do not share.', 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=10) as server:
                server.login(gmail_user, gmail_pass)
                server.sendmail(gmail_user, email, msg.as_string())

            otp_sent = True
            print(f"[OTP] Email sent to {email} ✅")

        except Exception as e:
            print(f"[OTP] Gmail error: {e}")

    # ── Console fallback ───────────────────────────────────
    if not otp_sent:
        print(f"\n{'='*50}")
        print(f"  📧 VitalsAI OTP {'(Gmail not configured)' if not gmail_user else '(Gmail error)'}")
        print(f"  Email : {email}")
        print(f"  OTP   : {otp}")
        print(f"  Valid for 2 minutes")
        print(f"{'='*50}\n")

    return jsonify({
        'success': True,
        'demo':    not otp_sent,
        'message': f'OTP sent to {email}' if otp_sent else f'Demo — OTP: {otp}'
    })



# ── Signup with OTP verification ──────────────────────────
@app.route('/api/signup', methods=['POST'])
def api_signup():
    d      = request.json
    email  = d.get('email', '').strip().lower()
    name   = d.get('name', '').strip()
    pwd    = d.get('password', '')
    otp_in = d.get('otp', '').strip()

    # Basic validation
    if not email or not name or not pwd:
        return jsonify({'success': False, 'error': 'All fields required'})
    if len(pwd) < 6:
        return jsonify({'success': False, 'error': 'Password min 6 characters'})
    if db_email_exists(email):
        return jsonify({'success': False, 'error': 'Email already registered. Please login.'})
    if not otp_in:
        return jsonify({'success': False, 'error': 'OTP verification required'})

    # ── Verify OTP ────────────────────────────────────────
    record = OTP_STORE.get(email)
    if not record:
        return jsonify({'success': False, 'error': 'OTP not sent or expired. Request a new one.'})
    if time.time() > record['expires']:
        OTP_STORE.pop(email, None)
        return jsonify({'success': False, 'error': 'OTP expired. Click Resend.'})

    record['attempts'] = record.get('attempts', 0) + 1
    if record['attempts'] > 5:
        OTP_STORE.pop(email, None)
        return jsonify({'success': False, 'error': 'Too many wrong attempts. Request a new OTP.'})
    if otp_in != record['otp']:
        remaining = 5 - record['attempts']
        return jsonify({'success': False, 'error': f'Wrong OTP. {remaining} attempts left.'})

    # ✅ OTP correct
    OTP_STORE.pop(email, None)

    # ── Create account (SQLite) ──────────────────────────
    db_create_user(email, name, hash_pwd(pwd), verified=True, google=False)
    print(f"[SIGNUP] New user: {email} | Email OTP verified ✅  → saved to DB")
    return jsonify({'success': True})


# ══════════════════════════════════════════════════════════════════════
# HOW TO SETUP REAL SMS (Optional — Twilio):
#
#   pip install twilio
#
#   Add to your .env file:
#     TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxx
#     TWILIO_AUTH_TOKEN=your_auth_token
#     TWILIO_FROM_NUMBER=+1234567890
#
#   Get free trial at: https://www.twilio.com
#   (Free trial gives ~$15 credit = ~1000 SMS)
#
# In DEMO mode (no Twilio):
#   OTP is printed in terminal/console.
#   The frontend will also show OTP in the success message.
# ══════════════════════════════════════════════════════════════════════



# ── Lifestyle Recommendation API ──────────────────────────
# ── Helper: Send Lifestyle Recommendations via Email ────────────────
# ધ્યાન રાખજો: આ ફંક્શનની ઉપર કોઈ @app.route ન હોવું જોઈએ
def send_lifestyle_email(email, recs):
    gmail_user = os.environ.get('GMAIL_USER', '')
    gmail_pass = os.environ.get('GMAIL_APP_PASSWORD', '')
    
    if not gmail_user or not gmail_pass:
        print("[ERROR] Gmail credentials missing. Cannot send recommendation email.")
        return False

    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText

        diet_list = "<br>• ".join(recs.get('diet', []))
        ex_list = "<br>• ".join(recs.get('exercise', []))
        sleep_list = "<br>• ".join(recs.get('sleep', []))
        med_list = "<br>• ".join(recs.get('medical', []))

        html_body = f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;
                    background:#0a0e27;color:#f0f4ff;border-radius:16px;padding:32px;border:1px solid #00d9ff;">
          <h2 style="color:#00d9ff;text-align:center;">🌿 VitalsAI Health Guide</h2>
          <p style="text-align:center;color:#9ca3b5;">We've analyzed your health profile and prepared some personalized tips for you.</p>
          <hr style="border:none;border-top:1px solid #2d3a5a;margin:20px 0">
          <div style="background:#1a1f3a;padding:20px;border-radius:12px;margin-bottom:20px;">
            <h3 style="color:#00ff88;">🍎 Diet Recommendations</h3>
            <p style="color:#f0f4ff;line-height:1.6;">{diet_list}</p>
          </div>
          <div style="background:#1a1f3a;padding:20px;border-radius:12px;margin-bottom:20px;">
            <h3 style="color:#ff6b35;">🏃 Exercise Tips</h3>
            <p style="color:#f0f4ff;line-height:1.6;">{ex_list}</p>
          </div>
          <div style="background:#1a1f3a;padding:20px;border-radius:12px;margin-bottom:20px;">
            <h3 style="color:#c77dff;">😴 Sleep & Wellness</h3>
            <p style="color:#f0f4ff;line-height:1.6;">{sleep_list}</p>
          </div>
          <div style="background:#1a1f3a;padding:20px;border-radius:12px;margin-bottom:20px;border-left:4px solid #ff1654;">
            <h3 style="color:#ff1654;">🏥 Important Medical Advice</h3>
            <p style="color:#f0f4ff;line-height:1.6;">{med_list}</p>
          </div>
          <p style="font-size:12px;color:#6b7280;text-align:center;margin-top:30px;">
            Disclaimer: This is an AI-generated guide. Please consult a certified doctor for medical diagnosis.
          </p>
        </div>
        """
        msg = MIMEMultipart('alternative')
        msg['Subject'] = '🌿 Your Personalized Health Tips from VitalsAI'
        msg['From'] = f'VitalsAI Health <{gmail_user}>'
        msg['To'] = email
        msg.attach(MIMEText("Please use an HTML viewer to see your personalized health guide.", 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, email, msg.as_string())
        return True
    except Exception as e:
        print(f"[Email Error] {e}")
        return False

# ── Lifestyle Recommendation API ──────────────────────────
@app.route('/api/recommend', methods=['POST'])
def recommend():
    d        = request.json
    bmi      = float(d.get('bmi', 22))
    bp       = int(d.get('bp', 120))
    chol     = int(d.get('chol', 180))
    smoker   = int(d.get('smoker', 0))
    activity = int(d.get('activity', 1))
    diabetes = int(d.get('diabetes', 0))
    sleep    = int(d.get('sleep', 7))
    stress   = d.get('stress') or 'low'
    age      = int(d.get('age', 30))
    diet_recs = []; exercise_recs = []; sleep_recs = []; medical_recs = []
    score = 100

    # --- BMI Logic ---
    if bmi > 30:
        diet_recs     += ["Follow a calorie-deficit diet", "Avoid junk food", "Eat more fruits and vegetables"]
        exercise_recs += ["Start 45 minutes of daily walking", "Do cardio exercise 4 times per week"]
        score -= 20
    elif bmi > 25:
        diet_recs     += ["Eat a healthy balanced diet", "Reduce processed foods"]
        exercise_recs += ["Daily 30 min brisk walking"]
        score -= 10
    elif bmi < 18.5:
        diet_recs += ["Eat protein-rich foods such as eggs, paneer, and lentils", "Eat healthy fats such as nuts, avocado, and small amounts of ghee"]
        score -= 10
    else:
        diet_recs.append("BMI is normal. Continue a balanced diet")

    # --- BP Logic ---
    if bp >= 140:
        diet_recs    += ["Follow a low-sodium diet", "Follow the DASH diet"]
        medical_recs += ["Monitor blood pressure daily", "Consult a doctor"]
        score -= 15
    elif bp >= 130:
        diet_recs.append("Reduce salt intake")
        score -= 5

    # --- Cholesterol Logic ---
    if chol >= 240:
        diet_recs    += ["Avoid saturated fats such as butter and fried food", "Eat oats and fiber-rich foods"]
        medical_recs.append("Get a lipid profile test")
        score -= 10
    elif chol >= 200:
        diet_recs.append("Prefer healthy fats such as olive oil and nuts")
        score -= 5

    # --- Smoker Logic ---
    if smoker:
        medical_recs  += ["Stop smoking immediately", "Get medical help to quit smoking"]
        exercise_recs.append("Exercise can help with smoking cessation")
        score -= 20

    # --- Activity Logic ---
    if not activity:
        exercise_recs += ["Start 30 minutes of daily walking", "Use stairs instead of elevators when possible"]
        score -= 10
    else:
        exercise_recs.append("Exercise excellent! Niyamit chalalu rakho")

    # --- Diabetes Logic ---
    if diabetes:
        diet_recs    += ["Avoid sugar and refined carbohydrates", "Prefer low-glycemic-index foods"]
        medical_recs += ["Monitor blood glucose daily", "Get an HbA1c test every 3 months"]
        score -= 15

    # --- Sleep Logic ---
    if sleep < 6:
        sleep_recs += ["Try to sleep 7 to 8 hours", "Turn off phone and TV 1 hour before sleep"]
        score -= 10
    elif sleep > 9:
        sleep_recs.append("Excess sleep can also be harmful. 7 to 8 hours is ideal")
    else:
        sleep_recs.append("Your sleep schedule is excellent")

    # --- Stress Logic ---
    if stress == 'high':
        sleep_recs    += ["Try 10 to 15 minutes of daily meditation", "Practice deep breathing exercises"]
        exercise_recs.append("Yoga is helpful for stress relief")
        score -= 10
    elif stress == 'medium':
        sleep_recs.append("Try relaxation techniques")

    # --- Age Logic ---
    if age >= 50:
        medical_recs += ["Get a yearly full-body checkup", "Check vitamin D, B12, and iron levels"]
    elif age >= 40:
        medical_recs.append("Test blood pressure, blood sugar, and cholesterol every year")
    else:
        medical_recs.append("Get a preventive health checkup every 2 years")

    if not diet_recs:     diet_recs     = ["Balanced diet lo — daal, chaval, shaak, fruit"]
    if not exercise_recs: exercise_recs = ["Continue exercising"]
    if not sleep_recs:    sleep_recs    = ["Your sleep schedule is excellent"]

    score      = max(0, min(100, score))
    risk_level = 'high' if score < 50 else ('medium' if score < 75 else 'low')

    # ── EMAIL NOTIFICATION LOGIC ──────────────────────────
    user = session.get('user')
    email_sent = False
    if user and user.get('email'):
        if risk_level in ['medium', 'high']:
            email_sent = send_lifestyle_email(user['email'], {
                'diet': diet_recs, 
                'exercise': exercise_recs, 
                'sleep': sleep_recs, 
                'medical': medical_recs
            })

    return jsonify({
        'health_score': score, 'risk_level': risk_level,
        'diet': diet_recs, 'exercise': exercise_recs,
        'sleep': sleep_recs, 'medical': medical_recs,
        'email_sent': email_sent
    })

# ── AI Chatbot API (Claude AI Powered) ────────────────────
CHAT_SYSTEM = """You are the AI Health Assistant for "Vitals AI" — a health prediction and wellness platform.

DEFAULT LANGUAGE: Always respond in English.
LANGUAGE RULE: Always reply in clear English, even if the user writes in another language.

For ANY disease or health question, structure your response:

**[Disease Name]**
**Symptoms:**
• List main symptoms
**Causes:**
• List main causes
**Treatment & Solutions:**
• Home remedies + Medical treatment
**When to See a Doctor:**
• Warning signs

Rules:
- Default language is English
- Every answer must be FRESH and specific — never give canned replies
- For wellness questions, answer naturally
- Always end with: "Please consult a doctor for personalized advice."
- For emergencies mention: Call 108 immediately
- You are NOT a doctor — health education only"""

ADVANCED_CHAT_INSTRUCTIONS = """Advanced VitalsAI capabilities:
- You receive a JSON context from the active page: page name, form values, visible result, and local_prediction.
- If the user asks to predict/check risk, use local_prediction and form values to provide a clear screening-style answer.
- If the user asks for explanation, explain the top risk factors and protective factors from context.
- If visible_result is present, summarize and explain that result before adding advice.
- You may answer broader health questions beyond the current disease: fever, cough/cold, headache, acidity, cholesterol, anemia, thyroid, stress, hydration, vaccines, diet, sleep, exercise, etc.
- Keep answers concise, practical, and in clear English.
- Do not claim a confirmed diagnosis. Say this is educational/screening support.
- For emergency symptoms, advise urgent medical care / Call 108 immediately."""

def _chat_is_predict_intent(msg):
    q = (msg or '').lower()
    return any(k in q for k in [
        'predict', 'prediction', 'risk', 'probability', 'check me',
        'risk score', 'risk level', 'screening result', 'estimate',
        'status'
    ])

def _chat_is_explain_intent(msg):
    q = (msg or '').lower()
    return any(k in q for k in [
        'explain', 'why', 'reason', 'result', 'summary',
        'samjavo', 'samja', 'સમજ', 'કારણ', 'પરિણામ'
    ])

def _format_chat_prediction(pred):
    import html as _html
    if not pred or pred.get('score') is None:
        return (
            "<b>Prediction requires form data.</b><br>"
            "Please fill the current disease form first, then ask: <b>predict my risk</b>."
        )
    disease = _html.escape(str(pred.get('disease', 'current disease')))
    status = _html.escape(str(pred.get('status', 'Unknown')))
    score = _html.escape(str(pred.get('score', '')))
    reasons = pred.get('reasons') or []
    protect = pred.get('protect') or []
    reason_html = '<br>• '.join(_html.escape(str(x)) for x in reasons[:6]) or 'Not enough strong risk factors found'
    protect_html = '<br>• '.join(_html.escape(str(x)) for x in protect[:4])
    html = (
        f"<b>Chatbot Screening Prediction</b><br>"
        f"Disease/Page: <b>{disease}</b><br>"
        f"Status: <b>{status}</b><br>"
        f"Probability hint: <b>{score}%</b><br><br>"
        f"<b>Explanation:</b><br>• {reason_html}"
    )
    if protect_html:
        html += f"<br><br><b>Protective points:</b><br>• {protect_html}"
    html += (
        "<br><br><i>This is an AI screening estimate, not a confirmed diagnosis. "
        "Please consult a doctor for personalized advice.</i>"
    )
    return html

from dotenv import load_dotenv
import os

load_dotenv() # આ લાઇન .env ફાઇલમાંથી ડેટા લોડ કરે છે
api_key = os.getenv("GEMINI_API_KEY")


@app.route('/api/chat', methods=['POST'])
def chat():
    import urllib.request as _ur
    import json as _json
    import re as _re

    data    = request.json or {}
    msg     = data.get('message', '').strip()
    history = data.get('history', [])
    mode    = data.get('mode', 'ai')
    context = data.get('context') or {}
    local_prediction = context.get('local_prediction') or data.get('local_prediction') or {}
    visible_result   = (context.get('visible_result') or '').strip() if isinstance(context, dict) else ''

    if not msg:
        return jsonify({'response': 'Please enter a message.'})

    # ── OpenRouter AI Mode ────────────────────────────────────
    OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
    if OPENROUTER_KEY:
        try:
            context_text = _json.dumps(context, ensure_ascii=False)[:3000] if context else '{}'

            system_prompt = (
                CHAT_SYSTEM + "\n\n" + ADVANCED_CHAT_INSTRUCTIONS +
                "\n\nCurrent page context: " + context_text
            )

            # Build messages — system + history + user
            messages = [{"role": "system", "content": system_prompt}]
            for h in history[-8:]:
                role = h.get('role', 'user')
                if role in ('user', 'assistant'):
                    messages.append({"role": role, "content": str(h.get('content', ''))[:500]})
            messages.append({"role": "user", "content": msg})

            payload = _json.dumps({
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": messages,
                "max_tokens": 2048,
                "temperature": 0.75,
                "top_p": 0.95
            }).encode('utf-8')

            req = _ur.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {OPENROUTER_KEY}",
                    "HTTP-Referer":  "http://localhost:5000",
                    "X-Title":       "VitalsAI Health Assistant"
                },
                method='POST'
            )

            with _ur.urlopen(req, timeout=30) as resp:
                result = _json.loads(resp.read())
                reply  = result['choices'][0]['message']['content']

            # Format markdown → HTML
            reply_html = _re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', reply)
            reply_html = _re.sub(r'\*(.*?)\*',     r'<i>\1</i>', reply_html)
            reply_html = _re.sub(r'^#{1,3}\s+(.+)$', r'<span class="sec-head">\1</span>', reply_html, flags=_re.MULTILINE)
            reply_html = _re.sub(r'^[-•]\s+(.+)$',   r'<li>\1</li>', reply_html, flags=_re.MULTILINE)
            reply_html = reply_html.replace('\n', '<br>')

            print(f"[OpenRouter OK] len={len(reply)}")
            return jsonify({'response': reply_html, 'source': 'openrouter'})

        except Exception as e:
            err = str(e)
            if hasattr(e, 'read'):
                try: err = e.read().decode()
                except: pass
            print(f"[OpenRouter Error] {err}")
            # Fall through to KB

    # ── Context-aware prediction/result fallback ─────────────
    if _chat_is_predict_intent(msg) and local_prediction:
        return jsonify({'response': _format_chat_prediction(local_prediction), 'source': 'context_prediction'})

    if _chat_is_explain_intent(msg) and local_prediction:
        return jsonify({'response': _format_chat_context_explanation(local_prediction), 'source': 'context_explanation'})

    if _chat_is_explain_intent(msg) and visible_result:
        safe_result = visible_result.replace('\n', '<br>')
        return jsonify({'response': (
            "<b>Current Result Explanation</b><br>" + safe_result +
            "<br><br>This is a screening estimate. Consult a doctor."
        ), 'source': 'context_result'})



    # ── Intelligent KB — 50+ topics, no API needed ──────────
    msg_lower = msg.lower()

    KB = [
        # ── DISEASES ────────────────────────────────────────
        (['hypertension','high bp','blood pressure','bp'],
         "<b>🫀 High Blood Pressure (Hypertension)</b><br><br><b>Symptoms:</b><br>• Headache, dizziness, blurred vision<br>• Chest pain, shortness of breath, nosebleeds<br><br><b>Causes:</b><br>• High salt diet, obesity, stress<br>• Smoking, alcohol, lack of exercise<br>• Genetics, age<br><br><b>Treatment:</b><br>• Reduce sodium intake below 1500mg/day<br>• DASH diet — fruits, vegetables, low fat<br>• Exercise 30 min daily<br>• Avoid smoking & alcohol<br>• Take prescribed medication regularly<br>• Monitor BP daily at home<br><br><b>When to See Doctor:</b><br>• BP above 180/120 — seek emergency care<br><br><i>Please consult a doctor for personalized advice.</i>"),

        (['diabetes','sugar level','blood sugar','type 1','type 2','insulin'],
         "<b>💉 Diabetes Mellitus</b><br><br><b>Symptoms:</b><br>• Frequent urination, excessive thirst<br>• Blurred vision, slow wound healing<br>• Fatigue, unexplained weight loss<br>• Tingling in hands/feet<br><br><b>Causes:</b><br>• Insulin resistance (Type 2)<br>• Autoimmune destruction of beta cells (Type 1)<br>• Obesity, genetics, sedentary lifestyle<br><br><b>Treatment:</b><br>• Avoid sugar, refined carbs, white rice<br>• Eat low glycemic foods — oats, vegetables<br>• Exercise 45-60 min daily<br>• Monitor blood sugar daily<br>• HbA1c test every 3 months<br>• Metformin / insulin as prescribed<br><br><i>Please consult a doctor for personalized advice.</i>"),

        (['heart disease','cardiac','heart attack','chest pain','cholesterol','coronary'],
         "<b>❤️ Heart Disease</b><br><br><b>Symptoms:</b><br>• Chest pain/tightness, shortness of breath<br>• Irregular heartbeat, palpitations<br>• Fatigue, swelling in legs<br><br><b>Causes:</b><br>• High cholesterol, high BP, smoking<br>• Diabetes, obesity, family history<br><br><b>Treatment:</b><br>• Mediterranean diet — olive oil, fish, nuts<br>• Avoid saturated fats, processed food<br>• 30-45 min cardio daily<br>• No smoking, limit alcohol<br>• Statins / beta blockers as prescribed<br>• Yearly ECG checkup<br><br><b>Emergency:</b> Chest pain + left arm pain → Call 108 immediately!<br><br><i>Please consult a doctor for personalized advice.</i>"),

        (['kidney','renal','creatinine','kidney disease','kidney failure','ckd'],
         "<b>🫘 Kidney Disease (CKD)</b><br><br><b>Symptoms:</b><br>• Swelling in legs/ankles, fatigue<br>• Reduced urine output, nausea<br>• Itching, loss of appetite, confusion<br><br><b>Causes:</b><br>• Diabetes, hypertension (top causes)<br>• Chronic UTI, kidney stones<br>• NSAIDs overuse, contrast dyes<br><br><b>Treatment:</b><br>• Drink 2-3L water daily (unless fluid restricted)<br>• Low protein, low potassium, low sodium diet<br>• Avoid NSAIDs (ibuprofen, aspirin)<br>• Control BP and blood sugar<br>• Regular kidney function tests (eGFR, creatinine)<br><br><i>Please consult a nephrologist for personalized advice.</i>"),

        (['stroke','brain stroke','paralysis','facial droop','slurred speech'],
         "<b>🧠 Brain Stroke</b><br><br><b>Symptoms (FAST test):</b><br>• <b>F</b>ace drooping on one side<br>• <b>A</b>rm weakness<br>• <b>S</b>peech difficulty/slurring<br>• <b>T</b>ime to call emergency!<br><br><b>Causes:</b><br>• Blood clot blocking brain artery (ischemic)<br>• Brain blood vessel rupture (hemorrhagic)<br>• High BP, AFib, diabetes, smoking<br><br><b>Treatment:</b><br>• Call 108 IMMEDIATELY — golden hour is critical<br>• tPA clot-busting drug within 4.5 hours<br>• Control BP, blood sugar<br>• Rehabilitation: physio, speech therapy<br><br><b>⚠️ Stroke is a medical emergency — Call 108!</b><br><br><i>Please consult a neurologist immediately.</i>"),

        (['depression','sad','sadness','low mood','hopeless','worthless'],
         "<b>🧠 Depression</b><br><br><b>Symptoms:</b><br>• Persistent sadness, hopelessness<br>• Loss of interest in activities<br>• Sleep changes (too much or too little)<br>• Fatigue, difficulty concentrating<br>• Changes in appetite/weight<br><br><b>Causes:</b><br>• Brain chemistry imbalance (serotonin, dopamine)<br>• Trauma, chronic stress, grief<br>• Genetics, chronic illness<br><br><b>Treatment:</b><br>• Therapy: CBT (Cognitive Behavioral Therapy)<br>• Antidepressants (SSRIs) as prescribed<br>• Regular exercise — proven to help<br>• Social support, routine<br>• Avoid alcohol<br><br><b>Need help?</b> iCall India: 9152987821<br><br><i>Please consult a psychiatrist for personalized advice.</i>"),

        (['anxiety','panic attack','panic','worry','nervous','phobia'],
         "<b>😰 Anxiety / Panic Disorder</b><br><br><b>Symptoms:</b><br>• Excessive worry, restlessness<br>• Rapid heartbeat, sweating, trembling<br>• Shortness of breath, chest tightness<br>• Panic attacks (sudden intense fear)<br><br><b>Causes:</b><br>• Stress, trauma, genetics<br>• Caffeine, thyroid issues<br><br><b>Treatment:</b><br>• Deep breathing: 4-7-8 technique<br>• CBT therapy<br>• Mindfulness & meditation<br>• SSRIs / benzodiazepines as prescribed<br>• Limit caffeine & alcohol<br><br><i>Please consult a doctor for personalized advice.</i>"),

        (['thyroid','hypothyroid','hyperthyroid','tsh','goiter'],
         "<b>🦋 Thyroid Disorder</b><br><br><b>Hypothyroid Symptoms:</b><br>• Fatigue, weight gain, cold intolerance<br>• Dry skin, hair loss, constipation, depression<br><br><b>Hyperthyroid Symptoms:</b><br>• Weight loss, rapid heartbeat, sweating<br>• Anxiety, tremors, heat intolerance<br><br><b>Causes:</b><br>• Autoimmune (Hashimoto's, Graves' disease)<br>• Iodine deficiency, surgery, radiation<br><br><b>Treatment:</b><br>• Levothyroxine for hypothyroid<br>• Antithyroids / radioiodine for hyperthyroid<br>• TSH test every 6-12 months<br>• Iodized salt in diet<br><br><i>Please consult an endocrinologist for personalized advice.</i>"),

        (['asthma','wheezing','inhaler','bronchial','breathing'],
         "<b>🫁 Asthma</b><br><br><b>Symptoms:</b><br>• Wheezing, shortness of breath<br>• Chest tightness, chronic cough (worse at night)<br><br><b>Triggers:</b><br>• Dust, pollen, pet dander, smoke<br>• Cold air, exercise, respiratory infections<br><br><b>Treatment:</b><br>• Rescue inhaler (salbutamol) for acute attacks<br>• Controller inhaler (corticosteroids) daily<br>• Avoid triggers — air purifier, dust covers<br>• Spirometry test yearly<br>• Action plan for attacks<br><br><b>Emergency:</b> Blue lips, cannot speak → Call 108!<br><br><i>Please consult a pulmonologist for personalized advice.</i>"),

        (['dengue','dengue fever','platelet','aedes'],
         "<b>🦟 Dengue Fever</b><br><br><b>Symptoms:</b><br>• High fever (104°F), severe headache<br>• Pain behind eyes, joint/muscle pain<br>• Skin rash, nausea, low platelets<br><br><b>Warning Signs:</b><br>• Bleeding gums/nose, blood in urine<br>• Severe abdominal pain, persistent vomiting<br><br><b>Treatment:</b><br>• Rest and hydration — drink 3-4L water/day<br>• Paracetamol for fever (NOT ibuprofen/aspirin)<br>• Monitor platelets daily if below 100,000<br>• Hospitalize if platelets below 20,000<br>• Papaya leaf extract may help platelets<br><br><i>Please consult a doctor immediately for dengue.</i>"),

        (['migraine','migraine headache','aura','throbbing headache'],
         "<b>🤕 Migraine</b><br><br><b>Symptoms:</b><br>• Throbbing headache (one side)<br>• Nausea, vomiting, light/sound sensitivity<br>• Visual aura (zigzag lines, blind spots)<br><br><b>Triggers:</b><br>• Stress, hormonal changes, bright lights<br>• Certain foods (cheese, chocolate, wine)<br>• Irregular sleep, dehydration<br><br><b>Treatment:</b><br>• Triptans (sumatriptan) for acute attacks<br>• Dark quiet room, cold compress<br>• Preventive: topiramate, beta-blockers<br>• Identify & avoid personal triggers<br>• Stay hydrated, regular sleep schedule<br><br><i>Please consult a neurologist for personalized advice.</i>"),

        (['eye','vision','glaucoma','cataract','retina','uveitis','bulging eye'],
         "<b>👁️ Eye Health</b><br><br><b>Common Conditions:</b><br>• Glaucoma: increased eye pressure → vision loss<br>• Cataracts: cloudy lens → blurred vision<br>• Uveitis: eye inflammation → pain, redness<br><br><b>Symptoms to Watch:</b><br>• Sudden vision loss, eye pain<br>• Floaters, flashes of light<br>• Red eye, discharge<br><br><b>Prevention:</b><br>• 20-20-20 rule for screen time<br>• Vitamin A rich food (carrots, spinach)<br>• UV protection sunglasses<br>• Annual eye checkup<br>• Control diabetes & BP (top causes of blindness)<br><br><i>Please consult an ophthalmologist for personalized advice.</i>"),

        # ── WELLNESS ────────────────────────────────────────
        (['sleep','insomnia','cant sleep','sleepless','sleep disorder'],
         "<b>😴 Sleep Health</b><br><br><b>Tips for Better Sleep:</b><br>• Fix sleep & wake time — even weekends<br>• No screens 1 hour before bed (blue light blocks melatonin)<br>• Keep bedroom dark, cool (18-20°C), quiet<br>• Avoid caffeine after 2 PM<br>• No alcohol before bed (disrupts REM sleep)<br>• 20 min walk daily improves sleep quality<br>• Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s<br><br><b>Sleep Stages:</b><br>• Adults need 7-9 hours<br>• Elderly need 7-8 hours<br>• Teens need 8-10 hours<br><br><i>Chronic insomnia? Please consult a sleep specialist.</i>"),

        (['stress','anxiety','mental health','burnout','overwhelmed'],
         "<b>🧘 Stress & Mental Wellness</b><br><br><b>Signs of Chronic Stress:</b><br>• Headaches, muscle tension, fatigue<br>• Irritability, difficulty concentrating<br>• Sleep problems, digestive issues<br><br><b>Science-backed Relief:</b><br>• Exercise: 30 min/day reduces cortisol by 26%<br>• Meditation: 10 min/day changes brain structure<br>• Deep breathing: activates parasympathetic nervous system<br>• Journaling: write 3 gratitudes daily<br>• Social connection: talk to someone<br>• Nature walks: 20 min reduces stress hormones<br><br><b>Professional Help:</b><br>• iCall: 9152987821 | Vandrevala Foundation: 1860-2662-345<br><br><i>Please consult a mental health professional if needed.</i>"),

        (['exercise','workout','gym','yoga','fitness','cardio','running'],
         "<b>🏃 Exercise & Fitness Guide</b><br><br><b>Weekly Targets (WHO Guidelines):</b><br>• 150 min moderate cardio (brisk walk, cycling)<br>• OR 75 min vigorous (running, HIIT)<br>• 2x strength training per week<br>• Daily stretching 10 min<br><br><b>Benefits by Type:</b><br>• Cardio: heart health, weight loss, mood<br>• Strength: bone density, metabolism, posture<br>• Yoga: flexibility, stress, balance<br>• Swimming: joint-friendly, full body<br><br><b>Getting Started:</b><br>• Walk 30 min daily — simplest & most effective<br>• Start slow, build gradually<br>• Warm up 5 min before, cool down after<br><br><i>Consult a doctor before starting if you have medical conditions.</i>"),

        (['diet','nutrition','food','eating','meal','healthy food'],
         "<b>🍎 Nutrition & Healthy Diet</b><br><br><b>Daily Plate (My Plate Method):</b><br>• 50% vegetables & fruits<br>• 25% whole grains (brown rice, oats, roti)<br>• 25% lean protein (daal, eggs, fish, paneer)<br>• Plus: healthy fats (nuts, olive oil, ghee-small)<br><br><b>Foods to Avoid:</b><br>• Ultra-processed food, packaged snacks<br>• Sugary drinks (soda, packaged juice)<br>• Trans fats (vanaspati, margarine)<br>• White refined carbs in excess<br><br><b>Superfoods to Include:</b><br>• Turmeric, ginger, garlic (anti-inflammatory)<br>• Leafy greens (iron, folate)<br>• Berries (antioxidants)<br>• Nuts & seeds (omega-3, minerals)<br><br><i>Consult a dietitian for a personalized meal plan.</i>"),

        (['weight','obesity','weight loss','fat','overweight','bmi'],
         "<b>⚖️ Weight Management</b><br><br><b>Healthy Weight Loss:</b><br>• Calorie deficit of 500 cal/day = 0.5 kg/week loss<br>• Do NOT crash diet — slows metabolism<br><br><b>Evidence-based Strategies:</b><br>• Eat slowly — takes 20 min for brain to feel full<br>• Protein at every meal (keeps you fuller)<br>• Fill half plate with vegetables<br>• Drink water before meals<br>• Sleep 7-8 hours (poor sleep causes weight gain)<br>• 10,000 steps daily<br><br><b>BMI Guide:</b><br>• Under 18.5: Underweight<br>• 18.5-24.9: Normal ✅<br>• 25-29.9: Overweight<br>• 30+: Obese<br><br><i>Consult a dietitian for personalized weight loss plan.</i>"),

        (['water','hydration','dehydration','drink water'],
         "<b>💧 Hydration Guide</b><br><br><b>Daily Water Needs:</b><br>• Adults: 2.5-3.5 liters/day<br>• More in summer, during exercise<br>• 8-10 glasses = general guideline<br><br><b>Signs of Dehydration:</b><br>• Dark yellow urine (should be pale yellow)<br>• Headache, fatigue, dry mouth<br>• Dizziness, poor concentration<br><br><b>Hydration Tips:</b><br>• Start morning with 1-2 glasses water<br>• Eat water-rich foods: cucumber, watermelon<br>• Coconut water — best natural electrolyte<br>• Avoid excessive tea/coffee (mild diuretics)<br><br><i>Increase intake during illness, heat, and exercise.</i>"),

        (['vitamin','vitamin d','vitamin b12','iron deficiency','anemia','supplement'],
         "<b>💊 Vitamins & Deficiencies</b><br><br><b>Common Deficiencies in India:</b><br><br><b>Vitamin D:</b><br>• Symptoms: Bone pain, fatigue, muscle weakness<br>• Source: Sunlight 15 min/day, fatty fish, eggs<br>• Normal: 30-100 ng/mL<br><br><b>Vitamin B12:</b><br>• Symptoms: Fatigue, numbness, memory issues<br>• Source: Dairy, eggs, meat (vegetarians at risk)<br>• Supplement: 500-1000 mcg/day if deficient<br><br><b>Iron/Anemia:</b><br>• Symptoms: Fatigue, pale skin, breathlessness<br>• Source: Spinach, lentils, jaggery, meat<br>• Vitamin C enhances iron absorption<br><br><i>Please get blood tests done before supplementing.</i>"),

        (['fever','temperature','high temperature','pyrexia'],
         "<b>🌡️ Fever</b><br><br><b>Classification:</b><br>• Low grade: 99-100.4°F (37.2-38°C)<br>• Moderate: 100.4-103°F (38-39.4°C)<br>• High: Above 103°F — needs attention<br><br><b>Home Management:</b><br>• Paracetamol (500-1000mg) every 4-6 hours<br>• Luke warm sponging<br>• Rest and hydration (3-4L fluids)<br>• Light clothing<br><br><b>See Doctor If:</b><br>• Fever above 103°F (39.4°C)<br>• Lasts more than 3 days<br>• With stiff neck, rash, confusion<br>• Infant under 3 months with any fever<br><br><i>Do NOT give aspirin to children.</i>"),

        (['covid','corona','coronavirus','long covid'],
         "<b>🦠 COVID-19</b><br><br><b>Common Symptoms:</b><br>• Fever, cough, fatigue, body aches<br>• Loss of taste/smell, sore throat<br>• Shortness of breath (severe cases)<br><br><b>Management:</b><br>• Rest, hydration, paracetamol for fever<br>• Isolate for 5-7 days from symptom onset<br>• Monitor oxygen saturation (should be above 95%)<br>• Seek care if SpO2 drops below 93%<br><br><b>Long COVID:</b><br>• Fatigue, brain fog, breathlessness weeks after<br>• Rehabilitation and gradual return to activity<br><br><b>Prevention:</b><br>• Vaccination, masks in crowded places<br>• Hand hygiene<br><br><i>Please consult a doctor for personalized care.</i>"),

        (['back pain','spine','lower back','sciatica','spondylitis'],
         "<b>🦴 Back Pain</b><br><br><b>Common Types:</b><br>• Muscle strain (most common)<br>• Disc herniation (radiates to leg — sciatica)<br>• Spondylitis (inflammatory)<br><br><b>Home Remedies:</b><br>• Ice first 48 hours → then heat<br>• Gentle stretching, cat-cow pose<br>• Avoid prolonged sitting<br>• Proper ergonomic chair setup<br><br><b>Exercises that Help:</b><br>• Core strengthening (planks, bridges)<br>• Walking, swimming<br>• Yoga (child's pose, pigeon pose)<br><br><b>See Doctor If:</b><br>• Pain radiates below knee<br>• Numbness/weakness in legs<br>• Bladder/bowel changes<br><br><i>Please consult an orthopedic for personalized advice.</i>"),
    ]

    # Smart matching — check multiple keywords
    best_match = None
    best_score = 0
    for keywords, response in KB:
        score = sum(1 for k in keywords if k in msg_lower)
        if score > best_score:
            best_score = score
            best_match = response

    if best_match and best_score > 0:
        return jsonify({'response': best_match, 'source': 'local'})

    # Default response
    return jsonify({'response': (
        "I can help you with health questions! 😊<br><br>"
        "<b>Ask me about:</b><br>"
        "🫀 BP / Hypertension &nbsp;|&nbsp; 💉 Diabetes &nbsp;|&nbsp; ❤️ Heart Disease<br>"
        "🫘 Kidney Disease &nbsp;|&nbsp; 🧠 Brain Stroke / Depression<br>"
        "🦋 Thyroid &nbsp;|&nbsp; 🫁 Asthma &nbsp;|&nbsp; 🦟 Dengue &nbsp;|&nbsp; 🤕 Migraine<br>"
        "😴 Sleep &nbsp;|&nbsp; 🧘 Stress &nbsp;|&nbsp; 🏃 Exercise &nbsp;|&nbsp; 🍎 Diet<br>"
        "⚖️ Weight Loss &nbsp;|&nbsp; 💧 Hydration &nbsp;|&nbsp; 💊 Vitamins &nbsp;|&nbsp; 🌡️ Fever<br><br>"
        "<i>Type your health question in detail for best results!</i>"
    ), 'source': 'local'})

@app.route('/api/last-prediction')
def get_last_prediction():
    user = session.get('user')
    if not user or not user.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    with get_db() as conn:
        row = conn.execute(
            "SELECT disease, risk, result FROM predictions WHERE user_email=? ORDER BY created_at DESC LIMIT 1", 
            (user['email'],)
        ).fetchone()
        
        if row:
            return jsonify({
                'disease': row['disease'],
                'risk': row['risk'],
                'result': json.loads(row['result']) if isinstance(row['result'], str) else row['result']
            })
    return jsonify({'no_prediction': True})

# ── BMI Calculator API ─────────────────────────────────────
@app.route('/api/bmi', methods=['POST'])
def calc_bmi():
    try:
        data    = request.json
        weight  = float(data.get('weight', 0))   # kg
        height  = float(data.get('height', 0))   # cm
        age     = int(data.get('age', 25))
        gender  = data.get('gender', 'male')

        if height <= 0 or weight <= 0:
            return jsonify({'error': 'Invalid input'}), 400

        h_m  = height / 100
        bmi  = round(weight / (h_m ** 2), 1)
        ibw  = round(22.5 * (h_m ** 2), 1)          # Ideal body weight
        diff = round(weight - ibw, 1)

        if bmi < 18.5:
            cat = 'Underweight'
            cat_guj = 'ઓછું વજન'
            color = '#2196F3'
            advice = 'વધુ પ્રોટીન અને કેલરી લો. Doctor ને મળો.'
        elif bmi < 25:
            cat = 'Normal'
            cat_guj = 'સામાન્ય'
            color = '#4CAF50'
            advice = 'સ્વસ્થ છો! નિયમિત exercise અને balanced diet ચાલુ રાખો.'
        elif bmi < 30:
            cat = 'Overweight'
            cat_guj = 'વધારે વજન'
            color = '#FF9800'
            advice = 'Exercise વધારો, processed food ઓછો કરો.'
        elif bmi < 35:
            cat = 'Obese Class I'
            cat_guj = 'સ્થૂળતા'
            color = '#F44336'
            advice = 'Doctor ને મળો. Diet plan follow કરો.'
        else:
            cat = 'Obese Class II+'
            cat_guj = 'ગંભીર સ્થૂળતા'
            color = '#B71C1C'
            advice = 'તાત્કાલિક Doctor ને મળો.'

        return jsonify({
            'bmi': bmi, 'category': cat, 'category_guj': cat_guj,
            'color': color, 'advice': advice,
            'ideal_weight': ibw, 'weight_diff': diff,
            'age': age, 'gender': gender
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Heart Predict ──────────────────────────────────────────
@app.route('/predict/heart', methods=['POST'])
def predict_heart():
    try:
        m = MODELS.get('heart')
        if not m: return jsonify({'error': 'Heart model not loaded.'}), 500
        data = request.json
        bp   = float(data.get('BloodPressure', 120))
        cho  = float(data.get('Cholesterol', 180))
        inputs = {
            'Age':          float(data.get('Age', 0)),
            'BMI':          float(data.get('BMI', 0)),
            'HighBP':       1 if bp >= 140 else 0,
            'HighChol':     1 if cho >= 240 else 0,
            'Diabetes':     int(data.get('Diabetes', 0)),
            'Smoker':       int(data.get('Smoker', 0)),
            'PhysActivity': int(data.get('PhysActivity', 1)),
            'GenHlth':      int(data.get('GenHlth', 3)),
            'Sex':          int(data.get('Sex', 0)),
        }
        patient = pd.DataFrame([inputs])
        prob    = float(m['model'].predict_proba(patient)[0, 1])
        risk    = 'HIGH RISK' if prob >= 0.65 else ('MEDIUM RISK' if prob >= 0.40 else 'LOW RISK')
        doctor  = DOCTOR_MAP['heart'].get(risk, 'General Physician')

        # ── XAI Explanation ─────────────────────────────────
        why_high, why_low, suggestions, lifestyle = [], [], [], {}
        age_v = inputs['Age']; bmi_v = inputs['BMI']

        if inputs['HighBP']:
            why_high.append(f"High Blood Pressure detected (BP ≥ 140 mmHg) — major heart disease risk factor")
        else:
            why_low.append("Blood Pressure is normal — good cardiovascular indicator")
        if inputs['HighChol']:
            why_high.append("High Cholesterol (≥ 240 mg/dL) — causes arterial plaque buildup")
        else:
            why_low.append("Cholesterol within healthy range — lower arterial risk")
        if inputs['Smoker']:
            why_high.append("Smoking — damages blood vessels and reduces oxygen supply to heart")
            suggestions.append("Quit smoking immediately — reduces heart risk by 50% within 1 year")
        if inputs['Diabetes']:
            why_high.append("Diabetes — doubles the risk of coronary artery disease")
            suggestions.append("Keep HbA1c < 7% and monitor blood sugar daily")
        if not inputs['PhysActivity']:
            why_high.append("Physical inactivity — weakens heart muscle and raises cholesterol")
            suggestions.append("Start 30 min brisk walk daily — reduces heart risk by 35%")
        else:
            why_low.append("Regular physical activity — strengthens heart and improves circulation")
        if bmi_v > 30:
            why_high.append(f"Obese BMI ({bmi_v}) — increases strain on heart")
            suggestions.append("Reduce BMI to 18.5–24.9 through diet + exercise")
        elif bmi_v > 25:
            why_high.append(f"Overweight BMI ({bmi_v}) — moderate cardiovascular risk")
        else:
            why_low.append(f"BMI ({bmi_v}) in healthy range — lower cardiac load")
        if inputs['GenHlth'] >= 4:
            why_high.append(f"Poor general health reported (rating {inputs['GenHlth']}/5)")
        elif inputs['GenHlth'] <= 2:
            why_low.append("Excellent/Very good general health reported")
        if age_v > 60:
            why_high.append(f"Age {int(age_v)} — risk increases significantly after 60")
        elif age_v > 45:
            why_high.append(f"Age {int(age_v)} — moderate age-related risk")
        else:
            why_low.append(f"Age {int(age_v)} — relatively lower age-related risk")

        if risk == 'HIGH RISK':
            suggestions += ["Consult Cardiologist immediately", "ECG and stress test recommended",
                            "Take prescribed medications regularly", "Follow low-sodium, low-fat diet"]
            lifestyle = {'diet': ['Mediterranean diet — olive oil, fish, nuts, vegetables',
                                  'Avoid fried food, red meat, processed snacks',
                                  'Reduce salt intake below 1500mg/day'],
                         'exercise': ['30 min cardio 5x/week', 'Avoid intense exercise until cleared by doctor'],
                         'sleep': ['7-8 hours quality sleep', 'Manage stress — high cortisol harms heart']}
        elif risk == 'MEDIUM RISK':
            suggestions += ["Annual cardiac checkup", "Monitor BP and cholesterol regularly",
                            "Maintain healthy weight"]
            lifestyle = {'diet': ['Balanced diet rich in fiber, fruits, vegetables',
                                  'Limit saturated fats and refined sugars'],
                         'exercise': ['30 min moderate exercise daily'],
                         'sleep': ['Maintain regular sleep schedule']}
        else:
            suggestions.append("Continue healthy lifestyle — annual checkup recommended")
            lifestyle = {'diet': ['Balanced nutritious diet', 'Stay hydrated (8 glasses/day)'],
                         'exercise': ['Continue regular physical activity'],
                         'sleep': ['Maintain 7-8 hours sleep']}

        feat_imp = [
            {'feature': 'High Blood Pressure', 'contribution_percent': 25, 'max_weight': 25},
            {'feature': 'High Cholesterol',    'contribution_percent': 22, 'max_weight': 25},
            {'feature': 'General Health',       'contribution_percent': 18, 'max_weight': 25},
            {'feature': 'BMI',                  'contribution_percent': 15, 'max_weight': 25},
            {'feature': 'Age',                  'contribution_percent': 10, 'max_weight': 25},
            {'feature': 'Smoking',              'contribution_percent': 6,  'max_weight': 25},
            {'feature': 'Diabetes',             'contribution_percent': 4,  'max_weight': 25},
        ]

        result = {'probability': round(prob*100, 2), 'risk': risk, 'doctor': doctor,
                  'why_high_risk': why_high, 'why_low_risk': why_low,
                  'suggestions': suggestions, 'lifestyle_changes': lifestyle,
                  'feature_importance': feat_imp}
        sid = session.get('sid', 'default')
        save_to_history(sid, 'heart', inputs, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Brain Predict ──────────────────────────────────────────
@app.route('/predict/brain', methods=['POST'])
def predict_brain():
    try:
        m = MODELS.get('brain')
        if not m: return jsonify({'error': 'Brain model not loaded.'}), 500
        data = request.json

        # ── Step 1: Raw inputs from HTML form ──────────────
        age               = float(data.get('age', 0))
        hypertension      = int(data.get('hypertension', 0))
        heart_disease     = int(data.get('heart_disease', 0))
        avg_glucose_level = float(data.get('avg_glucose_level', 0))
        bmi               = float(data.get('bmi', 0))
        ever_married      = data.get('ever_married', 'No')       # 'Yes' or 'No'
        work_type         = data.get('work_type', 'Private')     # 'Private','Self-employed','Govt_job','children'
        smoking_status    = data.get('smoking_status', 'never smoked')  # 'never smoked','formerly smoked','smokes'
        gender            = data.get('gender', 'Male')           # 'Male' or 'Female'
        residence         = data.get('Residence_type', 'Urban')  # 'Urban' or 'Rural'

        # ── Step 2: One-hot encode exactly like notebook ────
        row = {
            'age':                            age,
            'hypertension':                   hypertension,
            'heart_disease':                  heart_disease,
            'avg_glucose_level':              avg_glucose_level,
            'bmi':                            bmi,
            'ever_married_Yes':               1 if ever_married == 'Yes' else 0,
            'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
            'smoking_status_never smoked':    1 if smoking_status == 'never smoked' else 0,
            'smoking_status_smokes':          1 if smoking_status == 'smokes' else 0,
            'work_type_Private':              1 if work_type == 'Private' else 0,
            'work_type_Self-employed':        1 if work_type == 'Self-employed' else 0,
            'work_type_children':             1 if work_type == 'children' else 0,
            'Residence_type_Urban':           1 if residence == 'Urban' else 0,
            'gender_Male':                    1 if gender == 'Male' else 0,
        }

        # ── Step 3: Reindex to exact feature order from brain_features.pkl ──
        all_features = m['features']
        patient = pd.DataFrame([row]).reindex(columns=all_features, fill_value=0)

        # ── Step 4: Scale → Select → Predict (exact notebook pipeline) ──
        scaled = m['scaler'].transform(patient)
        sel    = m['selector'].transform(scaled)
        prob   = float(m['model'].predict_proba(sel)[0, 1])

        risk   = 'HIGH RISK' if prob >= 0.15 else ('MEDIUM RISK' if prob >= 0.10 else 'LOW RISK')
        doctor = DOCTOR_MAP['brain'].get(risk, 'General Physician')

        inputs_log = {
            'age': age, 'hypertension': hypertension, 'heart_disease': heart_disease,
            'avg_glucose_level': avg_glucose_level, 'bmi': bmi,
            'ever_married': ever_married, 'work_type': work_type,
            'smoking_status': smoking_status, 'gender': gender
        }
        # ── XAI Explanation ─────────────────────────────────
        why_high, why_low, suggestions, lifestyle = [], [], [], {}

        if hypertension:
            why_high.append("Hypertension — #1 modifiable stroke risk factor (3x increased risk)")
            suggestions.append("Control BP: target < 130/80 mmHg with medication + lifestyle")
        else:
            why_low.append("No hypertension — lower stroke risk")
        if heart_disease:
            why_high.append("Heart disease present — increases stroke risk via blood clots")
            suggestions.append("Regular cardiac monitoring + anticoagulation if prescribed")
        else:
            why_low.append("No heart disease — reduced embolic stroke risk")
        if avg_glucose_level >= 200:
            why_high.append(f"High glucose {avg_glucose_level:.0f} mg/dL — diabetes significantly raises stroke risk")
            suggestions.append("Control blood sugar: HbA1c < 7%, daily glucose monitoring")
        elif avg_glucose_level >= 140:
            why_high.append(f"Elevated glucose {avg_glucose_level:.0f} mg/dL — pre-diabetic range")
        else:
            why_low.append(f"Normal glucose {avg_glucose_level:.0f} mg/dL — lower metabolic risk")
        if bmi >= 30:
            why_high.append(f"Obese BMI ({bmi:.1f}) — associated with hypertension and stroke risk")
        elif bmi >= 25:
            why_high.append(f"Overweight BMI ({bmi:.1f}) — moderate risk factor")
        else:
            why_low.append(f"Healthy BMI ({bmi:.1f}) — lower stroke risk")
        if age >= 65:
            why_high.append(f"Age {int(age)} — stroke risk doubles every decade after 55")
        elif age >= 55:
            why_high.append(f"Age {int(age)} — elevated age-related risk (risk doubles after 55)")
        else:
            why_low.append(f"Age {int(age)} — relatively lower age-related stroke risk")
        if smoking_status == 'smokes':
            why_high.append("Current smoker — doubles stroke risk by damaging blood vessels")
            suggestions.append("Quit smoking — risk reduces to normal within 5 years")
        elif smoking_status == 'formerly smoked':
            why_high.append("Former smoker — residual elevated risk")
        else:
            why_low.append("Non-smoker — lower stroke risk")
        if ever_married == 'Yes' and age >= 50:
            why_low.append("Married status — associated with better health monitoring")

        if risk == 'HIGH RISK':
            suggestions += ["Consult Neurologist immediately",
                            "MRI/CT scan of brain recommended",
                            "Learn FAST signs: Face drooping, Arm weakness, Speech difficulty, Time to call 108",
                            "Daily aspirin only if prescribed by doctor"]
            lifestyle = {'diet': ['DASH diet — low sodium, high potassium',
                                  'Omega-3 rich food: fish, flaxseed, walnuts',
                                  'Avoid alcohol — raises BP and stroke risk',
                                  'Reduce red meat, processed food'],
                         'exercise': ['30 min moderate exercise daily',
                                      'Yoga reduces stress and BP effectively'],
                         'sleep': ['7-8 hours quality sleep',
                                   'Treat sleep apnea if present — major stroke risk factor']}
        elif risk == 'MEDIUM RISK':
            suggestions += ["Annual neurological checkup", "Monitor BP and glucose regularly"]
            lifestyle = {'diet': ['Heart-healthy balanced diet'],
                         'exercise': ['Regular moderate exercise'],
                         'sleep': ['Consistent sleep schedule']}
        else:
            suggestions.append("Maintain healthy lifestyle — regular health checkups")
            lifestyle = {'diet': ['Balanced nutritious diet'],
                         'exercise': ['Stay physically active'],
                         'sleep': ['7-8 hours quality sleep']}

        feat_imp = [
            {'feature': 'Age',                 'contribution_percent': 28, 'max_weight': 28},
            {'feature': 'Hypertension',        'contribution_percent': 25, 'max_weight': 28},
            {'feature': 'Avg Glucose Level',   'contribution_percent': 18, 'max_weight': 28},
            {'feature': 'BMI',                 'contribution_percent': 12, 'max_weight': 28},
            {'feature': 'Heart Disease',       'contribution_percent': 10, 'max_weight': 28},
            {'feature': 'Smoking Status',      'contribution_percent': 7,  'max_weight': 28},
        ]

        result = {'probability': round(prob*100, 2), 'risk': risk, 'doctor': doctor,
                  'why_high_risk': why_high, 'why_low_risk': why_low,
                  'suggestions': suggestions, 'lifestyle_changes': lifestyle,
                  'feature_importance': feat_imp}
        sid = session.get('sid', 'default')
        save_to_history(sid, 'brain', inputs_log, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Diabetes Predict ───────────────────────────────────────
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    try:
        m = MODELS.get('diabetes')
        if not m: return jsonify({'error': 'Diabetes model not loaded.'}), 500
        data     = request.json
        features = m['features']
        inputs   = {f: float(data.get(f, 0)) for f in features}
        patient  = pd.DataFrame([inputs])
        scaled   = m['scaler'].transform(patient)
        prob     = float(m['model'].predict_proba(scaled)[0, 1])
        risk     = 'HIGH RISK' if prob >= 0.60 else ('MEDIUM RISK' if prob >= 0.40 else 'LOW RISK')
        doctor   = DOCTOR_MAP['diabetes'].get(risk, 'General Physician')

        # ── XAI Explanation ─────────────────────────────────
        why_high, why_low, suggestions, lifestyle = [], [], [], {}
        gluc = inputs.get('Glucose',0); bmi_v = inputs.get('BMI',0)
        ins  = inputs.get('Insulin',0); age_v = inputs.get('Age',0)
        preg = inputs.get('Pregnancies',0); dpf = inputs.get('DiabetesPedigreeFunction',0)

        if gluc >= 126:
            why_high.append(f"Fasting Glucose {int(gluc)} mg/dL — diabetic range (≥ 126)")
        elif gluc >= 100:
            why_high.append(f"Glucose {int(gluc)} mg/dL — pre-diabetic range (100–125)")
        else:
            why_low.append(f"Glucose {int(gluc)} mg/dL — normal range (< 100)")

        if bmi_v >= 30:
            why_high.append(f"Obese BMI ({bmi_v}) — strongly associated with Type 2 diabetes")
            suggestions.append("Lose 5–10% body weight — reduces diabetes risk by 58%")
        elif bmi_v >= 25:
            why_high.append(f"Overweight BMI ({bmi_v}) — moderate diabetes risk")
        else:
            why_low.append(f"BMI ({bmi_v}) — healthy range, lower insulin resistance")

        if ins > 200:
            why_high.append(f"High Insulin ({int(ins)} μU/mL) — suggests insulin resistance")
        elif ins == 0:
            why_high.append("Insulin reading 0 — possible measurement issue or severe deficiency")

        if age_v >= 45:
            why_high.append(f"Age {int(age_v)} — risk increases significantly after 45")
        else:
            why_low.append(f"Age {int(age_v)} — lower age-related risk")

        if dpf > 0.5:
            why_high.append(f"Diabetes Pedigree Function {dpf:.2f} — strong family history indicator")
        else:
            why_low.append(f"Diabetes Pedigree Function {dpf:.2f} — lower genetic predisposition")

        if preg >= 4:
            why_high.append(f"{int(preg)} pregnancies — gestational diabetes history increases T2D risk")

        if risk == 'HIGH RISK':
            suggestions += ["Consult Endocrinologist immediately",
                            "HbA1c test recommended (target < 7%)",
                            "Daily blood glucose monitoring",
                            "Start low-glycemic index diet"]
            lifestyle = {'diet': ['Avoid sugar, white rice, refined carbs completely',
                                  'Eat: oats, quinoa, vegetables, legumes, lean protein',
                                  'Small meals every 3 hours to stabilize blood sugar',
                                  'Cinnamon, bitter gourd (karela) — natural glucose control'],
                         'exercise': ['45 min brisk walk daily — reduces glucose by 20–30%',
                                      'Resistance training 3x/week improves insulin sensitivity'],
                         'sleep': ['7-8 hours sleep — poor sleep worsens insulin resistance',
                                   'Manage stress — cortisol spikes raise blood sugar']}
        elif risk == 'MEDIUM RISK':
            suggestions += ["Glucose tolerance test recommended",
                            "Monitor blood sugar monthly", "Weight management program"]
            lifestyle = {'diet': ['Low-glycemic diet', 'Reduce sugar and processed foods'],
                         'exercise': ['30 min walk daily'],
                         'sleep': ['7-8 hours regular sleep']}
        else:
            suggestions.append("Healthy lifestyle maintenance — annual screening recommended")
            lifestyle = {'diet': ['Balanced diet with fiber-rich foods'],
                         'exercise': ['Regular moderate exercise'],
                         'sleep': ['Adequate 7-8 hours sleep']}

        feat_imp = [
            {'feature': 'Glucose Level',        'contribution_percent': 35, 'max_weight': 35},
            {'feature': 'BMI',                  'contribution_percent': 22, 'max_weight': 35},
            {'feature': 'Age',                  'contribution_percent': 15, 'max_weight': 35},
            {'feature': 'Insulin',              'contribution_percent': 12, 'max_weight': 35},
            {'feature': 'Diabetes Pedigree',    'contribution_percent': 10, 'max_weight': 35},
            {'feature': 'Pregnancies',          'contribution_percent': 6,  'max_weight': 35},
        ]

        result = {'probability': round(prob*100, 2), 'risk': risk, 'doctor': doctor,
                  'why_high_risk': why_high, 'why_low_risk': why_low,
                  'suggestions': suggestions, 'lifestyle_changes': lifestyle,
                  'feature_importance': feat_imp}
        sid = session.get('sid', 'default')
        save_to_history(sid, 'diabetes', inputs, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Kidney Predict ─────────────────────────────────────────
# ── Kidney Predict (EXACT MATCH with Training Code) ─────────────────────────
@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        m = MODELS.get('kidney')
        if not m: return jsonify({'error': 'Kidney model not loaded.'}), 500
        
        data = request.json
        # ટ્રેનિંગ ફાઈલ મુજબના Target Names
        reverse_map = {v: k for k, v in m['target_map'].items()}
        
        # 1. મોડેલ જે ફીચર્સ માંગે છે તેની લિસ્ટ લો (exact order)
        model_features = m['features'] 
        
        # 2. HTML માંથી ડેટા લો. જો નામ મેચ ન થાય તો np.nan મૂકો
        inputs_dict = {}
        for feat in model_features:
            val = data.get(feat)
            if val is not None:
                try:
                    inputs_dict[feat] = float(val)
                except:
                    inputs_dict[feat] = np.nan
            else:
                inputs_dict[feat] = np.nan

        # 3. DataFrame બનાવો (Exact Order)
        patient = pd.DataFrame([inputs_dict]).reindex(columns=model_features)
        
        # 4. Missing values ને Mean થી ભરો (જેથી મોડેલ કન્ફ્યુઝ ન થાય)
        # આ સ્ટેપ ખૂબ જરૂરી છે કારણ કે HTML માં કદાચ 1-2 ફિલ્ડ્સ મિસિંગ હોય
        patient = patient.fillna(patient.mean().fillna(0))

        # 5. Prediction
        # NOTE:
        # The saved kidney ML pipeline can produce clinically inverted outputs for
        # extreme rows (for example, very high creatinine + low eGFR may be pushed
        # toward No_Disease). To keep patient-facing predictions safe and sensible,
        # we use the ML output as supporting information and apply a clinical
        # severity correction based on standard kidney markers.
        pred_raw        = int(m['model'].predict(patient)[0])
        proba_raw       = m['model'].predict_proba(patient)[0]
        model_label     = reverse_map.get(pred_raw, 'Low_Risk')
        model_conf      = round(float(max(proba_raw))*100, 2)
        model_all_p     = {reverse_map[i]: round(float(p)*100, 2) for i, p in enumerate(proba_raw)}

        def _num(name, default=0):
            try:
                return float(data.get(name, default) or default)
            except Exception:
                return float(default)

        sc_rule       = _num('Serum creatinine (mg/dl)')
        egfr_rule     = _num('Estimated Glomerular Filtration Rate (eGFR)', 90)
        hemo_rule     = _num('Hemoglobin level (gms)', 14)
        bp_rule       = _num('Blood pressure (mm/Hg)', 120)
        albumin_rule  = _num('Albumin in urine')
        sugar_rule    = _num('Sugar in urine')
        glucose_rule  = _num('Random blood glucose level (mg/dl)', 100)
        urea_rule     = _num('Blood urea (mg/dl)', 25)
        upcr_rule     = _num('Urine protein-to-creatinine ratio')
        urine_rule    = _num('Urine output (ml/day)', 1500)
        potassium_rule= _num('Potassium level (mEq/L)', 4.5)
        edema_rule    = _num('Pedal edema (yes/no)')
        anemia_rule   = _num('Anemia (yes/no)')
        htn_rule      = _num('Hypertension (yes/no)')
        dm_rule       = _num('Diabetes mellitus (yes/no)')

        severity_points = 0
        clinical_flags = []

        def _flag(points, text):
            nonlocal severity_points
            severity_points += points
            clinical_flags.append(text)

        if egfr_rule < 15:
            _flag(55, f"eGFR is critically low ({egfr_rule})")
        elif egfr_rule < 30:
            _flag(42, f"eGFR is severely reduced ({egfr_rule})")
        elif egfr_rule < 60:
            _flag(28, f"eGFR is below normal ({egfr_rule})")
        elif egfr_rule < 90:
            _flag(10, f"eGFR is mildly reduced ({egfr_rule})")

        if sc_rule >= 5:
            _flag(45, f"Serum creatinine is very high ({sc_rule} mg/dL)")
        elif sc_rule >= 3:
            _flag(35, f"Serum creatinine is high ({sc_rule} mg/dL)")
        elif sc_rule >= 1.5:
            _flag(22, f"Serum creatinine is elevated ({sc_rule} mg/dL)")
        elif sc_rule > 1.2:
            _flag(10, f"Serum creatinine is slightly above normal ({sc_rule} mg/dL)")

        if albumin_rule >= 4:
            _flag(25, f"Albumin in urine is high grade ({albumin_rule})")
        elif albumin_rule >= 2:
            _flag(16, f"Albumin in urine is elevated ({albumin_rule})")
        elif albumin_rule >= 1:
            _flag(8, f"Trace albumin is present in urine ({albumin_rule})")

        if upcr_rule >= 3:
            _flag(25, f"Urine protein-to-creatinine ratio is very high ({upcr_rule})")
        elif upcr_rule >= 0.5:
            _flag(14, f"Urine protein-to-creatinine ratio is elevated ({upcr_rule})")

        if urea_rule >= 80:
            _flag(18, f"Blood urea is high ({urea_rule} mg/dL)")
        elif urea_rule >= 50:
            _flag(10, f"Blood urea is elevated ({urea_rule} mg/dL)")

        if hemo_rule < 9:
            _flag(16, f"Hemoglobin is very low ({hemo_rule} g/dL)")
        elif hemo_rule < 12:
            _flag(9, f"Hemoglobin is low ({hemo_rule} g/dL)")

        if bp_rule >= 160:
            _flag(16, f"Blood pressure is very high ({bp_rule} mmHg)")
        elif bp_rule >= 140:
            _flag(10, f"Blood pressure is high ({bp_rule} mmHg)")

        if glucose_rule >= 200 or sugar_rule >= 2:
            _flag(10, "High glucose/sugar markers are present")
        if potassium_rule >= 5.5:
            _flag(12, f"Potassium is high ({potassium_rule} mEq/L)")
        if urine_rule and urine_rule < 500:
            _flag(18, f"Urine output is low ({urine_rule} ml/day)")
        if edema_rule:
            _flag(8, "Pedal edema is present")
        if anemia_rule:
            _flag(6, "Anemia is marked as present")
        if htn_rule:
            _flag(6, "Hypertension history is present")
        if dm_rule:
            _flag(6, "Diabetes history is present")

        if severity_points >= 80 or egfr_rule < 15 or sc_rule >= 5:
            label = 'Severe_Disease'
        elif severity_points >= 55 or egfr_rule < 30 or sc_rule >= 3:
            label = 'High_Risk'
        elif severity_points >= 32 or egfr_rule < 60 or sc_rule >= 1.5 or albumin_rule >= 2:
            label = 'Moderate_Risk'
        elif severity_points >= 12 or egfr_rule < 90 or sc_rule > 1.2 or albumin_rule >= 1:
            label = 'Low_Risk'
        else:
            label = 'No_Disease'

        severity_rank = {'No_Disease': 0, 'Low_Risk': 1, 'Moderate_Risk': 2, 'High_Risk': 3, 'Severe_Disease': 4}
        conf = round(min(98, max(55, 52 + severity_points * 0.55)), 2)

        # Build patient-facing probabilities from corrected clinical severity.
        classes = ['No_Disease', 'Low_Risk', 'Moderate_Risk', 'High_Risk', 'Severe_Disease']
        rank = severity_rank[label]
        other_weights = {}
        for c in classes:
            if c == label:
                continue
            dist = abs(severity_rank[c] - rank)
            other_weights[c] = max(1.0, 10.0 - dist * 2.25)
        remaining = max(0.0, 100.0 - conf)
        weight_total = sum(other_weights.values()) or 1.0
        all_p = {c: round((other_weights.get(c, 0.0) / weight_total) * remaining, 2) for c in classes}
        all_p[label] = round(conf, 2)

        doctor      = DOCTOR_MAP['kidney'].get(label, 'General Physician')

        # ── XAI Explanation ─────────────────────────────────
        # અહીં આપણે HTML ID નો ઉપયોગ કરીને ચેક કરીશું
        why_high, why_low, suggestions, lifestyle = [], [], [], {}
        
        # સાચા નામ સાથે વેલ્યુઝ મેળવો
        sc   = data.get('Serum creatinine (mg/dl)', 0)
        egfr = data.get('Estimated Glomerular Filtration Rate (eGFR)', 0)
        hemo = data.get('Hemoglobin level (gms)', 0)
        bp_v = data.get('Blood pressure (mm/Hg)', 0)
        al   = data.get('Albumin in urine', 0)
        su   = data.get('Sugar in urine', 0)

        # Logic based on medical thresholds
        if float(sc or 0) > 1.2:
            why_high.append(f"High Serum Creatinine ({sc} mg/dL) — indicates reduced kidney filtration")
            suggestions.append("Repeat creatinine + eGFR test — monitor kidney function monthly")
        else:
            why_low.append(f"Serum Creatinine ({sc}) in normal range — good kidney filtration")
            
        if float(hemo or 0) < 12:
            why_high.append(f"Low Hemoglobin ({hemo} g/dL) — anemia common in kidney disease")
            suggestions.append("Check for renal anemia — may need erythropoietin therapy")
        else:
            why_low.append(f"Hemoglobin ({hemo} g/dL) — adequate, less anemia risk")
            
        if float(bp_v or 0) >= 140:
            why_high.append(f"High BP ({bp_v} mmHg) — damages kidney blood vessels")
            suggestions.append("Strict BP control < 130/80 — ACE inhibitors preferred for CKD")
        else:
            why_low.append(f"Blood pressure ({bp_v}) within acceptable range")
            
        if float(al or 0) >= 3:
            why_high.append(f"Albumin in urine (grade {al}) — kidney protein leakage sign")
        elif float(al or 0) == 0:
            why_low.append("No albumin in urine — healthy glomerular filtration")
            
        if float(su or 0) >= 2:
            why_high.append(f"Sugar in urine (grade {su}) — diabetic nephropathy indicator")

        # Stage-wise suggestions
        if label in ['Severe_Disease', 'High_Risk']:
            suggestions += ["Consult Nephrologist urgently", "24-hour urine protein test recommended",
                            "Strict fluid and protein restriction", "Avoid NSAIDs (ibuprofen) — nephrotoxic"]
            lifestyle = {'diet': ['Low protein diet', 'Low potassium/sodium', 'Limit fluid intake'],
                         'exercise': ['Light walking only', 'Gentle yoga'],
                         'sleep': ['8 hours sleep', 'Elevate legs to reduce swelling']}
        elif label in ['Moderate_Risk']:
            suggestions += ["Nephrology referral recommended", "Kidney function test every 3 months"]
            lifestyle = {'diet': ['Moderate protein restriction', 'Low salt diet'],
                         'exercise': ['30 min moderate walk daily'],
                         'sleep': ['Regular 7-8 hours sleep']}
        else:
            suggestions.append("Annual kidney function test — maintain hydration")
            lifestyle = {'diet': ['Drink 2-3 liters water daily', 'Balanced protein intake'],
                         'exercise': ['Regular physical activity'],
                         'sleep': ['7-8 hours sleep']}

        feat_imp = [
            {'feature': 'Serum Creatinine',  'contribution_percent': 30, 'max_weight': 30},
            {'feature': 'Hemoglobin',        'contribution_percent': 20, 'max_weight': 30},
            {'feature': 'Blood Pressure',    'contribution_percent': 18, 'max_weight': 30},
            {'feature': 'Albumin in Urine',  'contribution_percent': 15, 'max_weight': 30},
            {'feature': 'Blood Glucose',     'contribution_percent': 10, 'max_weight': 30},
            {'feature': 'Sugar in Urine',    'contribution_percent': 7, 'max_weight': 30},
        ]

        result = {'prediction': label, 'confidence': conf, 'probabilities': all_p,
                  'doctor': doctor, 'why_high_risk': why_high, 'why_low_risk': why_low,
                  'suggestions': suggestions, 'lifestyle_changes': lifestyle,
                  'feature_importance': feat_imp,
                  'clinical_flags': clinical_flags,
                  'model_prediction_raw': model_label,
                  'model_confidence_raw': model_conf,
                  'model_probabilities_raw': model_all_p}
        
        sid = session.get('sid', 'default')
        save_to_history(sid, 'kidney', inputs_dict, result)
        return jsonify(result)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


# ── Eye Predict ────────────────────────────────────────────
@app.route('/predict/eye', methods=['POST'])
def predict_eye():
    try:
        m = MODELS.get('eye')
        if not m: return jsonify({'error': 'Eye model not loaded.'}), 500
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        import tempfile, tensorflow as tf
        file = request.files['file']
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            file.save(tmp.name)
            img   = tf.keras.preprocessing.image.load_img(tmp.name, target_size=(150,150))
            arr   = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arr   = np.expand_dims(arr, 0)
            proba = m['model'].predict(arr, verbose=0)[0]
            pred  = int(np.argmax(proba))
            label = m['reverse'].get(pred, 'Unknown')
            conf  = round(float(max(proba))*100, 2)
            all_p = {m['reverse'].get(i,'?'): round(float(p)*100,2) for i,p in enumerate(proba)}
            doctor = DOCTOR_MAP['eye'].get(label, 'Ophthalmologist')
            os.unlink(tmp.name)

        result = {'prediction': label, 'confidence': conf, 'probabilities': all_p, 'doctor': doctor}
        sid = session.get('sid', 'default')
        save_to_history(sid, 'eye', {}, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/predict/lung', methods=['POST'])
def predict_lung():
    try:
        m = MODELS.get('lung')
        if not m: return jsonify({'error': 'Lung model not loaded.'}), 500

        data = request.json
        meta = m['meta']

        # Encode inputs - support both naming conventions from HTML form
        gender_raw    = data.get('gender', data.get('Gender', 'Male'))
        stage_raw     = data.get('cancer_stage', data.get('CancerStage', 0))
        family_raw    = data.get('family_history', data.get('FamilyHistory', 0))
        smoking_raw   = data.get('smoking_status', data.get('SmokingStatus', 0))
        treatment_raw = data.get('treatment_type', data.get('TreatmentType', 0))

        # Handle both string and integer inputs
        def safe_encode(le, val, fallback=0):
            try:
                if isinstance(val, (int, float)):
                    # Already numeric — check if it's a valid index
                    if int(val) < len(le.classes_):
                        return int(val)
                    return fallback
                return int(le.transform([str(val)])[0])
            except:
                return fallback

        gender    = safe_encode(m['le_gender'],  gender_raw,    0)
        stage     = safe_encode(m['le_stage'],   stage_raw,     0)
        family    = safe_encode(m['le_family'],  family_raw,    0)
        smoking   = safe_encode(m['le_smoking'], smoking_raw,   0)
        treatment = safe_encode(m['le_treat'],   treatment_raw, 0)

        age_val  = float(data.get('age', data.get('Age', 50)))
        bmi_val  = float(data.get('bmi', data.get('BMI', 25)))
        chol_val = int(data.get('cholesterol_level', data.get('CholesterolLevel', 200)))

        inputs = {
            'age':               age_val,
            'gender':            int(gender),
            'cancer_stage':      int(stage),
            'family_history':    int(family),
            'smoking_status':    int(smoking),
            'bmi':               bmi_val,
            'cholesterol_level': chol_val,
            'hypertension':      int(data.get('hypertension', data.get('Hypertension', 0))),
            'asthma':            int(data.get('asthma', data.get('Asthma', 0))),
            'cirrhosis':         int(data.get('cirrhosis', data.get('Cirrhosis', 0))),
            'other_cancer':      int(data.get('other_cancer', data.get('OtherCancer', 0))),
            'treatment_type':    int(treatment),
        }

        # Use features from metadata
        features = meta.get('features', list(inputs.keys()))
        patient  = pd.DataFrame([inputs])[features[:12]]  # first 12 base features
        prob     = float(m['model'].predict_proba(patient)[0, 1])

        # Clinical risk: factor in stage and smoking for better UX
        stage_val   = int(stage)
        smoking_val = int(smoking)
        # Stage IV or current smoker → push risk higher
        adj = 0.0
        if stage_val == 3: adj += 0.15   # Stage IV
        elif stage_val == 2: adj += 0.08 # Stage III
        if smoking_val == 0: adj += 0.10  # Current Smoker
        if age_val > 65:     adj += 0.05
        if bmi_val > 30:     adj += 0.03
        prob_adj = min(0.95, prob + adj)

        if prob_adj >= 0.55:
            risk = 'HIGH RISK'
        elif prob_adj >= 0.35:
            risk = 'MEDIUM RISK'
        else:
            risk = 'LOW RISK'

        doctor = DOCTOR_MAP.get('lung', {}).get(risk, 'Oncologist')
        # ── XAI Explanation ─────────────────────────────────
        why_high, why_low, suggestions, lifestyle = [], [], [], {}
        stage_labels = {0:'Stage I', 1:'Stage II', 2:'Stage III', 3:'Stage IV'}
        stage_name   = stage_labels.get(int(stage), 'Unknown')
        smoking_labels = {0:'Current Smoker', 1:'Former Smoker', 2:'Never Smoked', 3:'Passive Smoker'}
        smoking_name   = smoking_labels.get(int(smoking), 'Unknown')

        # Stage
        if int(stage) >= 3:
            why_high.append(f"Cancer Stage IV — metastatic spread to distant organs, poorest prognosis")
        elif int(stage) == 2:
            why_high.append(f"Cancer Stage III — locally advanced, requires aggressive treatment")
        elif int(stage) == 1:
            why_high.append(f"Cancer Stage II — localized spread, treatment can be effective")
        else:
            why_low.append("Cancer Stage I — early detection, highest survival rates (70–90%)")

        # Smoking
        if int(smoking) == 0:
            why_high.append("Current smoker — smoking causes 85% of lung cancers, worsens prognosis")
            suggestions.append("Quit smoking immediately — improves treatment response")
        elif int(smoking) == 1:
            why_high.append("Former smoker — residual lung damage affects recovery")
        elif int(smoking) == 3:
            why_high.append("Passive smoker — second-hand smoke exposure increases risk")
        else:
            why_low.append("Never smoked — better baseline lung function and prognosis")

        # Age
        if age_val > 70:
            why_high.append(f"Age {int(age_val)} — older patients have reduced treatment tolerance")
        elif age_val > 60:
            why_high.append(f"Age {int(age_val)} — moderate age-related treatment challenges")
        else:
            why_low.append(f"Age {int(age_val)} — better physiological reserve for treatment")

        # Comorbidities
        if inputs.get('cirrhosis',0):
            why_high.append("Cirrhosis — liver disease limits chemotherapy options")
        if inputs.get('hypertension',0):
            why_high.append("Hypertension — cardiovascular comorbidity affects surgical risk")
        if inputs.get('other_cancer',0):
            why_high.append("History of other cancer — multiple primary tumors worsen prognosis")
        if inputs.get('asthma',0):
            why_high.append("Asthma — compromised lung function affects treatment tolerance")

        if not any([inputs.get('cirrhosis',0), inputs.get('hypertension',0),
                    inputs.get('other_cancer',0), inputs.get('asthma',0)]):
            why_low.append("No major comorbidities — better treatment tolerance")

        # Family history
        if int(family) == 1:
            why_high.append("Family history of cancer — genetic predisposition to aggressive disease")
        else:
            why_low.append("No family history — lower genetic risk factor")

        if risk == 'HIGH RISK':
            suggestions += ["Consult Oncologist immediately — treatment urgency is critical",
                            "PET-CT scan for metastasis assessment",
                            "Multidisciplinary tumor board review recommended",
                            "Consider clinical trials if standard therapy fails",
                            "Palliative care consultation for quality of life"]
            lifestyle = {'diet': ['High protein, high calorie diet to prevent cancer cachexia',
                                  'Anti-inflammatory foods: turmeric, berries, green tea',
                                  'Avoid alcohol — interferes with chemotherapy',
                                  'Small frequent meals to manage nausea from treatment'],
                         'exercise': ['Gentle walking as tolerated',
                                      'Breathing exercises — improves lung capacity',
                                      'Avoid strenuous activity during chemo/radiation'],
                         'sleep': ['Rest is critical during treatment',
                                   'Pain management for better sleep quality']}
        elif risk == 'MEDIUM RISK':
            suggestions += ["Oncologist consultation for treatment plan",
                            "Regular CT scan follow-up every 3 months",
                            "Pulmonary function tests before surgery"]
            lifestyle = {'diet': ['Nutritious balanced diet rich in antioxidants',
                                  'Adequate protein for recovery'],
                         'exercise': ['Moderate walking daily', 'Breathing exercises'],
                         'sleep': ['7-8 hours quality sleep']}
        else:
            suggestions += ["Continue treatment plan as prescribed",
                            "Annual CT scan monitoring",
                            "Pulmonary rehabilitation program"]
            lifestyle = {'diet': ['Healthy balanced diet', 'Antioxidant-rich fruits and vegetables'],
                         'exercise': ['Regular moderate exercise', 'Breathing exercises daily'],
                         'sleep': ['7-8 hours sleep', 'Stress management']}

        feat_imp = [
            {'feature': 'Cancer Stage',      'contribution_percent': 35, 'max_weight': 35},
            {'feature': 'Age',               'contribution_percent': 20, 'max_weight': 35},
            {'feature': 'Smoking Status',    'contribution_percent': 18, 'max_weight': 35},
            {'feature': 'Treatment Type',    'contribution_percent': 12, 'max_weight': 35},
            {'feature': 'BMI',               'contribution_percent': 8,  'max_weight': 35},
            {'feature': 'Comorbidities',     'contribution_percent': 7,  'max_weight': 35},
        ]

        result = {
            'probability': round(prob_adj * 100, 2),
            'survival_probability': round(prob_adj * 100, 2),
            'risk':   risk,
            'doctor': doctor,
            'why_high_risk': why_high, 'why_low_risk': why_low,
            'suggestions': suggestions, 'lifestyle_changes': lifestyle,
            'feature_importance': feat_imp
        }

        save_to_history(session.get('sid', 'default'), 'lung', inputs, result)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    




# ── History API ────────────────────────────────────────────
@app.route('/api/history', methods=['GET'])
def get_history():
    user = session.get('user')
    if not user or not user.get('email'):
        return jsonify([])
    return jsonify(db_get_history(user['email']))

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    user = session.get('user')
    if user and user.get('email'):
        db_clear_history(user['email'])
    return jsonify({'status': 'cleared'})

@app.route('/api/history/delete/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    user = session.get('user')
    if user and user.get('email'):
        db_delete_prediction(user['email'], record_id)
    return jsonify({'status': 'deleted'})

# ── Trend API (last N predictions for a disease) ──────────
@app.route('/api/trend/<disease>', methods=['GET'])
def get_trend(disease):
    user = session.get('user')
    if not user or not user.get('email'):
        return jsonify([])
    all_records = db_get_history(user['email'], limit=100)
    trend = [r for r in all_records if r['disease'] == disease][-20:]
    return jsonify(trend)

# ── PDF Report API ─────────────────────────────────────────
@app.route('/api/report', methods=['POST'])
def generate_report():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        import io
        from flask import send_file

        data     = request.json
        disease  = data.get('disease', 'Unknown')
        result   = data.get('result', {})
        inputs   = data.get('inputs', {})
        pat_name = data.get('patient_name', 'Patient')
        pat_age  = data.get('patient_age', 'N/A')

        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   rightMargin=2*cm, leftMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # Title
        title_style = ParagraphStyle('title', parent=styles['Title'],
                                     fontSize=22, textColor=colors.HexColor('#1a237e'),
                                     spaceAfter=6)
        story.append(Paragraph('VitalsAI — Health Risk Report', title_style))
        story.append(Paragraph(f'Generated: {datetime.now().strftime("%d %b %Y, %H:%M")}',
                                styles['Normal']))
        story.append(Spacer(1, 0.5*cm))

        # Patient Info
        story.append(Paragraph('Patient Information', styles['Heading2']))
        pt = [['Name', pat_name], ['Age', str(pat_age)],
              ['Disease Module', disease.upper()],
              ['Report Date', datetime.now().strftime('%d %b %Y')]]
        t = Table(pt, colWidths=[5*cm, 10*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e8eaf6')),
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING',    (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.5*cm))

        # Result
        story.append(Paragraph('Prediction Result', styles['Heading2']))
        risk_val = result.get('risk') or result.get('prediction', 'N/A')
        prob_val = result.get('probability') or result.get('confidence', 'N/A')
        risk_color = colors.red if 'HIGH' in str(risk_val) or 'Severe' in str(risk_val) \
                     else (colors.orange if 'MEDIUM' in str(risk_val) or 'Moderate' in str(risk_val) \
                     else colors.green)

        rt = [['Risk Level', risk_val], ['Probability/Confidence', f'{prob_val}%'],
              ['Recommended Doctor', result.get('doctor', 'General Physician')]]
        t2 = Table(rt, colWidths=[5*cm, 10*cm])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#e8eaf6')),
            ('TEXTCOLOR',  (1,0), (1,0), risk_color),
            ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
            ('PADDING',    (0,0), (-1,-1), 6),
        ]))
        story.append(t2)
        story.append(Spacer(1, 0.5*cm))

        # Input Values
        if inputs:
            story.append(Paragraph('Clinical Input Values', styles['Heading2']))
            rows = [[str(k), str(v)] for k, v in inputs.items()]
            t3   = Table([['Parameter', 'Value']] + rows, colWidths=[8*cm, 7*cm])
            t3.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a237e')),
                ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f5f5f5')),
                ('FONTNAME',   (0,0), (-1,-1), 'Helvetica'),
                ('FONTSIZE',   (0,0), (-1,-1), 9),
                ('GRID',       (0,0), (-1,-1), 0.5, colors.grey),
                ('PADDING',    (0,0), (-1,-1), 5),
            ]))
            story.append(t3)
            story.append(Spacer(1, 0.5*cm))

        # Disclaimer
        disc = ParagraphStyle('disc', parent=styles['Normal'],
                              fontSize=8, textColor=colors.grey)
        story.append(Paragraph(
            '⚠️ Disclaimer: This report is generated by an AI model for informational '
            'purposes only. It is NOT a substitute for professional medical advice, '
            'diagnosis, or treatment. Always consult a qualified healthcare provider.',
            disc))

        doc.build(story)
        buf.seek(0)
        return send_file(buf, as_attachment=True,
                         download_name=f'VitalsAI_{disease}_report.pdf',
                         mimetype='application/pdf')
    except ImportError:
        return jsonify({'error': 'reportlab not installed. Run: pip install reportlab'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Language API ───────────────────────────────────────────
TRANSLATIONS = {
    'en': {
        'high_risk': 'HIGH RISK', 'medium_risk': 'MEDIUM RISK', 'low_risk': 'LOW RISK',
        'probability': 'Probability', 'doctor': 'Recommended Doctor',
        'history': 'History', 'report': 'Download Report', 'bmi': 'BMI Calculator'
    },
    'gu': {
        'high_risk': 'ઉચ્ચ જોખમ', 'medium_risk': 'મધ્યમ જોખમ', 'low_risk': 'ઓછું જોખમ',
        'probability': 'સંભાવના', 'doctor': 'ભલામણ કરેલ ડૉક્ટર',
        'history': 'ઇતિહાસ', 'report': 'રિપોર્ટ ડાઉનલોડ', 'bmi': 'BMI કેલ્ક્યુલેટર'
    },
    'hi': {
        'high_risk': 'उच्च जोखिम', 'medium_risk': 'मध्यम जोखिम', 'low_risk': 'कम जोखिम',
        'probability': 'संभावना', 'doctor': 'अनुशंसित डॉक्टर',
        'history': 'इतिहास', 'report': 'रिपोर्ट डाउनलोड', 'bmi': 'BMI कैलकुलेटर'
    }
}

@app.route('/api/translations/<lang>', methods=['GET'])
def get_translations(lang):
    return jsonify(TRANSLATIONS.get(lang, TRANSLATIONS['en']))

# ── Status ─────────────────────────────────────────────────
@app.route('/status')
def status():
    return jsonify({
        'loaded':  list(MODELS.keys()),
        'missing': [d for d in ['heart','brain','diabetes','kidney','eye','lung'] if d not in MODELS],
        'features': {
            'auth':          True,
            'history':       True,
            'pdf_report':    True,
            'bmi_calc':      True,
            'dark_mode':     True,
            'multilang':     True,
            'doctor_rec':    True,
            'trend_graph':   True,
            'ai_chatbot':    True,
        }
    })

# ── XAI Helper ─────────────────────────────────────────────
def build_explanation(features, values, shap_values):
    explanation = []
    for feat, val, sv in zip(features, values, shap_values):
        explanation.append({
            'feature':    feat,
            'value':      round(float(val), 3),
            'shap_value': round(float(sv), 4),
            'impact':     'increases risk' if sv > 0 else 'decreases risk',
            'importance': abs(float(sv))
        })
    explanation.sort(key=lambda x: x['importance'], reverse=True)
    return explanation
 
def get_shap_values(rf_model, X):
    try:
        import shap
        import numpy as np
        
        # 1. ડેટાને Numpy Array માં કન્વર્ટ કરવો (SASH-XAI માટે સૌથી મહત્વનું)
        # જો X DataFrame હોય તો તેની વેલ્યુઝ લો, નહીતર જેવું છે તેવું રાખો
        X_values = X.values if hasattr(X, 'values') else np.array(X)
        
        # 2. TreeExplainer બનાવો
        explainer = shap.TreeExplainer(rf_model)
        
        # 3. SHAP વેલ્યુઝ કેલ્ક્યુલેટ કરો
        shap_vals = explainer.shap_values(X_values)
        
        # 4. આઉટપુટ ફોર્મેટ હેન્ડલ કરવું (SHAP ના અલગ અલગ વર્ઝન માટે)
        if isinstance(shap_vals, list):
            # Binary classification માં લિસ્ટમાં બે એરે હોય છે [Class 0, Class 1]
            # આપણે Class 1 (Disease) ની વેલ્યુઝ જોઈએ છે
            vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            return vals[0] if len(vals.shape) > 1 else vals
            
        elif isinstance(shap_vals, np.ndarray):
            # જો આઉટપુટ (samples, features, classes) હોય
            if len(shap_vals.shape) == 3:
                return shap_vals[0, :, 1] # પહેલી સેમ્પલ, બધા ફીચર્સ, ક્લાસ 1
            # જો આઉટપુટ (samples, features) હોય
            return shap_vals[0]
            
        else:
            # જો SHAP Explanation ઓબ્જેક્ટ રિટર્ન કરે
            if hasattr(shap_vals, 'values'):
                return shap_vals.values[0]
            return shap_vals[0]

    except Exception as e:
        import traceback
        print("\n--- 🔴 DETAILED SHAP ERROR ---")
        print(traceback.format_exc()) 
        print("-----------------------------\n")
        return None

 
# ── XAI — Heart ────────────────────────────────────────────
@app.route('/api/explain/heart', methods=['POST'])
def explain_heart():
    try:
        m = MODELS.get('heart')
        if not m: return jsonify({'error': 'Heart model not loaded'}), 500
        
        data = request.json
        bp   = float(data.get('BloodPressure', 120))
        cho  = float(data.get('Cholesterol', 180))
        
        # EXACT FEATURE ORDER (જે મોડેલ ટ્રેનિંગ વખતે હતો)
        inputs = {
            'Age':          float(data.get('Age', 0)),
            'BMI':          float(data.get('BMI', 0)),
            'HighBP':       1 if bp >= 140 else 0,
            'HighChol':     1 if cho >= 240 else 0,
            'Diabetes':     int(data.get('Diabetes', 0)),
            'Smoker':       int(data.get('Smoker', 0)),
            'PhysActivity': int(data.get('PhysActivity', 1)),
            'GenHlth':      int(data.get('GenHlth', 3)),
            'Sex':          int(data.get('Sex', 0)),
        }
        
        features = m['features']
        # DataFrame બનાવો અને ખાતરી કરો કે કોલમનો ક્રમ સાચો છે
        patient = pd.DataFrame([inputs])[features] 
        
        # Stacking મોડેલમાંથી RF મોડેલ કાઢો
        try:
            rf_model = m['model'].named_estimators_['rf']
        except KeyError:
            # જો 'rf' નામ ન મળે, તો પહેલું ઉપલબ્ધ મોડેલ લો
            rf_model = list(m['model'].named_estimators_.values())[0]
        
        # SHAP વેલ્યુઝ મેળવો
        sv = get_shap_values(rf_model, patient)
        
        if sv is None: 
            return jsonify({'error': 'SHAP calculation failed. Check terminal logs.'}), 500
            
        # Explanation લિસ્ટ બનાવો
        explanation = build_explanation(features, patient.values[0], sv)
        return jsonify({'explanation': explanation, 'top_factors': explanation[:3]})
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

 
# ── XAI — Brain ────────────────────────────────────────────
@app.route('/api/explain/brain', methods=['POST'])
def explain_brain():
    try:
        m = MODELS.get('brain')
        if not m: return jsonify({'error': 'Brain model not loaded'}), 500
        data = request.json
        row = {
            'age':                            float(data.get('age', 0)),
            'hypertension':                   int(data.get('hypertension', 0)),
            'heart_disease':                  int(data.get('heart_disease', 0)),
            'avg_glucose_level':              float(data.get('avg_glucose_level', 0)),
            'bmi':                            float(data.get('bmi', 0)),
            'gender_Male':                    1 if data.get('gender') == 'Male' else 0,
            'smoking_status_formerly smoked': 1 if data.get('smoking_status') == 'formerly smoked' else 0,
            'smoking_status_never smoked':    1 if data.get('smoking_status') == 'never smoked' else 0,
            'smoking_status_smokes':          1 if data.get('smoking_status') == 'smokes' else 0,
            'work_type_Private':              1,
            'work_type_Self-employed':        0,
            'work_type_children':             0,
            'ever_married_Yes':               1,
            'Residence_type_Urban':           1,
        }
        features = m['features']
        patient  = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
        scaled   = m['scaler'].transform(patient)
        sel      = m['selector'].transform(scaled)
        sel_feats = [features[i] for i in m['selector'].get_support(indices=True)]
        rf_model  = m['model'].named_estimators_['rf']
        sv = get_shap_values(rf_model, sel)
        if sv is None: return jsonify({'error': 'SHAP failed'}), 500
        explanation = build_explanation(sel_feats, sel[0], sv)
        return jsonify({'explanation': explanation, 'top_factors': explanation[:3]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
# ── XAI — Diabetes ─────────────────────────────────────────
@app.route('/api/explain/diabetes', methods=['POST'])
def explain_diabetes():
    try:
        m = MODELS.get('diabetes')
        if not m: return jsonify({'error': 'Diabetes model not loaded'}), 500
        data     = request.json
        features = m['features']
        inputs   = {f: float(data.get(f, 0)) for f in features}
        patient  = pd.DataFrame([inputs])
        scaled   = m['scaler'].transform(patient)
        rf_model = m['model'].named_estimators_['rf']
        sv = get_shap_values(rf_model, scaled)
        if sv is None: return jsonify({'error': 'SHAP failed'}), 500
        explanation = build_explanation(features, patient.values[0], sv)
        return jsonify({'explanation': explanation, 'top_factors': explanation[:3]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
# ── XAI — Kidney ───────────────────────────────────────────
@app.route('/api/explain/kidney', methods=['POST'])
def explain_kidney():
    try:
        m = MODELS.get('kidney')
        if not m: return jsonify({'error': 'Kidney model not loaded'}), 500
        data    = request.json
        inputs  = {f: float(data.get(f, 0)) for f in m['features']}
        patient = pd.DataFrame([inputs]).reindex(columns=m['features'], fill_value=0)
        try:
            preprocessed = m['model'].named_steps['preprocessor'].transform(patient)
            rf_model     = m['model'].named_steps['model'].named_estimators_['rf']
        except:
            preprocessed = patient.values
            rf_model     = m['model'].named_estimators_['rf']
        pred_class = int(m['model'].predict(patient)[0])
        import shap
        explainer  = shap.TreeExplainer(rf_model)
        shap_vals  = explainer.shap_values(preprocessed)
        if isinstance(shap_vals, list):
            sv = shap_vals[pred_class][0]
        else:
            sv = shap_vals[0]
        explanation = build_explanation(m['features'], patient.values[0], sv)
        return jsonify({'explanation': explanation, 'top_factors': explanation[:3]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
# ── XAI — Eye (Confidence scores as explanation) ───────────
@app.route('/api/explain/eye', methods=['POST'])
def explain_eye():
    try:
        import tempfile, tensorflow as tf
        m = MODELS.get('eye')
        if not m: return jsonify({'error': 'Eye model not loaded'}), 500
        if 'file' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        file = request.files['file']
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            file.save(tmp.name)
            img   = tf.keras.preprocessing.image.load_img(tmp.name, target_size=(150,150))
            arr   = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            arr   = np.expand_dims(arr, 0)
            proba = m['model'].predict(arr, verbose=0)[0]
            pred  = int(np.argmax(proba))
            label = m['reverse'].get(pred, 'Unknown')
            os.unlink(tmp.name)
        all_p   = {m['reverse'].get(i,'?'): round(float(p)*100,2) for i,p in enumerate(proba)}
        sorted_p = sorted(all_p.items(), key=lambda x: x[1], reverse=True)
        explanation = [{'feature': cls, 'value': pct,
                        'shap_value': pct/100,
                        'impact': 'detected' if cls == label else 'not detected',
                        'importance': pct/100} for cls, pct in sorted_p]
        return jsonify({'explanation': explanation, 'top_factors': explanation[:3],
                        'note': 'CNN confidence scores per class'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



# SHAP Explainability Route — Fixed (redirects to disease-specific routes)
# UPDATED CODE (Add this to app.py)
@app.route('/api/explain-shap', methods=['POST'])
def explain_shap():
    try:
        data = request.json
        if not data or 'inputs' not in data:
            return jsonify({'error': 'No input data received'}), 400

        disease = data.get('disease')
        inputs = data.get('inputs')
        if not inputs or len(inputs) == 0:
            return jsonify({'error': 'Backend received empty inputs. Check your frontend code.'}), 400
        # ૧. મોડલ લોડિંગ
        model_path = os.path.join('models', f'{disease}_model.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': f'Model for {disease} not found'}), 404
        
        model = joblib.load(model_path)
        
        # ૨. ડેટા પ્રિપેરેશન (ફીચરના નામ અને વેલ્યુઝ)
        feature_names = list(inputs.keys())
        # ખાતરી કરો કે બધી વેલ્યુ નંબર (float) છે
        input_values = np.array([[float(v) for v in inputs.values()]], dtype=np.float32)
        
        # ૩. SHAP લોજિક
        def model_predict(d):
            # Stacking model માટે પ્રોબેબિલિટી પ્રેડિક્શન
            return model.predict_proba(d)[:, 1]

        # બેકગ્રાઉન્ડ ડેટા સેટ કરવો (0-reference)
        background = np.zeros((1, len(feature_names))) 
        explainer = shap.KernelExplainer(model_predict, background)
        
        # SHAP વેલ્યુ ગણો (આમાં ૫-૧૦ સેકન્ડ લાગી શકે છે)
        shap_values = explainer.shap_values(input_values)
        
    # ૪. SHAP વેલ્યુ ગણો
        shap_values = explainer.shap_values(input_values)
        
        explanation = []
        
        # --- આ લોજિક એરર વગર ડેટા કાઢશે ---
        # જો shap_values લિસ્ટ હોય (જેમ કે Stacking માં ઘણીવાર હોય છે), તો પહેલો એલિમેન્ટ લો
        if isinstance(shap_values, list):
            # Binary classification માં ક્યારેક [array, array] હોય છે, આપણે 1st array જોઈએ
            actual_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            actual_vals = shap_values

        for i, name in enumerate(feature_names):
            try:
                # જો ડેટા 2D હોય (1, 9), તો [0][i] વાપરો, નહીંતર [i]
                if len(actual_vals.shape) > 1:
                    val = float(actual_vals[0][i])
                else:
                    val = float(actual_vals[i])
            except:
                val = 0.0
                
            explanation.append({
                'feature': name,
                'shap_value': val,
                'impact': 'High Risk Factor' if val > 0 else 'Low Risk Factor'
            })
            
        # ૫. રિસ્પોન્સ (સાથે શોર્ટેડ લિસ્ટ)
        return jsonify({
            'status': 'success',
            'explanation': sorted(explanation, key=lambda x: abs(x['shap_value']), reverse=True)
        })

    except Exception as e:
        import traceback
        print(f"SHAP Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    
@app.route('/about-contact')
def about_contact():
    return render_template('about_contact.html')
 

# ══════════════════════════════════════════════════════════════
# EMERGENCY ASSIST API ROUTES  (SQLite-backed)
# ══════════════════════════════════════════════════════════════

def _ensure_emergency_tables():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS emergency_contacts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                name       TEXT NOT NULL,
                num        TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS medical_id (
                user_email  TEXT PRIMARY KEY,
                blood_group TEXT DEFAULT '',
                allergies   TEXT DEFAULT '',
                conditions  TEXT DEFAULT '',
                updated_at  TEXT DEFAULT (datetime('now'))
            );
        """)

_ensure_emergency_tables()

# ── Emergency Contacts ─────────────────────────────────────────
@app.route('/api/emergency/contacts', methods=['GET'])
@login_required
def get_emergency_contacts():
    email = session['user']['email']
    with get_db() as conn:
        rows = conn.execute(
            'SELECT id, name, num FROM emergency_contacts WHERE user_email=? ORDER BY id LIMIT 6',
            (email,)
        ).fetchall()
    return jsonify({'contacts': [dict(r) for r in rows]})

@app.route('/api/emergency/contacts', methods=['POST'])
@login_required
def save_emergency_contact():
    email = session['user']['email']
    data  = request.json or {}
    name  = data.get('name', '').strip()
    num   = data.get('num', '').strip()
    if not name or not num:
        return jsonify({'success': False, 'error': 'Name and number required'}), 400
    with get_db() as conn:
        count = conn.execute(
            'SELECT COUNT(*) FROM emergency_contacts WHERE user_email=?', (email,)
        ).fetchone()[0]
        if count >= 6:
            return jsonify({'success': False, 'error': 'Maximum 6 contacts allowed'}), 400
        dup = conn.execute(
            'SELECT 1 FROM emergency_contacts WHERE user_email=? AND num=?', (email, num)
        ).fetchone()
        if dup:
            return jsonify({'success': False, 'error': 'This number already saved'}), 400
        conn.execute(
            'INSERT INTO emergency_contacts (user_email, name, num) VALUES (?,?,?)',
            (email, name, num)
        )
        rows = conn.execute(
            'SELECT id, name, num FROM emergency_contacts WHERE user_email=? ORDER BY id LIMIT 6',
            (email,)
        ).fetchall()
    return jsonify({'success': True, 'contacts': [dict(r) for r in rows]})

@app.route('/api/emergency/contacts/<int:contact_id>', methods=['DELETE'])
@login_required
def delete_emergency_contact(contact_id):
    email = session['user']['email']
    with get_db() as conn:
        conn.execute(
            'DELETE FROM emergency_contacts WHERE id=? AND user_email=?',
            (contact_id, email)
        )
        rows = conn.execute(
            'SELECT id, name, num FROM emergency_contacts WHERE user_email=? ORDER BY id LIMIT 6',
            (email,)
        ).fetchall()
    return jsonify({'success': True, 'contacts': [dict(r) for r in rows]})

# ── Medical ID ─────────────────────────────────────────────────
@app.route('/api/emergency/medical-id', methods=['GET'])
@login_required
def get_medical_id():
    email = session['user']['email']
    with get_db() as conn:
        row = conn.execute(
            'SELECT blood_group, allergies, conditions FROM medical_id WHERE user_email=?',
            (email,)
        ).fetchone()
    return jsonify(dict(row) if row else {'blood_group': '', 'allergies': '', 'conditions': ''})

@app.route('/api/emergency/medical-id', methods=['POST'])
@login_required
def save_medical_id_api():
    email = session['user']['email']
    data  = request.json or {}
    bg    = data.get('blood_group', '').strip()
    al    = data.get('allergies', '').strip()
    co    = data.get('conditions', '').strip()
    with get_db() as conn:
        conn.execute("""
            INSERT INTO medical_id (user_email, blood_group, allergies, conditions, updated_at)
            VALUES (?,?,?,?,datetime('now'))
            ON CONFLICT(user_email) DO UPDATE SET
                blood_group=excluded.blood_group,
                allergies=excluded.allergies,
                conditions=excluded.conditions,
                updated_at=excluded.updated_at
        """, (email, bg, al, co))
    return jsonify({'success': True})

# ── Voice Command (server-side processing) ─────────────────────
@app.route('/api/emergency/voice-command', methods=['POST'])
@login_required
def process_voice_command():
    data = request.json or {}
    said = (data.get('text') or '').lower().strip()
    if not said:
        return jsonify({'matched': False, 'message': 'No speech detected'})
    VOICE_COMMANDS = [
        {'kws': ['call 108','108 call','ambulance','108 dial','ek sau aath'],
         'action': 'call', 'target': '108', 'guide': None,
         'message': '🚑 Calling 108 Ambulance...', 'level': 'critical'},
        {'kws': ['call 112','112','all emergency'],
         'action': 'call', 'target': '112', 'guide': None,
         'message': '🚨 Calling 112 Emergency...', 'level': 'critical'},
        {'kws': ['call 100','police'],
         'action': 'call', 'target': '100', 'guide': None,
         'message': '🚔 Calling 100 Police...', 'level': 'critical'},
        {'kws': ['call 101','fire brigade','fire call'],
         'action': 'call', 'target': '101', 'guide': None,
         'message': '🔥 Calling 101 Fire Brigade...', 'level': 'critical'},
        {'kws': ['call family','family call','parivar','kuttumb','ghar'],
         'action': 'call_family', 'target': None, 'guide': None,
         'message': '👨‍👩‍👧 Calling Family Emergency Contact...', 'level': 'critical'},
        {'kws': ['send sos','sos','send help','bachao','madad'],
         'action': 'sos', 'target': None, 'guide': None,
         'message': '📍 Sending SOS with your location...', 'level': 'critical'},
        {'kws': ['find hospital','nearest hospital','hospital find'],
         'action': 'hospital', 'target': None, 'guide': None,
         'message': '🏥 Finding nearest hospitals...', 'level': 'critical'},
        {'kws': ['heart attack','chest pain','hraday','dil ka dora'],
         'action': 'call', 'target': '108', 'guide': 'heart',
         'message': '🔴 Heart Attack! Calling 108. See CPR guide.', 'level': 'critical'},
        {'kws': ['stroke','brain stroke','paralysis','laqva'],
         'action': 'call', 'target': '108', 'guide': 'stroke',
         'message': '🔴 Stroke! Calling 108. Do FAST test.', 'level': 'critical'},
        {'kws': ['unconscious','fainted','behosh','murcha'],
         'action': 'call', 'target': '108', 'guide': 'heart',
         'message': '🔴 Unconscious! Calling 108. Start CPR.', 'level': 'critical'},
        {'kws': ['accident','road accident','apghaat','hadsa'],
         'action': 'call', 'target': '108', 'guide': 'bleed',
         'message': '🔴 Accident! Calling 108.', 'level': 'critical'},
        {'kws': ['bleeding','blood','rakat','khun'],
         'action': 'call', 'target': '108', 'guide': 'bleed',
         'message': '🔴 Bleeding! Apply pressure. Calling 108.', 'level': 'critical'},
        {'kws': ['choke','choking','ghuti'],
         'action': 'call', 'target': '108', 'guide': 'choke',
         'message': '🔴 Choking! Heimlich now. Calling 108.', 'level': 'critical'},
        {'kws': ['burn','jalavu','daajyu'],
         'action': None, 'target': None, 'guide': 'burn',
         'message': '🟡 Burn! Cool with water 15 min.', 'level': 'moderate'},
        {'kws': ['fracture','broken bone','hadku'],
         'action': 'call', 'target': '108', 'guide': 'fracture',
         'message': '🔴 Fracture! Immobilize limb. Calling 108.', 'level': 'critical'},
        {'kws': ['poison','poisoning','zaher'],
         'action': 'call', 'target': '108', 'guide': 'poison',
         'message': '🔴 Poisoning! Do NOT induce vomiting. Calling 108.', 'level': 'critical'},
        {'kws': ['snake','snake bite','saap'],
         'action': 'call', 'target': '108', 'guide': 'snake',
         'message': '🔴 Snake bite! Keep still. Calling 108.', 'level': 'critical'},
        {'kws': ['sugar low','sugar high','diabetic'],
         'action': None, 'target': None, 'guide': 'diabetic',
         'message': '🟡 Diabetic emergency! Give sugar if conscious.', 'level': 'moderate'},
        {'kws': ['siren','play siren'],
         'action': 'siren', 'target': None, 'guide': None,
         'message': '📢 Playing loud siren!', 'level': 'critical'},
    ]
    matched = next((c for c in VOICE_COMMANDS if any(kw in said for kw in c['kws'])), None)
    if not matched:
        return jsonify({'matched': False, 'heard': said,
                        'message': 'Not recognized. Try: "Call 108", "Heart attack", "Send SOS"'})
    resp = {'matched': True, 'heard': said, 'action': matched['action'],
            'target': matched['target'], 'guide': matched['guide'],
            'message': matched['message'], 'level': matched['level']}
    if matched['guide'] and matched['guide'] in FIRST_AID_DATA:
        resp['first_aid'] = FIRST_AID_DATA[matched['guide']]
    print(f"[VOICE] '{said}' → {matched['action']} {matched.get('target','')}")
    return jsonify(resp)

# ── SOS with location ──────────────────────────────────────────
@app.route('/api/emergency/sos', methods=['POST'])
@login_required
def send_sos():
    email = session['user']['email']
    data  = request.json or {}
    lat   = data.get('lat')
    lng   = data.get('lng')
    loc_link = f'https://maps.google.com/?q={lat},{lng}' if lat and lng else None
    loc_text = f'📍 My Location: {loc_link}' if loc_link else '📍 Location not available'
    with get_db() as conn:
        mid = conn.execute(
            'SELECT blood_group, conditions FROM medical_id WHERE user_email=?', (email,)
        ).fetchone()
        rows = conn.execute(
            'SELECT name, num FROM emergency_contacts WHERE user_email=? ORDER BY id LIMIT 6',
            (email,)
        ).fetchall()
    contacts = [dict(r) for r in rows]
    med_info = ''
    if mid:
        bg, co = mid['blood_group'], mid['conditions']
        if bg or co:
            med_info = f'\n🩸 Blood: {bg or "Unknown"} | 💊 Conditions: {co or "None"}'
    now_str  = datetime.now().strftime('%d %b %Y, %H:%M')
    msg_body = (
        f'🚨 EMERGENCY — VitalsAI SOS Alert!\n'
        f'I need IMMEDIATE medical help!\n'
        f'{loc_text}{med_info}\n'
        f'⏰ Time: {now_str}\n'
        f'🚑 Please call 108 for me or come NOW!\n'
        f'— VitalsAI Health App'
    )
    print(f'[SOS] Triggered by {email} | location: {loc_link} | contacts: {len(contacts)}')
    return jsonify({
        'success':      True,
        'contacts':     contacts,
        'message_body': msg_body,
        'location_link': loc_link,
        'note': f'Open SMS for {len(contacts)} contact(s) on your device'
    })

# ── Log SOS Event ──────────────────────────────────────────────
@app.route('/api/emergency/sos-log', methods=['POST'])
@login_required
def log_sos_event():
    """Log when SOS was triggered (for audit/safety)."""
    user  = session.get('user', {})
    email = user.get('email', '')
    data  = request.json or {}
    lat   = data.get('lat')
    lng   = data.get('lng')
    contacts_notified = data.get('contacts', [])
    log_entry = {
        'time':     datetime.now().isoformat(),
        'email':    email,
        'lat':      lat,
        'lng':      lng,
        'contacts': contacts_notified,
    }
    print(f"[SOS] 🚨 SOS triggered by {email} at {log_entry['time']}")
    if lat and lng:
        print(f"[SOS] Location: https://maps.google.com/?q={lat},{lng}")
    print(f"[SOS] Notified {len(contacts_notified)} contact(s)")
    return jsonify({'success': True, 'logged': log_entry})

# ── Voice Command Log ──────────────────────────────────────────
@app.route('/api/emergency/voice-log', methods=['POST'])
@login_required
def log_voice_command():
    """Log voice command used (for analytics)."""
    user    = session.get('user', {})
    email   = user.get('email', '')
    data    = request.json or {}
    command = data.get('command', '')
    action  = data.get('action', '')
    print(f"[VOICE] User {email} said: '{command}' → action: {action}")
    return jsonify({'success': True})

# ── Get First Aid Content (Dynamic) ───────────────────────────
FIRST_AID_DATA = {
    'heart': {
        'title': '❤️ Heart Attack — CPR Guide',
        'tags':  ['heart attack', 'chest pain', 'cpr', 'cardiac', 'hraday', 'hirday'],
        'steps': [
            'Call 108 immediately',
            'Sit or lie down in comfortable position — keep calm',
            'Loosen all tight clothing (belt, collar)',
            'Chew 1 aspirin 325mg if NOT allergic',
            'Do NOT eat, drink, or let person walk',
            'If unconscious + not breathing → Start CPR',
        ],
        'cpr': 'CPR: 30 chest compressions (hard + fast, 100/min) + 2 rescue breaths. Repeat until help arrives.',
        'call': '108',
        'level': 'critical',
    },
    'stroke': {
        'title': '🧠 Stroke — FAST Test',
        'tags':  ['stroke', 'brain', 'paralysis', 'laqva', 'face drooping', 'speech'],
        'steps': [
            'F — Face: Ask to smile — is one side drooping?',
            'A — Arm: Raise both arms — does one drift down?',
            'S — Speech: Repeat sentence — is it slurred?',
            'T — Time: Any YES above → Call 108 IMMEDIATELY',
            'Lay patient on side, head slightly raised',
            'Do NOT give food, water, or medications',
        ],
        'warn': 'Every minute = 2 million brain cells lost. Act NOW.',
        'call': '108',
        'level': 'critical',
    },
    'burn': {
        'title': '🔥 Burn Injury',
        'tags':  ['burn', 'fire', 'scald', 'jalavu', 'daajyu'],
        'steps': [
            'Cool under cold running water 10-20 minutes',
            'Remove jewelry and tight clothing near burn',
            'Cover loosely with clean non-fluffy cloth',
            'DO NOT use ice, butter, toothpaste, or oil',
            'DO NOT break blisters',
            'If larger than palm or on face/hands → Call 108',
        ],
        'warn': 'Chemical burn: Flush with water 20 min. Remove clothes carefully.',
        'call': None,
        'level': 'moderate',
    },
    'bleed': {
        'title': '🩸 Bleeding Control',
        'tags':  ['bleeding', 'blood', 'wound', 'cut', 'accident', 'rakat'],
        'steps': [
            'Apply firm direct pressure with clean cloth',
            'Elevate wound above heart level if possible',
            'DO NOT remove cloth — add more layers if soaked',
            'Press continuously for 10-15 minutes',
            'For objects in wound — do NOT remove them',
            'Severe bleeding → Call 108 immediately',
        ],
        'warn': 'Tourniquet: Only for limb bleeding. Tighten 5cm above wound. Note time applied.',
        'call': '108',
        'level': 'critical',
    },
    'choke': {
        'title': '😮 Choking — Heimlich',
        'tags':  ['choke', 'choking', 'heimlich', 'throat', 'food', 'ghuti'],
        'steps': [
            'Ask "Are you choking?" — if they can cough, encourage it',
            'Lean person forward — give 5 firm back blows between shoulder blades',
            'Stand behind, arms around waist — fist above navel',
            'Thrust inward + upward 5 times (Heimlich)',
            'Alternate back blows and thrusts until cleared',
            'If unconscious → Start CPR, Call 108',
        ],
        'warn': 'Infant: Use 5 back blows + 5 chest thrusts (not abdominal). Hold face-down.',
        'call': '108',
        'level': 'critical',
    },
    'fracture': {
        'title': '🦴 Fracture / Broken Bone',
        'tags':  ['fracture', 'broken bone', 'hadku', 'todayu'],
        'steps': [
            'DO NOT move the person unless unsafe',
            'Immobilize the broken limb with splint or padded boards',
            'Apply ice pack wrapped in cloth (not directly on skin)',
            'Elevate if possible — DO NOT try to straighten bone',
            'Control any bleeding with gentle pressure',
            'Open fracture (bone visible) → Call 108 immediately',
        ],
        'warn': 'Spine/neck injury: Do NOT move patient. Wait for paramedics.',
        'call': '108',
        'level': 'critical',
    },
    'poison': {
        'title': '☠️ Poisoning',
        'tags':  ['poison', 'poisoning', 'chemical', 'zaher', 'nasha'],
        'steps': [
            'Call 108 or Poison Control immediately',
            'DO NOT induce vomiting unless told to',
            'If chemical on skin → Remove clothes, rinse with water 15 min',
            'If in eyes → Rinse with clean water 15 min',
            'Bring the poison container/label to hospital',
            'If unconscious + breathing → Recovery position',
        ],
        'warn': 'India Poison Control: 1800-116-117 (24hr free). Keep number saved!',
        'call': '108',
        'level': 'critical',
    },
    'snake': {
        'title': '🐍 Snake Bite',
        'tags':  ['snake', 'snake bite', 'venom', 'saap', 'naag'],
        'steps': [
            'Move away from snake — do NOT try to catch it',
            'Keep person calm and still — movement spreads venom faster',
            'Remove watches, rings, tight clothing near bite',
            'Keep bite below heart level',
            'Mark edge of swelling with pen + note time',
            'Get to hospital within 1 hour — Call 108',
        ],
        'warn': 'DO NOT: cut the bite, suck venom, apply tourniquet, or give alcohol.',
        'call': '108',
        'level': 'critical',
    },
    'diabetic': {
        'title': '💉 Diabetic Emergency',
        'tags':  ['diabetic', 'sugar low', 'sugar high', 'blood sugar', 'insulin', 'glucose'],
        'low_sugar': [
            'Give 4 glucose tablets or 150ml fruit juice or sugar water',
            'Wait 15 min — recheck symptoms',
            'Give small snack (biscuits, bread)',
            'If unconscious — DO NOT give anything by mouth. Call 108',
        ],
        'high_sugar': [
            'Encourage drinking water',
            'Check for ketones if possible',
            'Administer prescribed insulin if available',
            'If vomiting/confusion → Call 108 immediately',
        ],
        'warn': 'Confusion + sweating + shakiness = LOW sugar. Always carry glucose tablets.',
        'call': '108',
        'level': 'moderate',
    },
    'drown': {
        'title': '🌊 Drowning',
        'tags':  ['drowning', 'water', 'drown', 'dubayu'],
        'steps': [
            'Do NOT jump in unless trained — throw a rope/float object',
            'Once out of water — lay flat on firm surface',
            'Check breathing — if not breathing → Start CPR immediately',
            '30 compressions + 2 rescue breaths',
            'Turn to recovery position once breathing returns',
            'Call 108 — all drowning victims need hospital check',
        ],
        'warn': 'Secondary drowning can occur hours later. Always get medical evaluation.',
        'call': '108',
        'level': 'critical',
    },
}

@app.route('/api/emergency/first-aid', methods=['GET'])
def get_first_aid_all():
    """Return all first aid guides."""
    result = {}
    for key, data in FIRST_AID_DATA.items():
        result[key] = {
            'title': data['title'],
            'tags':  data.get('tags', []),
            'level': data.get('level', 'moderate'),
            'call':  data.get('call'),
        }
    return jsonify({'guides': result})

@app.route('/api/emergency/first-aid/<condition>', methods=['GET'])
def get_first_aid(condition):
    """Return first aid guide for specific condition."""
    condition = condition.lower()
    data = FIRST_AID_DATA.get(condition)
    if not data:
        # Try partial match
        for key, val in FIRST_AID_DATA.items():
            tags = val.get('tags', [])
            if any(condition in tag for tag in tags):
                data = val
                break
    if not data:
        return jsonify({'error': f'No first aid guide found for: {condition}'}), 404
    return jsonify(data)

@app.route('/api/emergency/search-first-aid', methods=['GET'])
def search_first_aid():
    """Search first aid guides by keyword."""
    query = request.args.get('q', '').lower().strip()
    if not query:
        return jsonify({'results': list(FIRST_AID_DATA.keys())})
    matches = []
    for key, data in FIRST_AID_DATA.items():
        tags  = ' '.join(data.get('tags', []))
        title = data.get('title', '').lower()
        steps = ' '.join(data.get('steps', data.get('low_sugar', [])))
        if query in tags or query in title or query in steps.lower():
            matches.append({
                'key':   key,
                'title': data['title'],
                'level': data.get('level', 'moderate'),
                'call':  data.get('call'),
            })
    return jsonify({'results': matches, 'query': query})


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico')



if __name__ == '__main__':
    print("\n" + "="*55)
    print("  VitalsAI — http://localhost:5000  SQLite DB — Active ✅")
    print("  Login     — http://localhost:5000/login")
    print("  Assistant — http://localhost:5000/assistant")
    print("  Status    — http://localhost:5000/status")
    print("  History   — http://localhost:5000/history")
    print("  BMI       — http://localhost:5000/bmi")
    print("="*55 + "\n")
    app.run(debug=True , port=5000)