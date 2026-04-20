from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import json
import uuid
import hashlib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from authlib.integrations.flask_client import OAuth

# ── Load .env file ─────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] .env loaded")
except ImportError:
    print("[INFO] python-dotenv not installed")

app = Flask(__name__)
app.secret_key = 'vitalsai-secret-2024'

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

MODELS  = {}
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyDIpc_Y5jqmotHI8Jb5yAef1UI1RnkfRKk')
USERS   = {}   # email -> {name, pwd, created}

# ── In-memory storage for history & trends ────────────────
HISTORY = {}  # session_id -> list of predictions

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
            'model':    joblib.load('models/brain_ml_model.pkl'),
            'features': joblib.load('models/brain_features.pkl'),
            'selector': joblib.load('models/brain_selector.pkl'),
            'scaler':   joblib.load('models/brain_scaler.pkl'),  # FIX: scaler add karyu
            'threshold': 0.15
        }
        print("[OK] Brain model loaded")
    except Exception as e:
        print(f"[WARN] Brain: {e}")

    # ── DIABETES ───────────────────────────────────────────
    try:
        MODELS['diabetes'] = {
            'model':   joblib.load('models/diabetes_stack_model.pkl'),
            'scaler':  joblib.load('models/diabetes_scaler.pkl'),
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
            'model':      joblib.load('models/kidney_pipeline.pkl'),
            'features':   joblib.load('models/kidney_features.pkl'),
            'target_map': joblib.load('models/kidney_target_map.pkl'),
        }
        print("[OK] Kidney model loaded")
    except Exception as e:
        print(f"[WARN] Kidney: {e}")

    # ── EYE ────────────────────────────────────────────────
    try:
        import tensorflow as tf, json as _json

        # Fix: quantization_config issue — patch Dense layer
        from tensorflow.keras.layers import Dense as _OrigDense
        class _PatchedDense(_OrigDense):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        MODELS['eye'] = {
            'model': tf.keras.models.load_model(
                'models/eye_cnn_model.h5',
                custom_objects={'Dense': _PatchedDense},
                compile=False
            ),
        }
        with open('models/eye_class_indices.json') as f:
            ci = _json.load(f)
        MODELS['eye']['reverse'] = {v: k for k, v in ci.items()}
        print("[OK] Eye CNN loaded")
        print("[OK] Eye classes:", list(MODELS['eye']['reverse'].values()))
    except Exception as e:
        print(f"[WARN] Eye: {e}")

      # ── LUNG ───────────────────────────────────────────────
    try:
        import json as _json_lung
        MODELS['lung'] = {
            'model':      joblib.load('models/lung_stacking_model.pkl'),
            'le_gender':  joblib.load('models/lung_le_gender.pkl'),
            'le_stage':   joblib.load('models/lung_le_stage.pkl'),
            'le_family':  joblib.load('models/lung_le_family.pkl'),
            'le_smoking': joblib.load('models/lung_le_smoking.pkl'),
            'le_treat':   joblib.load('models/lung_le_treatment.pkl'),
        }
        with open('models/lung_metadata.json') as f:
            MODELS['lung']['meta'] = _json_lung.load(f)
        print("[OK] Lung model loaded")
    except Exception as e:
        print(f"[WARN] Lung: {e}")


## 2. DOCTOR_MAP dictionary માં add કरो:

    DOCTOR_MAP = {
    'lung': {
        'HIGH RISK': 'Oncologist (Lung Cancer Specialist) → Urgent!',
        'MEDIUM RISK': 'Pulmonologist → Oncology Referral',
        'LOW RISK': 'Pulmonologist → Regular Monitoring',
    }
}
load_models()

# ── Helper: Save to history ────────────────────────────────
def save_to_history(session_id, disease, inputs, result):
    if session_id not in HISTORY:
        HISTORY[session_id] = []
    HISTORY[session_id].append({
        'id':        str(uuid.uuid4())[:8],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'disease':   disease,
        'inputs':    inputs,
        'result':    result
    })
    # Keep last 50 records only
    HISTORY[session_id] = HISTORY[session_id][-50:]

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
    }
}

# ── Pages ──────────────────────────────────────────────────
@app.route('/')
def index():
    if 'sid' not in session:
        session['sid'] = str(uuid.uuid4())
    if 'user' not in session:
        return redirect(url_for('login_page'))
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
        if email not in USERS:
            USERS[email] = {'name': name, 'pwd': None, 'created': datetime.now().isoformat(), 'google': True}
        if 'sid' not in session:
            session['sid'] = str(uuid.uuid4())
        session['user'] = {'email': email, 'name': name}
        return redirect(url_for('index'))
    except Exception as e:
        print(f"[Google OAuth Error] {e}")
        return redirect(url_for('login_page') + '?error=google_failed')

def login_required(f):
    """Decorator: redirect to login if not logged in"""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login_page'))
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
@app.route('/api/signup', methods=['POST'])
def api_signup():
    d     = request.json
    email = d.get('email', '').strip().lower()
    name  = d.get('name', '').strip()
    pwd   = d.get('password', '')
    if not email or not name or not pwd:
        return jsonify({'success': False, 'error': 'All fields required'})
    if len(pwd) < 6:
        return jsonify({'success': False, 'error': 'Password min 6 characters'})
    if email in USERS:
        return jsonify({'success': False, 'error': 'Email already registered'})
    USERS[email] = {'name': name, 'pwd': hash_pwd(pwd), 'created': datetime.now().isoformat()}
    return jsonify({'success': True})

@app.route('/api/login', methods=['POST'])
def api_login():
    d     = request.json
    email = d.get('email', '').strip().lower()
    pwd   = d.get('password', '')
    u     = USERS.get(email)
    if not u or u['pwd'] != hash_pwd(pwd):
        return jsonify({'success': False, 'error': 'Invalid email or password'})
    session['user'] = {'email': email, 'name': u['name']}
    return jsonify({'success': True, 'name': u['name']})

@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.pop('user', None)
    return jsonify({'success': True})

@app.route('/api/me')
def api_me():
    u = session.get('user')
    if u:
        return jsonify({'logged_in': True, 'name': u['name'], 'email': u['email']})
    return jsonify({'logged_in': False})


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
    stress   = d.get('stress', 'low')
    age      = int(d.get('age', 30))

    diet_recs = []; exercise_recs = []; sleep_recs = []; medical_recs = []
    score = 100

    if bmi > 30:
        diet_recs     += ["Calorie deficit diet follow karo", "Junk food avoid karo", "Fruits & vegetables vadhu lo"]
        exercise_recs += ["Daily 45 min walking shuru karo", "Cardio exercise 4x per week"]
        score -= 20
    elif bmi > 25:
        diet_recs     += ["Healthy balanced diet lo", "Processed food ochhu karo"]
        exercise_recs += ["Daily 30 min brisk walking"]
        score -= 10
    elif bmi < 18.5:
        diet_recs += ["Protein-rich food vadhu lo (eggs, paneer, daal)", "Healthy fats lo (nuts, avocado, ghee)"]
        score -= 10
    else:
        diet_recs.append("BMI normal che! Balanced diet continue rakho")

    if bp >= 140:
        diet_recs    += ["Low sodium diet — namak bilkul ochhu", "DASH diet follow karo"]
        medical_recs += ["BP rojana monitor karo", "Doctor ne miljo"]
        score -= 15
    elif bp >= 130:
        diet_recs.append("Salt intake thodi kam karo")
        score -= 5

    if chol >= 240:
        diet_recs    += ["Saturated fats avoid karo (butter, fried food)", "Oats & fiber-rich food lo"]
        medical_recs.append("Lipid profile test karavo")
        score -= 10
    elif chol >= 200:
        diet_recs.append("Healthy fats prefer karo — olive oil, nuts")
        score -= 5

    if smoker:
        medical_recs  += ["Smoking taatkaalik band karo", "Doctor ni help lo quit karva"]
        exercise_recs.append("Exercise smoking chhadvama help kare che")
        score -= 20

    if not activity:
        exercise_recs += ["Daily 30 min walking shuru karo!", "Lift ni jagya stairs use karo"]
        score -= 10
    else:
        exercise_recs.append("Exercise excellent! Niyamit chalalu rakho")

    if diabetes:
        diet_recs    += ["Sugar & refined carbs bilkul avoid karo", "Low glycemic index food prefer karo"]
        medical_recs += ["Blood glucose daily monitor karo", "HbA1c test daekaek 3 mahine ma karavo"]
        score -= 15

    if sleep < 6:
        sleep_recs += ["7-8 hours suvani koshish karo", "Suvata 1 hour pehla phone/TV band karo"]
        score -= 10
    elif sleep > 9:
        sleep_recs.append("Vadhu suvath pan hanikarak — 7-8 hours ideal che")
    else:
        sleep_recs.append("Sleep schedule excellent che!")

    if stress == 'high':
        sleep_recs    += ["Daily 10-15 min meditation try karo", "Deep breathing exercises karo"]
        exercise_recs.append("Yoga — stress relief mate best che")
        score -= 10
    elif stress == 'medium':
        sleep_recs.append("Relaxation techniques try karo")

    if age >= 50:
        medical_recs += ["Yearly full body checkup karavo", "Vitamin D, B12, Iron levels check karavo"]
    elif age >= 40:
        medical_recs.append("BP, Sugar, Cholesterol yearly test karavo")
    else:
        medical_recs.append("Preventive health checkup daekaek 2 year ma karavo")

    if not diet_recs:     diet_recs     = ["Balanced diet lo — daal, chaval, shaak, fruit"]
    if not exercise_recs: exercise_recs = ["Exercise continue karo"]
    if not sleep_recs:    sleep_recs    = ["Sleep schedule excellent che!"]

    score      = max(0, min(100, score))
    risk_level = 'high' if score < 50 else ('medium' if score < 75 else 'low')

    return jsonify({
        'health_score': score, 'risk_level': risk_level,
        'diet': diet_recs, 'exercise': exercise_recs,
        'sleep': sleep_recs, 'medical': medical_recs
    })

# ── AI Chatbot API (Claude AI Powered) ────────────────────
CHAT_SYSTEM = """You are the AI Health Assistant for "Vitals AI" — a health prediction and wellness platform.

DEFAULT LANGUAGE: Always respond in English unless the user writes in a different language.
LANGUAGE RULE: If user writes in Gujarati → reply in Gujarati. If Hindi → reply in Hindi. Otherwise always English.

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

@app.route('/api/chat', methods=['POST'])
def chat():
    import urllib.request, json as _json
    data    = request.json or {}
    msg     = data.get('message', '').strip()
    history = data.get('history', [])
    mode    = data.get('mode', 'ai')

    if not msg:
        return jsonify({'response': 'Please enter a message.'})


    # ── Gemini AI mode ───────────────────────────────────────
    api_key = GEMINI_API_KEY
    if api_key:  # Always try Gemini regardless of mode
        try:
            # Build system prompt + full question in single user turn
            full_prompt = CHAT_SYSTEM + "\n\nUser question: " + msg

            # Add recent history context if available
            if history:
                hist_text = ""
                for h in history[-6:]:
                    role_label = "User" if h.get('role') == 'user' else "Assistant"
                    hist_text += f"\n{role_label}: {str(h.get('content',''))[:300]}"
                if hist_text:
                    full_prompt = CHAT_SYSTEM + "\n\nRecent conversation:" + hist_text + "\n\nNow answer this: " + msg

            payload = _json.dumps({
                'contents': [{'role': 'user', 'parts': [{'text': full_prompt}]}],
                'generationConfig': {
                    'temperature': 0.7,
                    'maxOutputTokens': 1024,
                },
                'safetySettings': [
                    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
                    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
                    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
                    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
                ]
            }).encode('utf-8')

            url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}'
            req = urllib.request.Request(
                url,
                data=payload,
                headers={'Content-Type': 'application/json'},
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = _json.loads(resp.read())
                reply  = result['candidates'][0]['content']['parts'][0]['text']
                # Format markdown to HTML
                reply_html = reply.replace('\n', '<br>')
                reply_html = reply_html.replace('**', '<b>', 1)
                import re
                reply_html = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', reply)
                reply_html = reply_html.replace('\n', '<br>')
                print(f"[Gemini OK] reply len={len(reply)}")
                return jsonify({'response': reply_html, 'source': 'gemini'})
        except Exception as e:
            import traceback; traceback.print_exc()
            if hasattr(e, 'read'):
                try:
                    body = e.read().decode()
                    print("[Gemini API body]", body)
                except: pass
            print(f"[Gemini API error] {e}")
            # Fall through to KB

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
@app.route('/predict/kidney', methods=['POST'])
def predict_kidney():
    try:
        m = MODELS.get('kidney')
        if not m: return jsonify({'error': 'Kidney model not loaded.'}), 500
        data        = request.json
        reverse_map = {v: k for k, v in m['target_map'].items()}
        inputs      = {f: float(data.get(f, 0)) for f in m['features']}
        patient     = pd.DataFrame([inputs]).reindex(columns=m['features'], fill_value=0)
        pred        = int(m['model'].predict(patient)[0])
        proba       = m['model'].predict_proba(patient)[0]
        label       = reverse_map[pred]
        conf        = round(float(max(proba))*100, 2)
        all_p       = {reverse_map[i]: round(float(p)*100, 2) for i, p in enumerate(proba)}
        doctor      = DOCTOR_MAP['kidney'].get(label, 'General Physician')

        # ── XAI Explanation ─────────────────────────────────
        why_high, why_low, suggestions, lifestyle = [], [], [], {}
        sc  = inputs.get('sc', 0)   # serum creatinine
        egfr = inputs.get('bgr',0)  # use as proxy
        hemo = inputs.get('hemo',0) # hemoglobin
        bp_v = inputs.get('bp',0)
        al   = inputs.get('al',0)   # albumin
        su   = inputs.get('su',0)   # sugar in urine

        if sc > 1.2:
            why_high.append(f"High Serum Creatinine ({sc:.2f} mg/dL) — indicates reduced kidney filtration")
            suggestions.append("Repeat creatinine + eGFR test — monitor kidney function monthly")
        else:
            why_low.append(f"Serum Creatinine ({sc:.2f}) in normal range — good kidney filtration")
        if hemo < 12:
            why_high.append(f"Low Hemoglobin ({hemo:.1f} g/dL) — anemia common in kidney disease")
            suggestions.append("Check for renal anemia — may need erythropoietin therapy")
        else:
            why_low.append(f"Hemoglobin ({hemo:.1f} g/dL) — adequate, less anemia risk")
        if bp_v >= 90:
            why_high.append(f"High diastolic BP ({bp_v} mmHg) — damages kidney blood vessels")
            suggestions.append("Strict BP control < 130/80 — ACE inhibitors preferred for CKD")
        else:
            why_low.append(f"Blood pressure ({bp_v}) within acceptable range")
        if al >= 3:
            why_high.append(f"Albumin in urine (grade {int(al)}) — kidney protein leakage sign")
        elif al == 0:
            why_low.append("No albumin in urine — healthy glomerular filtration")
        if su >= 2:
            why_high.append(f"Sugar in urine (grade {int(su)}) — diabetic nephropathy indicator")

        if label in ['Severe_Disease', 'High_Risk']:
            suggestions += ["Consult Nephrologist urgently",
                            "24-hour urine protein test recommended",
                            "Strict fluid and protein restriction",
                            "Avoid NSAIDs (ibuprofen) — nephrotoxic"]
            lifestyle = {'diet': ['Low protein diet: 0.6-0.8g/kg body weight',
                                  'Low potassium: avoid banana, orange, potato',
                                  'Low phosphorus: avoid dairy, nuts, cola drinks',
                                  'Limit fluid intake as advised by doctor',
                                  'Low sodium: < 2g/day'],
                         'exercise': ['Light walking only — avoid intense exercise',
                                      'Gentle yoga if BP is controlled'],
                         'sleep': ['8 hours sleep', 'Elevate legs to reduce swelling']}
        elif label in ['Moderate_Risk']:
            suggestions += ["Nephrology referral recommended",
                            "Kidney function test every 3 months",
                            "Control diabetes and BP strictly"]
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
            {'feature': 'Sugar in Urine',    'contribution_percent': 7,  'max_weight': 30},
        ]

        result = {'prediction': label, 'confidence': conf, 'probabilities': all_p,
                  'doctor': doctor, 'why_high_risk': why_high, 'why_low_risk': why_low,
                  'suggestions': suggestions, 'lifestyle_changes': lifestyle,
                  'feature_importance': feat_imp}
        sid = session.get('sid', 'default')
        save_to_history(sid, 'kidney', inputs, result)
        return jsonify(result)
    except Exception as e:
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

        # Encode inputs
        gender    = m['le_gender'].transform([data.get('Gender','Male')])[0]
        stage     = m['le_stage'].transform([data.get('CancerStage','Stage II')])[0]
        family    = m['le_family'].transform([data.get('FamilyHistory','No')])[0]
        smoking   = m['le_smoking'].transform([data.get('SmokingStatus','Never Smoked')])[0]
        treatment = m['le_treat'].transform([data.get('TreatmentType','Chemotherapy')])[0]

        age_val  = float(data.get('Age', 50))
        bmi_val  = float(data.get('BMI', 25))
        chol_val = int(data.get('CholesterolLevel', 200))

        inputs = {
            'age':               age_val,
            'gender':            int(gender),
            'cancer_stage':      int(stage),
            'family_history':    int(family),
            'smoking_status':    int(smoking),
            'bmi':               bmi_val,
            'cholesterol_level': chol_val,
            'hypertension':      int(data.get('Hypertension', 0)),
            'asthma':            int(data.get('Asthma', 0)),
            'cirrhosis':         int(data.get('Cirrhosis', 0)),
            'other_cancer':      int(data.get('OtherCancer', 0)),
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
    sid = session.get('sid', 'default')
    return jsonify(HISTORY.get(sid, []))

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    sid = session.get('sid', 'default')
    HISTORY[sid] = []
    return jsonify({'status': 'cleared'})

@app.route('/api/history/delete/<record_id>', methods=['DELETE'])
def delete_record(record_id):
    sid = session.get('sid', 'default')
    if sid in HISTORY:
        HISTORY[sid] = [r for r in HISTORY[sid] if r['id'] != record_id]
    return jsonify({'status': 'deleted'})

# ── Trend API (last N predictions for a disease) ──────────
@app.route('/api/trend/<disease>', methods=['GET'])
def get_trend(disease):
    sid     = session.get('sid', 'default')
    records = HISTORY.get(sid, [])
    trend   = [r for r in records if r['disease'] == disease][-20:]
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
        'missing': [d for d in ['heart','brain','diabetes','kidney','eye'] if d not in MODELS],
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
        explainer = shap.TreeExplainer(rf_model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            return shap_vals[1][0]
        return shap_vals[0]
    except Exception as e:
        print(f"[SHAP error] {e}")
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
        patient  = pd.DataFrame([inputs])
        rf_model = m['model'].named_estimators_['rf']
        sv = get_shap_values(rf_model, patient)
        if sv is None: return jsonify({'error': 'SHAP failed'}), 500
        explanation = build_explanation(features, patient.values[0], sv)
        return jsonify({'explanation': explanation, 'top_factors': explanation[:3]})
    except Exception as e:
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
 
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  VitalsAI — http://localhost:5000  Gemini AI — Active (Free)")
    print("  Login     — http://localhost:5000/login")
    print("  Assistant — http://localhost:5000/assistant")
    print("  Status    — http://localhost:5000/status")
    print("  History   — http://localhost:5000/history")
    print("  BMI       — http://localhost:5000/bmi")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)


    