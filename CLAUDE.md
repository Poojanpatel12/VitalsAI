# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

- **Run**: `python app.py` — Starts Flask dev server on http://localhost:5000
- **Install deps**: `pip install -r requirements.txt`
- **Virtual env**: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)

## Environment Setup

Create `.env` file with:
```
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
GEMINI_API_KEY=your-gemini-key
```

## High-Level Architecture

**VitalsAI** is a Flask-based multi-disease health prediction platform with 6 ML modules.

### Tech Stack
- **Backend**: Flask 2.3.2, Python 3.10+
- **ML Models**: scikit-learn (stacking ensembles), XGBoost, TensorFlow/MobileNetV2 (CNN for eye disease)
- **Auth**: Google OAuth 2.0 via Authlib + email/password (in-memory `USERS` dict)
- **PDF Reports**: ReportLab
- **AI Chatbot**: Gemini API (not Claude - uses `GEMINI_API_KEY`)

### Project Structure

```
app.py                 # Main Flask app - all routes and model loading
models/                # Pre-trained model files (.pkl for ML, .h5 for CNN)
templates/             # HTML templates (Jinja2)
datasets/              # Training datasets (CSV + image folders)
```

### Disease Prediction Modules

| Route | Model | Input | Output |
|-------|-------|-------|--------|
| `/predict/heart` | Stacking (RF+XGB+LR) | Age, BMI, BP, Chol, etc. | Risk % + doctor + XAI |
| `/predict/brain` | Stacking + Selector | Age, glucose, smoking, etc. | Risk % + doctor + XAI |
| `/predict/diabetes` | Stacking + Scaler | Glucose, insulin, BMI, etc. | Risk % + doctor + XAI |
| `/predict/kidney` | Pipeline classifier | 42 clinical features | Stage + confidence |
| `/predict/eye` | CNN MobileNetV2 | Retinal image upload | Disease class |
| `/predict/lung` | Stacking + LabelEncoders | Gender, stage, smoking, etc. | Risk + oncologist ref |

### Key Components

**Model Loading** (`load_models()` in app.py:48-152):
- Loads 6 models from `models/` directory on startup
- Eye model uses custom Dense layer patch to handle quantization_config issues
- Each model dict includes features list, thresholds, and preprocessing objects

**Authentication** (app.py:269-306):
- In-memory `USERS` dict for email/password (SHA256 hashed)
- Google OAuth via `/auth/google` and `/auth/google/callback`
- Session-based with `login_required` decorator

**XAI Explanations** (built into predict routes):
- `why_high_risk` / `why_low_risk` lists based on input values
- `feature_importance` array with contribution percentages
- `suggestions` and `lifestyle_changes` (diet/exercise/sleep) per risk level
- Bilingual support: English + Gujarati/Hindi in responses

**Chatbot** (`/api/chat` in app.py:433-598):
- Primary: Gemini API (`gemini-2.0-flash` model)
- Fallback: 50+ topic local knowledge base (KB array)
- Auto-detects language (English/Gujarati/Hindi)
- Structured response format: Symptoms → Causes → Treatment → When to see doctor

**History Tracking** (app.py:41-42, 156-167):
- In-memory `HISTORY` dict (session_id → predictions list)
- Stores last 50 predictions per session
- No database - data lost on server restart

### Routes Overview

- `/` — Dashboard (requires login)
- `/login` — Auth page (Google + email/password)
- `/heart`, `/brain`, `/diabetes`, `/kidney`, `/eye`, `/lung` — Prediction forms
- `/assistant` — AI health chatbot
- `/history` — Prediction history
- `/bmi`, `/lifestyle` — Utility tools

### API Endpoints

- `POST /api/login`, `/api/signup`, `/api/logout` — Auth
- `POST /predict/<disease>` — Get prediction with XAI
- `POST /api/chat` — AI chatbot
- `POST /api/bmi` — BMI calculator
- `POST /api/recommend` — Lifestyle recommendations
- `GET/POST /api/history` — Session history

### Important Notes

- Models are loaded globally in `MODELS` dict at startup
- Session data (history, users) is in-memory only - no persistence
- Eye CNN requires TensorFlow with custom_objects patch for quantization_config
- Brain model uses feature selector: `scaler → selector → predict`
- Kidney model returns stage labels via `target_map` pickle
- Lung model uses 5 LabelEncoders for categorical features
