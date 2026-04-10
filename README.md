
# 🏥 VitalsAI — AI-Powered Health Prediction Platform

# VitalsAI

### AI-powered health prediction platform for Heart, Brain, Diabetes, Kidney & Eye diseases

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=flat-square&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.x-blue?style=flat-square&logo=scikit-learn)
![Claude AI](https://img.shields.io/badge/Claude_AI-Anthropic-green?style=flat-square)
![Google OAuth](https://img.shields.io/badge/Google-OAuth2-red?style=flat-square&logo=google)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

**AI-powered platform for predicting Heart Disease, Brain Stroke, Diabetes, Kidney Disease & Eye Disease — with an intelligent health chatbot powered by Claude AI.**

*Final Year Project · SAL Institute of Technology · GTU · 2026*

</div>

---

## 🩺 Overview

VitalsAI is an end-to-end AI health prediction platform that uses trained machine learning and deep learning models to assess disease risk from clinical inputs and medical images. The platform provides instant risk assessment, doctor recommendations, health history tracking, PDF reports, and an AI-powered health chatbot — all in one unified web interface.

| Traditional Health Checkup | VitalsAI |
|---|---|
| Requires doctor visit for initial screening | Instant AI risk assessment from home |
| Single disease focus per visit | 5 disease modules in one platform |
| No historical tracking | Full prediction history with trends |
| Generic advice | Personalized doctor recommendations |
| No AI assistance | Claude AI health chatbot 24/7 |

---

## ✨ Key Features

### 🔬 Disease Prediction Modules

| Module | Model Type | Input | Output |
|---|---|---|---|
| 🫀 Heart Disease | Stacking Ensemble | Age, BMI, BP, Cholesterol, etc. | Risk % + Doctor |
| 🧠 Brain Stroke | ML Pipeline | Clinical features | Risk % + Doctor |
| 🩸 Diabetes | Stacking + Scaler | Glucose, BMI, Insulin, etc. | Risk % + Doctor |
| 🫘 Kidney Disease | Pipeline Classifier | Creatinine, BP, etc. | Category + Doctor |
| 👁️ Eye Disease | CNN (TensorFlow) | Retinal image upload | Disease class + Doctor |

### 🤖 AI Health Chatbot
- Powered by **Claude AI (Anthropic)**
- Answers any health question in **Gujarati, Hindi, English**
- Disease symptoms, causes, treatment — dynamic AI responses
- Fallback keyword-based answers when offline

### 📊 Additional Features
- **Google OAuth** + Email/Password login
- **Prediction History** — full log with timestamps
- **PDF Report** generation per prediction
- **BMI Calculator** with health advice
- **Lifestyle Recommender** — personalized diet, exercise, sleep tips
- **Health Score** based on vitals
- **Multi-language** support (English, Gujarati, Hindi)
- **Dark mode** UI
- **Trend graphs** per disease

---

## 🏗️ System Architecture

```
User (Browser)
      │
      ▼
Flask Web Server (app.py)
      │
      ├── /heart     → Stacking Ensemble Model (.pkl)
      ├── /brain     → Brain ML Pipeline (.pkl)
      ├── /diabetes  → Stacking + Scaler (.pkl)
      ├── /kidney    → Pipeline Classifier (.pkl)
      ├── /eye       → CNN Model (.h5) ← TensorFlow
      │
      ├── /api/chat  → Claude AI API (Anthropic)
      ├── /api/bmi   → BMI Calculator
      ├── /api/recommend → Lifestyle Engine
      ├── /api/report    → PDF Generator (ReportLab)
      ├── /api/history   → In-memory Session Store
      │
      └── Google OAuth 2.0 Authentication
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime |
| **Web Framework** | Flask 3.0 | Backend server & routing |
| **ML Models** | Scikit-learn | Heart, Brain, Diabetes, Kidney prediction |
| **Deep Learning** | TensorFlow / Keras | Eye CNN model |
| **AI Chatbot** | Anthropic Claude API | Dynamic health Q&A |
| **Authentication** | Google OAuth 2.0 + Authlib | User login |
| **PDF Reports** | ReportLab | Downloadable health reports |
| **Data** | Pandas, NumPy | Data processing |
| **Frontend** | HTML, CSS, JavaScript | Web UI |
| **Environment** | python-dotenv | Credential management |

---

## 📁 Project Structure

```
VitalsAI/
│
├── app.py                          # Main Flask application
│
├── models/
│   ├── final_stacking_model.pkl    # Heart disease model
│   ├── brain_ml_model.pkl          # Brain stroke model
│   ├── brain_features.pkl
│   ├── brain_selector.pkl
│   ├── diabetes_stack_model.pkl    # Diabetes model
│   ├── diabetes_scaler.pkl
│   ├── kidney_pipeline.pkl         # Kidney disease model
│   ├── kidney_features.pkl
│   ├── kidney_target_map.pkl
│   ├── eye_cnn_model.h5            # Eye CNN (TensorFlow)
│   └── eye_class_indices.json
│
├── templates/
│   ├── index.html                  # Dashboard
│   ├── login.html                  # Login page
│   ├── heart.html                  # Heart prediction
│   ├── brain.html                  # Brain prediction
│   ├── diabetes.html               # Diabetes prediction
│   ├── kidney.html                 # Kidney prediction
│   ├── eye.html                    # Eye prediction
│   ├── assistant.html              # AI Health Chatbot
│   ├── history.html                # Prediction history
│   ├── bmi.html                    # BMI calculator
│   └── lifestyle.html              # Lifestyle recommender
│
├── .env                            # API keys (never committed)
├── .env.example                    # Template for setup
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

**Estimated setup time: ~5 minutes**

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Poojanpatel12/VitalsAI.git
cd VitalsAI
```

### Step 2 — Create Virtual Environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```env
# Anthropic Claude AI (for chatbot)
ANTHROPIC_API_KEY=your-anthropic-key-here

# Google OAuth (for login)
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
```

> ✅ The app runs without API keys — chatbot falls back to keyword mode, Google login falls back to email/password.

### Step 5 — Run the App

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🔐 API Keys Setup

### Anthropic API Key (for AI Chatbot)
1. Go to **console.anthropic.com**
2. Sign up / Login
3. Go to **API Keys** → Create Key
4. Add to `.env` as `ANTHROPIC_API_KEY`

### Google OAuth (for Google Login)
1. Go to **console.cloud.google.com**
2. Create project → Enable Google OAuth API
3. Create OAuth 2.0 credentials
4. Add redirect URI: `http://localhost:5000/auth/google/callback`
5. Add Client ID & Secret to `.env`

---

## 📸 Disease Modules

### 🫀 Heart Disease Prediction
Uses a **Stacking Ensemble** model trained on CDC BRFSS dataset. Inputs include Age, BMI, Blood Pressure, Cholesterol, Diabetes history, Smoking, Physical Activity, and General Health score.

### 🧠 Brain Stroke Risk
ML pipeline trained on clinical stroke dataset. Uses feature selection to identify high-risk patients based on neurological and lifestyle indicators.

### 🩸 Diabetes Prediction
Stacking classifier with StandardScaler trained on Pima Indians Diabetes dataset. Uses 8 clinical features including Glucose, Insulin, and BMI.

### 🫘 Kidney Disease Classification
Multi-class pipeline classifying into: `No_Disease`, `Low_Risk`, `Moderate_Risk`, `High_Risk`, `Severe_Disease` — with confidence scores for each class.

### 👁️ Eye Disease Detection
CNN model trained on retinal images. Detects: `Cataracts`, `Glaucoma`, `Bulging_Eyes`, `Crossed_Eyes`, `Uveitis`, `Eye_diseases`.

---

## 🤖 AI Chatbot

The assistant page (`/assistant`) features a full-screen chat interface powered by **Claude AI**:

- **Claude AI Mode** — Dynamic, context-aware answers via Anthropic API
- **Quick Mode** — Fast keyword-based local answers (works offline)
- Supports **Gujarati, Hindi, English** — auto-detects language
- Structured responses: Symptoms → Causes → Treatment → When to see a doctor
- Conversation history maintained per session

---

## 📊 Sample Prediction Output

```json
{
  "risk": "HIGH RISK",
  "probability": 78.4,
  "doctor": "Cardiologist (હૃદય રોગ નિષ્ણાત)",
  "inputs": {
    "Age": 55,
    "BMI": 31.2,
    "HighBP": 1,
    "HighChol": 1,
    "Smoker": 1
  }
}
```

---

## 🏥 Doctor Recommendation System

Each prediction includes a recommended specialist:

| Risk Level | Heart | Brain | Diabetes | Kidney |
|---|---|---|---|---|
| HIGH RISK | Cardiologist | Neurologist | Endocrinologist | Nephrologist |
| MEDIUM RISK | General Physician | General Physician | General Physician | GP + Referral |
| LOW RISK | Annual Checkup | Lifestyle Check | Diet Consultation | Annual Test |

---

## 🌐 Available Routes

| Route | Description |
|---|---|
| `/` | Main dashboard |
| `/login` | Login page |
| `/heart` | Heart disease prediction |
| `/brain` | Brain stroke risk |
| `/diabetes` | Diabetes prediction |
| `/kidney` | Kidney disease detection |
| `/eye` | Eye disease classification |
| `/assistant` | AI health chatbot |
| `/history` | Prediction history |
| `/bmi` | BMI calculator |
| `/lifestyle` | Lifestyle recommender |
| `/status` | Model & feature status |
| `/api/chat` | Chatbot API endpoint |
| `/api/report` | PDF report generator |

---

## 🔮 Future Improvements

- Deploy on cloud (AWS / Render / Railway)
- Mobile app (Flutter / React Native)
- More disease modules (Liver, Thyroid, Cancer screening)
- Wearable device integration (Apple Watch, Fitbit)
- Patient account persistence (database instead of in-memory)
- Doctor portal for reviewing patient predictions
- Real-time health monitoring dashboard

---

## 👩‍💻 Author

| Field | Detail |
|---|---|
| **Name** | Poojan Patel |
| **College** | SAL Institute of Technology and Engineering Research |
| **Department** | Information & Communication Technology (ICT) |
| **University** | Gujarat Technological University (GTU) |
| **Year** | Final Year — 2026 |
| **GitHub** | [@Poojanpatel12](https://github.com/Poojanpatel12) |

---

## 📄 License

This project is distributed under the **MIT License**.

---

<div align="center">

⭐ **If this project was useful, please consider starring the repository!**

*Built with ❤️ by Poojan Patel · Final Year Project · SAL Institute of Technology · GTU 2026*

</div>
