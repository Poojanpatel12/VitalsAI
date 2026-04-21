# VitalsAI

An AI-powered multi-disease health risk prediction platform built with Flask and machine learning models.

## Overview

VitalsAI provides instant disease risk assessment for five medical conditions. Developed as a final year project at Ahmedabad Institute of Technology (GTU, 2026), it combines machine learning models with a web-based interface for accessible health screening.

## Features

| Module | Model | Input | Output |
|--------|-------|-------|--------|
| Heart Disease | Stacking Ensemble (RF + XGBoost + Logistic Regression) | Age, BMI, blood pressure, cholesterol, glucose, ECG data | Risk % + cardiologist recommendation |
| Brain Stroke | Stacking + Feature Selector | Age, glucose, BMI, smoking history, heart disease | Risk % + neurologist recommendation |
| Diabetes | Stacking + StandardScaler | Glucose, insulin, BMI, skin thickness, pedigree | Risk % + endocrinologist recommendation |
| Kidney Disease | Pipeline Classifier | 42 clinical blood/urine markers | Stage (1-5) + nephrologist referral |
| Eye Disease | MobileNetV2 CNN | Retinal fundus image (upload) | Disease classification + ophthalmologist |
| Lung Cancer | Stacking + LabelEncoders | Age, gender, stage, smoking history, treatment | Risk % + oncologist referral |

**Additional Features:**
- Google OAuth 2.0 + email/password authentication
- AI health chatbot (Gemini API) with Gujarati/Hindi/English support
- Prediction history tracking
- PDF report generation
- BMI calculator with health score
- Lifestyle recommendations

## Tech Stack

- **Backend**: Flask 2.3.2, Python 3.10+
- **ML Models**: scikit-learn, XGBoost, TensorFlow/Keras (MobileNetV2)
- **Authentication**: Authlib (Google OAuth)
- **AI Chatbot**: Gemini API
- **PDF Reports**: ReportLab

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Poojanpatel12/VitalsAI.git
cd VitalsAI

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
GEMINI_API_KEY=your-gemini-api-key
```

### Run

```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Project Structure

```
VitalsAI/
├── app.py                 # Main Flask application
├── models/                # Pre-trained ML model files (.pkl, .h5)
├── templates/             # HTML templates (Jinja2)
├── datasets/              # Training datasets
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (not committed)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict/<disease>` | POST | Get prediction with XAI explanations (heart, brain, diabetes, kidney, eye, lung) |
| `/api/chat` | POST | AI health assistant (Gemini API) |
| `/api/history` | GET/POST | Prediction history management |
| `/api/bmi` | POST | BMI calculation |
| `/api/report` | POST | PDF report generation |
| `/status` | GET | Model loading status |

## Disease Prediction Input

### Heart Disease
- Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar
- Max Heart Rate, ST Depression, Slope, Vessels, Thalassemia

### Brain Stroke
- Age, Hypertension (0/1), Heart Disease (0/1), Ever Married
- Work Type, Residence Type, Average Glucose Level, BMI
- Smoking Status

### Diabetes
- Pregnancies, Glucose, Blood Pressure, Skin Thickness
- Insulin, BMI, Diabetes Pedigree Function, Age

### Kidney Disease
- 42 features including blood urea, serum creatinine, hemoglobin
- Red/white blood cell counts, serum sodium, potassium

### Eye Disease
- Upload retinal fundus image
- Supported formats: JPG, PNG, WebP

### Lung Cancer
- Gender, Age, Smoking, Yellow Fingers, Anxiety, Peer Pressure
- Chronic Disease, Fatigue, Allergy, Wheezing, Swallowing Difficulty

## Authentication

- **Email/Password**: In-memory user storage with SHA256 hashing
- **Google OAuth**: Sign in with Google account

Session-based authentication with protected routes requiring login.

## Limitations

- In-memory data storage (history lost on server restart)
- Single-user sessions per browser
- No persistent database

## Future Work

- PostgreSQL database integration
- Cloud deployment (AWS/Render)
- Mobile companion app
- Additional disease modules (liver, thyroid)
- Doctor portal for healthcare providers

## License

MIT License

## Author

**Poojan Patel**
- Affiliation: Ahmedabad Institute of Technology
- Department: Computer Engineering
- University: Gujarat Technological University (GTU)
- Year: Final Year — 2026
- GitHub: [@Poojanpatel12](https://github.com/Poojanpatel12)

---

*Built for the Final Year Project — Ahmedabad Institute of Technology, GTU 2026*
