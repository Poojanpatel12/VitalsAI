# 🏥 VitalsAI — AI-Powered Health Prediction Platform

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

*Final Year Project · Ahmedabad Institute of Technology · GTU · 2026*

</div>

---

## 📸 Screenshots

### 🔐 Login Page
![Login](docs/images/login.png)

### 🏠 Dashboard
![Dashboard](docs/images/dashboard.png)

### ❤️ Heart Disease Prediction
<table><tr>
<td><img src="docs/images/heart_form.png"/></td>
<td><img src="docs/images/heart_result.png"/></td>
</tr><tr>
<td align="center">Input Form</td>
<td align="center">Result + PDF</td>
</tr></table>

### 🧠 Brain Stroke Prediction
<table><tr>
<td><img src="docs/images/brain_form.png"/></td>
<td><img src="docs/images/brain_result.png"/></td>
</tr><tr>
<td align="center">Input Form</td>
<td align="center">Risk Assessment</td>
</tr></table>

### 🩸 Diabetes Prediction
<table><tr>
<td><img src="docs/images/diabetes_form.png"/></td>
<td><img src="docs/images/diabetes_result.png"/></td>
</tr><tr>
<td align="center">Clinical Input</td>
<td align="center">Result + Tips</td>
</tr></table>

### 🫘 Kidney Disease Prediction
<table><tr>
<td><img src="docs/images/kidney_form.png"/></td>
<td><img src="docs/images/kidney_result.png"/></td>
</tr><tr>
<td align="center">42 Clinical Features</td>
<td align="center">5-Stage Classification</td>
</tr></table>

### 👁️ Eye Disease Classification
<table><tr>
<td><img src="docs/images/eye_form.png"/></td>
<td><img src="docs/images/eye_result.png"/></td>
</tr><tr>
<td align="center">Image Upload</td>
<td align="center">CNN Result</td>
</tr></table>

### 🤖 AI Health Chatbot
![Chatbot](docs/images/chatbot.png)

### 🌿 Lifestyle & ⚖️ BMI
<table><tr>
<td><img src="docs/images/lifestyle.png"/></td>
<td><img src="docs/images/bmi.png"/></td>
</tr><tr>
<td align="center">Health Plan + Score</td>
<td align="center">BMI Calculator</td>
</tr></table>

### 📋 Prediction History
![History](docs/images/history.png)

---

## 🩺 Overview

| Traditional | VitalsAI |
|---|---|
| Doctor visit required | Instant AI assessment |
| Single disease per visit | 5 modules in one platform |
| No history tracking | Full prediction history |
| Generic advice | Personalized recommendations |
| No AI | Claude AI chatbot 24/7 |

---

## ✨ Disease Modules

| Module | Model | Output |
|---|---|---|
| 🫀 Heart | Stacking (RF+XGBoost+LR) | Risk % + Doctor |
| 🧠 Brain | Stacking (RF+XGBoost+SVC) | Risk % + Doctor |
| 🩸 Diabetes | Stacking + Scaler | Risk % + Doctor |
| 🫘 Kidney | Pipeline — 5 Stage | Stage + Confidence |
| 👁️ Eye | CNN MobileNetV2 | Disease + Doctor |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10, Flask 3.0 |
| ML | Scikit-learn (RF, XGBoost, SVC) |
| Deep Learning | TensorFlow/Keras (MobileNetV2) |
| AI Chatbot | Anthropic Claude API |
| Auth | Google OAuth 2.0 |
| PDF | ReportLab |
| Frontend | HTML5, CSS3, Vanilla JS |

---

## ⚡ Quick Start
```bash
git clone https://github.com/Poojanpatel12/VitalsAI.git
cd VitalsAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
python app.py
```

---

## 👩‍💻 Author

| Field | Detail |
|---|---|
| **Name** | Poojan Patel |
| **College** | Ahmedabad Institute of Technology |
| **Department** | ICT |
| **University** | GTU — 2026 |
| **GitHub** | [@Poojanpatel12](https://github.com/Poojanpatel12) |

---

<div align="center">

⭐ **Star this repo if you found it useful!**

*Built with ❤️ by Poojan Patel · AIT · GTU 2026*

</div>
