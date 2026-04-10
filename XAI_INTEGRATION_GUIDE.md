# 🎯 Vitals AI - Explainable AI Integration Guide

## 📋 Overview
તમારા existing models માટે **Dynamic Explainable AI** implement કર્યું છે જે:
✅ **Why HIGH Risk?** - કયા factors વધુ અસર કર્યો
✅ **Why LOW Risk?** - સારું શું છે
✅ **Dynamic Suggestions** - Risk level પર આધારે બદલાય છે
✅ **Feature Importance** - ટોપ contributing factors દર્શાય છે
✅ **Lifestyle Changes** - Disease-specific recommendations

---

## 📁 Files Created

### 1. **Backend (Python)**
```
app_with_xai.py
```
**Key Functions:**
- `get_xai_explanation(disease, risk_level, inputs, probability)` 
  - Dynamic explanations based on risk level
  - Hindi/Gujarati support
  
- `calculate_feature_importance(disease, inputs)`
  - Shows which features contributed most
  - SHAP-style contribution percentages

**Modified API Endpoint:**
```python
@app.route('/api/predict/<disease>', methods=['POST'])
```
Now returns:
```json
{
  "risk": "HIGH RISK",
  "probability": 78,
  "why_high_risk": [
    "🩸 તમારું બ્લડ પ્રેશર ઊંચું છે - BP હૃદય રોગનો મુખ્ય કારણ છે",
    "🚭 તમે ધૂમ્રપાન કરો છો - જોખમ 2-3x વધે છે"
  ],
  "why_low_risk": [],
  "key_factors": ["BP: ⚠️ HIGH (45%)", "Cholesterol: ⚠️ HIGH (32%)"],
  "suggestions": [
    "🏥 તાત્કાલિક Cardiologist મળો",
    "💊 દવા શરૂ કરો: Aspirin + Statin",
    "🍎 DASH Diet અનુસરો"
  ],
  "lifestyle_changes": {
    "diet": ["ઓછું સોડિયમ", "તાજું શાક"],
    "exercise": ["30 મિનિટ દર દિવસે"],
    "sleep": ["7-8 કલાક"]
  },
  "feature_importance": [
    {"feature": "બ્લડ પ્રેશર", "contribution_percent": 45, "value": 1},
    {"feature": "કોલેસ્ટ્રોલ", "contribution_percent": 32, "value": 1}
  ]
}
```

---

### 2. **Frontend (HTML)**

#### **heart_xai.html**
Dynamic Heart Disease Prediction with:
- ✅ Risk Level Display (HIGH/MEDIUM/LOW)
- ✅ Why HIGH Risk? (Top factors)
- ✅ Feature Importance Bar Charts
- ✅ Dynamic Suggestions based on Risk
- ✅ Lifestyle Changes (Diet, Exercise, Sleep)

#### **diabetes_xai.html**
Same structure for Diabetes module

---

## 🔧 Integration Steps

### Step 1: Replace Your Backend
```bash
# Backup your original
cp app.py app_backup.py

# Replace with XAI version
cp app_with_xai.py app.py
```

### Step 2: Update HTML Pages
For each disease page, use the new XAI template:

```html
<!-- Add this div for dynamic results -->
<div id="result" class="result-box">
  <div class="risk-indicator" id="risk-ind"></div>
  
  <!-- Why HIGH RISK -->
  <div id="exp-high" class="exp-section high">
    <h3>⚠️ Why HIGH Risk?</h3>
    <ul class="exp-list" id="high-list"></ul>
  </div>
  
  <!-- Feature Importance -->
  <div class="features">
    <h3>📊 Top Contributing Factors</h3>
    <div id="features"></div>
  </div>
  
  <!-- Suggestions -->
  <div id="suggestions" class="suggestions"></div>
  
  <!-- Lifestyle -->
  <div id="lifestyle" class="lifestyle"></div>
</div>
```

### Step 3: Update JavaScript
```javascript
// In your form submission handler:
async function predictDisease(disease) {
  const data = { /* your form data */ };
  
  const res = await fetch(`/api/predict/${disease}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
  });
  
  const result = await res.json();
  
  // Display all XAI data
  displayXAIResult(result);
}

function displayXAIResult(result) {
  // Risk indicator
  document.getElementById('risk-ind').innerHTML = `
    <div class="risk-level">${result.risk}</div>
    <div>Probability: ${result.probability}%</div>
  `;
  
  // Why HIGH RISK
  if (result.why_high_risk?.length > 0) {
    document.getElementById('exp-high').innerHTML = `
      <h3>⚠️ Why HIGH Risk?</h3>
      <ul class="exp-list">
        ${result.why_high_risk.map(e => `<li>${e}</li>`).join('')}
      </ul>
    `;
  }
  
  // Feature importance
  if (result.feature_importance) {
    let html = '';
    result.feature_importance.forEach(f => {
      html += `
        <div class="feature-item">
          <div>${f.feature}: ${f.contribution_percent}%</div>
          <div class="bar" style="width:${f.contribution_percent}%"></div>
        </div>
      `;
    });
    document.getElementById('features').innerHTML = html;
  }
  
  // Suggestions
  if (result.suggestions?.length > 0) {
    document.getElementById('suggestions').innerHTML = `
      <h3>💊 Recommended Actions</h3>
      <ul>
        ${result.suggestions.map(s => `<li>${s}</li>`).join('')}
      </ul>
    `;
  }
  
  // Show result
  document.getElementById('result').classList.add('show');
}
```

---

## 📊 XAI Explanations by Disease

### ❤️ Heart Disease
**HIGH RISK Factors:**
- Blood Pressure (45% contribution)
- Cholesterol (32%)
- Smoking (25%)
- BMI (18%)
- Physical Activity (12%)

**Dynamic Suggestions:**
- `if BP high` → "Cardiologist + Statin drug"
- `if Smoker` → "Smoking cessation program"
- `if BMI > 30` → "Weight loss 5-10kg = 58% risk reduction"

### 🩺 Diabetes
**HIGH RISK Factors:**
- Glucose Level (48%)
- BMI (45%)
- Age (20%)
- Pregnancies (15%)

**Dynamic Suggestions:**
- `if Glucose > 200` → "Type 2 Diabetes present, start Insulin"
- `if BMI > 30` → "Weight loss is #1 priority"
- `if Age > 50` → "Age-appropriate screening"

### 🧠 Brain/Stroke
**HIGH RISK Factors:**
- Blood Pressure (40%)
- Age (30%)
- Cholesterol (25%)
- Smoking (20%)

**Dynamic Suggestions:**
- `if AF present` → "Anticoagulant (Warfarin) required"
- `if TIA history` → "Urgent Neurologist consultation"
- `if Age > 60` → "Enhanced monitoring needed"

### 🫘 Kidney Disease
**Disease Stages Based on GFR:**
- GFR < 15: **Severe Disease** → Dialysis required
- GFR 15-30: **High Risk** → Nephrologist consultation
- GFR 30-60: **Moderate Risk** → Conservative management
- GFR > 60: **Low Risk** → Regular monitoring

---

## 🎨 UI Components

### Risk Indicator
```html
<div class="risk-indicator">
  <div class="risk-level high">HIGH RISK</div>
  <div class="risk-prob">Probability: 78%</div>
</div>
```

### Feature Importance Bar
```html
<div class="feature-item">
  <div class="feature-label">
    <span>Blood Pressure</span>
    <span>45%</span>
  </div>
  <div class="feature-bar">
    <div class="feature-bar-fill" style="width:45%"></div>
  </div>
</div>
```

### Suggestions Box
```html
<div class="suggestions">
  <h3>💊 Recommended Actions</h3>
  <ul class="sug-list">
    <li>✓ તાત્કાલિક Cardiologist મળો</li>
    <li>✓ Aspirin + Statin શરૂ કરો</li>
  </ul>
</div>
```

---

## 🚀 Advanced Features (Optional)

### 1. SHAP Values Calculation
```python
# Add to app_with_xai.py
import shap

def add_shap_explanations(disease, model, X):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return {
        'base_value': explainer.expected_value,
        'shap_values': shap_values[0],
        'feature_impact': dict(zip(feature_names, shap_values[0]))
    }
```

### 2. Predictive Timeline
```python
# Show how risk changes over time
def predict_future_risk(disease, current_inputs, months=6):
    # Simulate lifestyle improvements
    improved_inputs = current_inputs.copy()
    
    if disease == 'diabetes':
        improved_inputs['BMI'] -= 2  # 5kg weight loss
        improved_inputs['Glucose'] -= 15  # Diet improvement
    
    future_risk = model.predict_proba(improved_inputs)
    return {
        'current_risk': current_risk,
        'future_risk': future_risk,
        'improvement_potential': current_risk - future_risk
    }
```

### 3. Comparison with Population
```python
def compare_with_population(disease, inputs, population_avg):
    user_risk = get_risk(inputs)
    avg_risk = population_avg[disease]
    
    return {
        'user_risk': user_risk,
        'average_risk': avg_risk,
        'relative_risk': user_risk / avg_risk,
        'percentile': calculate_percentile(user_risk, disease)
    }
```

---

## 📱 Testing

### Test Case 1: High Risk Heart Patient
```python
data = {
    'Age': 58,
    'BMI': 31,
    'HighBP': 1,  # Yes
    'HighChol': 1,  # Yes
    'Smoker': 1,  # Yes
    'PhysActivity': 0,  # No
    'Diabetes': 1,  # Yes
}
```

**Expected:**
- ✅ Risk: HIGH RISK (78%)
- ✅ Shows all 6 why_high_risk factors
- ✅ Suggestions include: Cardiologist, Aspirin+Statin, DASH Diet
- ✅ Lifestyle changes include: Quit smoking, 150 min exercise

### Test Case 2: Low Risk Diabetes Patient
```python
data = {
    'Glucose': 95,  # Normal
    'BMI': 23,  # Normal
    'Age': 30,
    'Pregnancies': 0
}
```

**Expected:**
- ✅ Risk: LOW RISK (15%)
- ✅ why_low_risk shows positive factors
- ✅ Suggestions: Regular monitoring, maintain lifestyle

---

## 🎯 Deployment

### Docker Setup
```dockerfile
FROM python:3.8
WORKDIR /app
COPY app_with_xai.py .
COPY models/ models/
COPY templates/ templates/
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python", "app_with_xai.py"]
```

### Heroku Deployment
```bash
git add .
git commit -m "Add XAI to Vitals AI"
git push heroku main
```

---

## 📚 Reference URLs

- **SHAP Documentation**: https://shap.readthedocs.io/
- **XAI Papers**: https://arxiv.org/list/stat.ML/recent
- **Flask Documentation**: https://flask.palletsprojects.com/

---

## ✅ Checklist

- [ ] Replace app.py with app_with_xai.py
- [ ] Update heart.html with heart_xai.html
- [ ] Update diabetes.html with diabetes_xai.html
- [ ] Update brain.html (same pattern as heart)
- [ ] Update kidney.html (same pattern)
- [ ] Test all prediction endpoints
- [ ] Verify XAI explanations in Hindi/Gujarati
- [ ] Check responsive design on mobile
- [ ] Deploy to production

---

## 🆘 Troubleshooting

**Q: API returns empty explanations?**
A: Check that `get_xai_explanation()` is handling all risk levels for your disease

**Q: Feature importance shows wrong percentages?**
A: Verify that feature scaling in `calculate_feature_importance()` matches your model training

**Q: Suggestions not showing?**
A: Make sure `result.suggestions` exists in API response

---

**Created by:** VitalsAI XAI Team
**Date:** 2026-04-09
**Version:** 1.0
