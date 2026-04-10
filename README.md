# VitalsAI — Setup & Run Guide

## Step 1 — Install dependencies
```bash
pip install flask pandas numpy scikit-learn xgboost imbalanced-learn joblib tensorflow werkzeug
```

## Step 2 — Copy model files to models/ folder

Download these from Google Drive (Colab saves them there):

| Disease | Files needed |
|---------|-------------|
| Heart   | `final_stacking_model.pkl` |
| Brain   | `brain_ml_model.pkl`, `brain_features.pkl`, `brain_selector.pkl` |
| Diabetes| `diabetes_stack_model.pkl`, `diabetes_scaler.pkl` |
| Kidney  | `kidney_pipeline.pkl`, `kidney_features.pkl`, `kidney_target_map.pkl` |
| Eye     | `eye_cnn_model.h5`, `eye_class_indices.json` (optional) |

All files go inside the `models/` folder.

## Step 3 — Run server
```bash
python app.py
```

## Step 4 — Open browser
```
http://localhost:5000
```

Check which models loaded:
```
http://localhost:5000/status
```
