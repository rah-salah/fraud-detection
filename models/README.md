# Model Artifacts

Models trained locally - too large for GitHub (500MB+).

## E-Commerce Fraud Models:
- lr_fraud.pkl - Logistic Regression - F1=0.27, AUC-PR=0.42
- rf_fraud.pkl - Random Forest - F1=0.66, AUC-PR=0.62 - BEST MODEL

## Credit Card Models:
- lr_credit.pkl - Logistic Regression - F1=0.20, AUC-PR=0.70
- rf_credit.pkl - Random Forest - BEST MODEL

## To regenerate:
Run: PYTHONPATH=. python src/train.py
