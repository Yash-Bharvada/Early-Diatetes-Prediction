# Deployment Guide

## Streamlit Cloud
- Push repo with `requirements.txt`, artifacts (`diabetes_model.pkl`, `scaler.pkl`, `feature_names.pkl`).
- Create app targeting `app.py`.
- Set environment variables if using encryption:
  - `MODEL_KEY` – Fernet key for encrypted joblib files.

## GitHub Actions
- CI workflow runs lint and tests on PRs and pushes to main.
- Extend with deployment steps (e.g., to Streamlit Cloud or a container registry) as needed.

## Local
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
