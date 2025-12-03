# Troubleshooting

## Version mismatch warnings
Regenerate artifacts with the installed `scikit-learn` version:
```bash
python train_model.py
```

## Dark mode contrast
Theme-aware CSS is injected in `app.py:54–145`. Adjust accent and text colors if needed.

## Missing artifacts
Ensure `diabetes_model.pkl`, `scaler.pkl`, and `feature_names.pkl` exist at repo root.

## Encryption
If `MODEL_KEY` is set, artifacts must be encrypted with Fernet. Otherwise, leave unset for plain joblib files.
