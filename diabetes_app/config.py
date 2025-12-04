import os

MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "feature_names.pkl")

# Optional encryption key for encrypted artifacts (Fernet key)
MODEL_KEY = os.getenv("MODEL_KEY", "")
