from __future__ import annotations

import numpy as np
import pandas as pd

from .model import load_artifacts


def get_predictor():
    model, scaler, feature_names = load_artifacts()

    def predict(payload_array):
        X_df = pd.DataFrame([payload_array], columns=feature_names)
        X_scaled = scaler.transform(X_df)
        pred = int(model.predict(X_scaled)[0])
        proba = float(model.predict_proba(X_scaled)[0][1])
        return {"prediction": pred, "probability": proba, "feature_names": feature_names}

    return predict
