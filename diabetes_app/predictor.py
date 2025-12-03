from __future__ import annotations

import numpy as np
from typing import Dict

from .model import load_artifacts


def get_predictor():
    model, scaler, feature_names = load_artifacts()

    def predict(payload_array):
        X = np.array(payload_array).reshape(1, -1)
        X_scaled = scaler.transform(X)
        pred = int(model.predict(X_scaled)[0])
        proba = float(model.predict_proba(X_scaled)[0][1])
        return {"prediction": pred, "probability": proba, "feature_names": feature_names}

    return predict

