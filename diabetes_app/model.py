from __future__ import annotations

import io
import logging
from typing import Tuple, List

import joblib

from .config import MODEL_PATH, SCALER_PATH, FEATURES_PATH, MODEL_KEY

logger = logging.getLogger(__name__)


def _load_plain_joblib(path: str):
    return joblib.load(path)


def _load_encrypted_joblib(path: str, key: str):
    from cryptography.fernet import Fernet

    f = Fernet(key)
    with open(path, "rb") as fh:
        data = fh.read()
    decrypted = f.decrypt(data)
    buf = io.BytesIO(decrypted)
    return joblib.load(buf)


def load_artifacts() -> Tuple[object, object, List[str]]:
    use_encryption = bool(MODEL_KEY)
    loader = _load_encrypted_joblib if use_encryption else _load_plain_joblib
    logger.info("Loading artifacts (encrypted=%s)", use_encryption)

    model = loader(MODEL_PATH, MODEL_KEY) if use_encryption else loader(MODEL_PATH)
    scaler = loader(SCALER_PATH, MODEL_KEY) if use_encryption else loader(SCALER_PATH)
    feature_names: List[str] = loader(FEATURES_PATH, MODEL_KEY) if use_encryption else loader(FEATURES_PATH)
    return model, scaler, feature_names

