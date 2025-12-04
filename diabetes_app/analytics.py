import json
import os
from typing import Any

import requests


def emit_event(name: str, payload: dict[str, Any]):
    endpoint = os.getenv("ANALYTICS_ENDPOINT", "")
    if not endpoint:
        return
    try:
        headers = {"Content-Type": "application/json"}
        data = {"event": name, "payload": payload}
        requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=2)
    except Exception:
        pass
