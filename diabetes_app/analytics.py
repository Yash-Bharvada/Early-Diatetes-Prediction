import json
import os
from typing import Any, Dict

import requests


def emit_event(name: str, payload: Dict[str, Any]):
    endpoint = os.getenv("ANALYTICS_ENDPOINT", "")
    if not endpoint:
        return
    try:
        headers = {"Content-Type": "application/json"}
        data = {"event": name, "payload": payload}
        requests.post(endpoint, headers=headers, data=json.dumps(data), timeout=2)
    except Exception:
        pass

