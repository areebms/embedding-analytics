import json


def extract_index(event):
    if not isinstance(event, dict):
        return None

    if "index" in event:
        return event["index"]

    body = event.get("body")
    if not body:
        return None

    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(payload, dict):
        return payload.get("index")

    return None
