import json


def extract_field(event, field):
    if not isinstance(event, dict):
        return None

    if field in event:
        return event[field]

    body = event.get("body")
    if not body:
        return None

    try:
        payload = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        return None

    if isinstance(payload, dict):
        return payload.get(field)

    return None


def extract_index(event):
    return extract_field(event, "index")
