"""JSON extraction helper."""

import json
import re


def parse_json(text):
    if text is None:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None
