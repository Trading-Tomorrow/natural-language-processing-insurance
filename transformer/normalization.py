import re

def normalize_speed(text: str) -> str:
    text = re.sub(
        r'\b(\d+(?:[.,]\d+)?)\s*km\s*/\s*h\b',
        r'<speed> \1 kmh',
        text,
        flags=re.IGNORECASE
    )
    return text
