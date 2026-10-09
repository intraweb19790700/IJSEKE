"""Local redaction utilities for synthetic PII."""
import re

SYNTHETIC_PII_PATTERNS = [
    (re.compile(r'\+1-555-\d{4}'), '[PHONE]'),
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'PID-\d{6}'), '[PATIENT_ID]'),
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), '[SSN]'),
]

def redact_text(text):
    redacted = text
    for pattern, replacement in SYNTHETIC_PII_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted

def contains_pii(text):
    for pattern, _ in SYNTHETIC_PII_PATTERNS:
        if pattern.search(text):
            return True
    return False
