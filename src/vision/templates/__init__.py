"""Digit templates for OCR recognition."""

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent
DIGITS_DIR = TEMPLATES_DIR / "digits"

# Template dimensions for standard digit recognition
TEMPLATE_WIDTH = 20
TEMPLATE_HEIGHT = 30
