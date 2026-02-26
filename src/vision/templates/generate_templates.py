#!/usr/bin/env python3
"""Generate digit templates for HUD OCR recognition.

Creates 20x30 pixel grayscale templates for digits 0-9 in a HUD-style font.
Run this script to regenerate templates if needed.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision.templates.patterns import (
    DIGIT_PATTERNS,
    PATTERN_GRID_HEIGHT,
    PATTERN_GRID_WIDTH,
)


def create_digit_template(
    digit: int, width: int = 20, height: int = 30, font_size: int = 24
) -> np.ndarray:
    """Create a template image for a single digit using font rendering.

    Args:
        digit: The digit (0-9) to create a template for
        width: Template width in pixels
        height: Template height in pixels
        font_size: Font size to use

    Returns:
        Grayscale template as numpy array (white digit on black background)
    """
    # Create black background
    img = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(img)

    # Try to use a built-in font, fall back to default if not available
    try:
        # Try common monospace fonts that look like HUD text
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Monaco.ttf", font_size)
        except OSError:
            # Fall back to default font
            font = ImageFont.load_default()

    # Get text bounding box for centering
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Center the digit
    x = (width - text_width) // 2 - bbox[0]
    y = (height - text_height) // 2 - bbox[1]

    # Draw white digit
    draw.text((x, y), text, fill=255, font=font)

    return np.array(img)


def generate_all_templates(output_dir: Path) -> None:
    """Generate templates for all digits 0-9 using font rendering.

    Args:
        output_dir: Directory to save templates
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for digit in range(10):
        template = create_digit_template(digit)
        output_path = output_dir / f"{digit}.png"
        Image.fromarray(template).save(output_path)
        print(f"Generated template for digit {digit}: {output_path}")


def create_synthetic_templates(output_dir: Path, width: int = 20, height: int = 30) -> None:
    """Create synthetic templates using hand-crafted patterns.

    This creates more game-like digit patterns when font rendering
    doesn't produce ideal results.

    Args:
        output_dir: Directory to save templates
        width: Template width in pixels
        height: Template height in pixels
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_x = width / PATTERN_GRID_WIDTH
    scale_y = height / PATTERN_GRID_HEIGHT

    for digit, pattern in DIGIT_PATTERNS.items():
        img = np.zeros((height, width), dtype=np.uint8)
        for gx, gy in pattern:
            # Scale grid coordinates to pixel coordinates
            x_start = int(gx * scale_x)
            x_end = int((gx + 1) * scale_x)
            y_start = int(gy * scale_y)
            y_end = int((gy + 1) * scale_y)
            img[y_start:y_end, x_start:x_end] = 255

        output_path = output_dir / f"{digit}.png"
        Image.fromarray(img).save(output_path)
        print(f"Generated synthetic template for digit {digit}: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate digit templates for OCR")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic templates instead of font-based",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "digits",
        help="Output directory for templates",
    )
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_templates(args.output)
    else:
        generate_all_templates(args.output)
