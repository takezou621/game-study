#!/usr/bin/env python3
"""Generate digit templates for HUD OCR recognition.

Creates 20x30 pixel grayscale templates for digits 0-9 in a HUD-style font.
Run this script to regenerate templates if needed.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_digit_template(
    digit: int, width: int = 20, height: int = 30, font_size: int = 24
) -> np.ndarray:
    """Create a template image for a single digit.

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
    """Generate templates for all digits 0-9.

    Args:
        output_dir: Directory to save templates
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for digit in range(10):
        template = create_digit_template(digit)
        output_path = output_dir / f"{digit}.png"
        Image.fromarray(template).save(output_path)
        print(f"Generated template for digit {digit}: {output_path}")


def create_synthetic_templates(output_dir: Path) -> None:
    """Create synthetic templates using hand-crafted patterns.

    This creates more game-like digit patterns when font rendering
    doesn't produce ideal results.

    Args:
        output_dir: Directory to save templates
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define digit patterns as lists of (x, y) coordinates
    # Each digit is defined in a 5x7 grid, scaled to 20x30
    digit_patterns = {
        0: [
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (4, 1),
            (0, 2),
            (4, 2),
            (0, 3),
            (4, 3),
            (0, 4),
            (4, 4),
            (0, 5),
            (4, 5),
            (0, 6),
            (1, 6),
            (2, 6),
            (3, 6),
            (4, 6),
        ],
        1: [(2, 0), (1, 1), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (1, 6), (2, 6), (3, 6)],
        2: [
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 1),
            (4, 2),
            (3, 3),
            (2, 4),
            (1, 5),
            (0, 6),
            (1, 6),
            (2, 6),
            (3, 6),
            (4, 6),
        ],
        3: [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 1),
            (4, 2),
            (3, 3),
            (4, 4),
            (4, 5),
            (0, 6),
            (1, 6),
            (2, 6),
            (3, 6),
        ],
        4: [
            (0, 0),
            (4, 0),
            (0, 1),
            (4, 1),
            (0, 2),
            (4, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (4, 4),
            (4, 5),
            (4, 6),
        ],
        5: [
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 4),
            (4, 5),
            (0, 6),
            (1, 6),
            (2, 6),
            (3, 6),
        ],
        6: [
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (0, 5),
            (4, 5),
            (1, 6),
            (2, 6),
            (3, 6),
        ],
        7: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (4, 1), (3, 2), (3, 3), (2, 4), (2, 5), (2, 6)],
        8: [
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (4, 1),
            (0, 2),
            (4, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (0, 4),
            (4, 4),
            (0, 5),
            (4, 5),
            (1, 6),
            (2, 6),
            (3, 6),
        ],
        9: [
            (1, 0),
            (2, 0),
            (3, 0),
            (0, 1),
            (4, 1),
            (0, 2),
            (4, 2),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
            (4, 4),
            (4, 5),
            (1, 6),
            (2, 6),
            (3, 6),
        ],
    }

    grid_width, grid_height = 5, 7
    scale_x = 20 / grid_width
    scale_y = 30 / grid_height

    for digit, pattern in digit_patterns.items():
        img = np.zeros((30, 20), dtype=np.uint8)
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
