#!/usr/bin/env python3
"""Create a simple test video for game-study."""

import cv2
import numpy as np
from pathlib import Path

def create_test_video(output_path: str = "samples/test_video.mp4", duration_sec: int = 5):
    """
    Create a simple test video with some HUD-like elements.

    Args:
        output_path: Output video path
        duration_sec: Video duration in seconds
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Video settings
    width, height = 1920, 1080
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Create frames
    total_frames = duration_sec * fps

    for frame_idx in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some HUD-like elements

        # HP bar (bottom left)
        hp_value = max(0, 100 - frame_idx // 3)
        cv2.rectangle(frame, (50, 900), (350, 950), (0, 255, 0), 2)
        cv2.putText(frame, f"HP: {hp_value}", (60, 940),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Shield bar (bottom left, below HP)
        shield_value = max(0, 50 - frame_idx // 6)
        cv2.rectangle(frame, (50, 960), (350, 1010), (0, 100, 255), 2)
        cv2.putText(frame, f"Shield: {shield_value}", (60, 1000),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Minimap (top right)
        cv2.circle(frame, (1700, 150), 100, (100, 100, 100), 2)
        cv2.putText(frame, "Minimap", (1650, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Weapon (bottom right)
        weapon_names = ["Assault Rifle", "Shotgun", "Pistol"]
        weapon_idx = (frame_idx // fps) % len(weapon_names)
        cv2.putText(frame, weapon_names[weapon_idx], (1700, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Ammo: 30", (1700, 990),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (900, 540),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"✅ Test video created: {output_path}")
    print(f"   Duration: {duration_sec}s, FPS: {fps}, Frames: {total_frames}")

if __name__ == "__main__":
    create_test_video()
