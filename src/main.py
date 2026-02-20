#!/usr/bin/env python3
"""Main entry point for game-study."""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from capture.video_file import VideoFileCapture
from vision.roi import ROIExtractor
from vision.anchors import AnchorDetector
from vision.yolo_detector import YOLODetector
from vision.ocr import OCRDetector
from vision.state_builder import StateBuilder
from trigger.engine import TriggerEngine
from dialogue.templates import DialogueTemplateManager
from dialogue.openai_client import OpenAIClient
from dialogue.realtime_client import RealtimeVoiceClient
from utils.logger import SessionLogger
from utils.time import get_timestamp_ms
from constants import (
    DEFAULT_ROI_CONFIG,
    DEFAULT_TRIGGERS_CONFIG,
    DEFAULT_SYSTEM_PROMPT,
    LOG_INTERVAL_FRAMES,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fortnite AI Coach for English Learning"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        choices=["video"],
        help="Input type"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--triggers",
        type=str,
        default=DEFAULT_TRIGGERS_CONFIG,
        help="Path to triggers config file"
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=DEFAULT_ROI_CONFIG,
        help="Path to ROI config file"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for session logs"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice output (Realtime API)"
    )
    parser.add_argument(
        "--voice-model",
        type=str,
        default="tts-1",
        choices=["tts-1", "tts-1-hd"],
        help="Voice model to use"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize logger
    logger = SessionLogger(args.out)

    # Load configurations
    roi_config_path = Path(args.roi)
    triggers_config_path = Path(args.triggers)
    system_prompt_path = Path(args.system_prompt)

    if not roi_config_path.exists():
        logger.error(f"ROI config not found: {roi_config_path}")
        sys.exit(1)

    if not triggers_config_path.exists():
        logger.error(f"Triggers config not found: {triggers_config_path}")
        sys.exit(1)

    # Initialize components
    logger.info("Initializing components...")

    # Vision components
    roi_extractor = ROIExtractor(str(roi_config_path))
    anchor_detector = AnchorDetector()
    yolo_detector = YOLODetector()  # MVP: no model
    ocr_detector = OCRDetector(use_template_matching=True)
    state_builder = StateBuilder()

    # Trigger engine
    logger.info("Loading trigger rules...")
    trigger_engine = TriggerEngine(str(triggers_config_path))

    # Dialogue
    template_manager = DialogueTemplateManager()

    # OpenAI client (optional - can work without API key)
    try:
        openai_client = OpenAIClient(
            system_prompt_path=str(system_prompt_path)
        )
        logger.info("OpenAI client initialized.")
    except ValueError as e:
        logger.warning(f"OpenAI client not initialized: {e}")
        logger.info("Continuing with template-only mode.")
        openai_client = None

    # Realtime voice client (optional - Phase 2)
    voice_client = None
    if args.voice:
        try:
            voice_client = RealtimeVoiceClient(
                system_prompt_path=str(system_prompt_path),
                enable_audio_output=True
            )
            logger.info("Realtime voice client initialized.")
        except Exception as e:
            logger.warning(f"Voice client not initialized: {e}")
            logger.info("Continuing with text-only mode.")

    # Video capture
    if args.input == "video":
        if not args.video:
            logger.error("--video argument required for video input")
            sys.exit(1)

        if not Path(args.video).exists():
            logger.error(f"Video file not found: {args.video}")
            sys.exit(1)

        logger.info(f"Opening video: {args.video}")
        with VideoFileCapture(args.video) as capture:
            metadata = capture.get_metadata()
            logger.info(f"Video metadata: {metadata}")

            # Process frames
            logger.info("Processing frames...")
            logger.info("Press Ctrl+C to stop")

            frame_count = 0
            total_triggers = 0

            try:
                for frame in capture:
                    frame_count += 1

                    # Extract ROIs
                    rois = roi_extractor.extract_all_rois(frame)

                    # Build state from vision
                    # HP detection
                    hp_roi = rois.get("hp_shield")
                    if hp_roi is not None and hp_roi.size > 0:
                        try:
                            hp_result = ocr_detector.extract_hp(hp_roi)
                            state_builder.update_hp(
                                hp_result["value"],
                                hp_result["source"],
                                hp_result["confidence"]
                            )
                        except Exception as e:
                            logger.log_error(f"HP detection error: {e}", e)

                    # Shield detection (simplified for MVP)
                    # In full implementation, would extract from hp_shield ROI

                    # Knocked detection
                    knocked_roi = rois.get("knocked_revive")
                    if knocked_roi is not None:
                        knocked_result = yolo_detector.detect_knocked_status(knocked_roi)
                        state_builder.update_knocked(
                            knocked_result["value"],
                            knocked_result["source"],
                            knocked_result["confidence"]
                        )

                    # Get current state
                    state = state_builder.get_state()
                    movement_state = state_builder.get_movement_state()

                    # Log state
                    state["frame_number"] = frame_count
                    logger.log_state(state)

                    # Evaluate triggers
                    trigger_result = trigger_engine.evaluate_triggers(state, movement_state)

                    if trigger_result:
                        total_triggers += 1
                        logger.info(f"[{frame_count}] Trigger: {trigger_result['rule_name']}")

                        # Generate response
                        template = trigger_result['template']
                        response_text = template

                        # Try OpenAI enhancement if available
                        if openai_client:
                            try:
                                response_text = openai_client.generate_response(
                                    trigger_result,
                                    state,
                                    movement_state
                                )
                            except Exception as e:
                                logger.warning(f"OpenAI error: {e}, using template")

                        logger.info(f"  Response: {response_text}")

                        # Voice output (Phase 2)
                        voice_response = None
                        if voice_client:
                            try:
                                voice_response = voice_client.speak_with_trigger(
                                    trigger_result,
                                    state,
                                    movement_state
                                )
                                if voice_response:
                                    logger.debug(f"  Voice: generated ({voice_response.duration_ms}ms)")
                            except Exception as e:
                                logger.warning(f"Voice error: {e}")

                        # Log trigger and response
                        logger.log_trigger({
                            "trigger": trigger_result,
                            "response": response_text,
                            "voice_duration_ms": voice_response.duration_ms if voice_response else None,
                        })

                        logger.log_response({
                            "trigger_id": trigger_result["rule_id"],
                            "trigger_name": trigger_result["rule_name"],
                            "priority": trigger_result["priority"],
                            "response": response_text,
                            "movement_state": movement_state,
                            "timestamp_ms": trigger_result["timestamp_ms"],
                            "voice_duration_ms": voice_response.duration_ms if voice_response else None,
                        })

                    # Progress update every N frames
                    if frame_count % LOG_INTERVAL_FRAMES == 0:
                        logger.info(f"Processed {frame_count} frames, {total_triggers} triggers")

            except KeyboardInterrupt:
                logger.info(f"Stopped at frame {frame_count}")

            logger.info("Processing complete!")
            logger.info(f"Total frames: {frame_count}")
            logger.info(f"Total triggers: {total_triggers}")
            logger.info(f"Logs saved to: {args.out}")


if __name__ == "__main__":
    main()
