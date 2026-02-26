"""Main CLI entry point."""

import argparse
import logging
import sys
from typing import Any

from infrastructure.config.settings import Settings, load_settings
from infrastructure.exceptions import InfrastructureError
from presentation.exceptions import CLIError, UserInputError
from utils.logger import get_logger

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="AI Game Coach - Real-time voice coaching for gamers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with video file
  python -m src.main --input video --video samples/gameplay.mp4

  # Run with screen capture
  python -m src.main --input screen

  # Run with custom config
  python -m src.main --config configs/custom.yaml
        """,
    )

    # Input options
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--input",
        "-i",
        choices=["video", "screen"],
        default="video",
        help="Input source type (default: video)",
    )
    input_group.add_argument(
        "--video",
        type=str,
        help="Path to video file (required if --input=video)",
    )
    input_group.add_argument(
        "--screen",
        type=int,
        default=1,
        help="Monitor index for screen capture (default: 1)",
    )
    input_group.add_argument(
        "--region",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Screen capture region (x y width height)",
    )
    input_group.add_argument(
        "--loop",
        action="store_true",
        help="Loop video when it ends",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--out",
        "-o",
        type=str,
        default="./logs",
        help="Output directory for logs and results (default: ./logs)",
    )
    output_group.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    output_group.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)",
    )

    # Configuration
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to configuration file (YAML)",
    )
    config_group.add_argument(
        "--triggers",
        type=str,
        default="./configs/triggers.yaml",
        help="Path to triggers configuration (default: ./configs/triggers.yaml)",
    )
    config_group.add_argument(
        "--prompts",
        type=str,
        default="./configs/prompts",
        help="Path to prompts directory (default: ./configs/prompts)",
    )

    # Audio options
    audio_group = parser.add_argument_group("Audio Options")
    audio_group.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio output (text-only mode)",
    )
    audio_group.add_argument(
        "--audio-device",
        type=int,
        help="Audio input device index",
    )
    audio_group.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )

    # Voice options
    voice_group = parser.add_argument_group("Voice Options")
    voice_group.add_argument(
        "--voice",
        type=str,
        default="alloy",
        choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
        help="TTS voice to use (default: alloy)",
    )
    voice_group.add_argument(
        "--no-realtime",
        action="store_true",
        help="Disable Realtime API (use TTS fallback)",
    )

    # Misc options
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without actual audio/video processing",
    )
    misc_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    return parser


def setup_logging(settings: Settings, verbose: int, debug: bool) -> None:
    """Configure logging based on settings."""
    if debug or verbose >= 3:
        level = logging.DEBUG
    elif verbose >= 2:
        level = logging.INFO
    elif verbose >= 1:
        level = logging.WARNING
    else:
        level = getattr(logging, settings.app.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.debug(f"Logging configured at {logging.getLevelName(level)} level")


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments.

    Args:
        args: Parsed arguments

    Raises:
        UserInputError: If arguments are invalid
    """
    # Validate video input
    if args.input == "video" and not args.video:
        raise UserInputError(
            field="video",
            value=None,
            reason="--video is required when --input=video",
            suggestion="Specify a video file path with --video <path>",
        )

    # Validate region if provided
    if args.region:
        if len(args.region) != 4:
            raise UserInputError(
                field="region",
                value=args.region,
                reason="Region must have exactly 4 values (x y width height)",
            )
        if any(v < 0 for v in args.region):
            raise UserInputError(
                field="region",
                value=args.region,
                reason="Region values must be non-negative",
            )

    logger.debug(f"Arguments validated: input={args.input}")


def run_coach(args: argparse.Namespace, settings: Settings) -> int:
    """
    Main entry point for the coach.

    Args:
        args: Parsed command line arguments
        settings: Application settings

    Returns:
        Exit code
    """
    logger.info("Starting AI Game Coach...")

    # Check for list devices
    if args.list_devices:
        try:
            from infrastructure.audio.capture.device import list_audio_devices

            devices = list_audio_devices()
            print("\nAvailable audio input devices:")
            for device in devices:
                default_marker = " (default)" if device.is_default else ""
                print(f"  [{device.index}] {device.name}{default_marker}")
                print(f"      Channels: {device.channels}, Sample Rate: {device.sample_rate}")
            return 0
        except InfrastructureError as e:
            logger.error(f"Failed to list audio devices: {e}")
            print(f"Error: {e.message}", file=sys.stderr)
            return 1

    # Validate arguments
    try:
        validate_arguments(args)
    except UserInputError as e:
        logger.error(f"Invalid arguments: {e}")
        print(f"Error: {e.user_message}", file=sys.stderr)
        return 2

    # Import and run main
    try:
        from main import main as run_main

        return run_main(
            input_type=args.input,
            video_path=args.video,
            screen_index=args.screen,
            region=tuple(args.region) if args.region else None,
            loop=args.loop,
            output_dir=args.out,
            config_path=args.config,
            triggers_path=args.triggers,
            prompts_path=args.prompts,
            enable_audio=not args.no_audio,
            audio_device=args.audio_device,
            voice=args.voice,
            use_realtime=not args.no_realtime,
            dry_run=args.dry_run,
            debug=args.debug,
        )
    except ImportError:
        # If main module has different structure, use fallback
        logger.warning("Could not import main module, using simplified runner")
        return run_simplified(args, settings)


def run_simplified(args: argparse.Namespace, settings: Settings) -> int:
    """Simplified runner for testing."""
    logger.info("Running in simplified mode")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.out}")
    logger.info("Simplified mode - no actual processing")
    return 0


def handle_exception(exc: Exception, debug: bool = False) -> int:
    """Handle exceptions and return appropriate exit code.

    Args:
        exc: Exception that occurred
        debug: Whether debug mode is enabled

    Returns:
        Exit code
    """
    if isinstance(exc, CLIError):
        logger.error(f"CLI error: {exc}")
        print(f"Error: {exc.user_message}", file=sys.stderr)
        return exc.exit_code
    elif isinstance(exc, UserInputError):
        logger.error(f"User input error: {exc}")
        print(f"Error: {exc.user_message}", file=sys.stderr)
        return 2
    elif isinstance(exc, InfrastructureError):
        logger.error(f"Infrastructure error: {exc}")
        print(f"Error: {exc.message}", file=sys.stderr)
        return 1
    elif isinstance(exc, KeyboardInterrupt):
        logger.info("Interrupted by user")
        print("\nInterrupted by user")
        return 0
    else:
        logger.error(f"Unexpected error: {exc}", exc_info=debug)
        print(f"Error: {exc}", file=sys.stderr)
        if debug:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Load settings
    try:
        settings = load_settings(args.config)
    except Exception as e:
        logger.error(f"Failed to load settings: {e}")
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Setup logging
    setup_logging(settings, args.verbose, args.debug)

    # Run coach
    try:
        return run_coach(args, settings)
    except Exception as e:
        return handle_exception(e, debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())
