# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Type checking improvements:
  - Enabled strict mypy mode (disallow_untyped_defs, disallow_incomplete_defs, disallow_untyped_decorators)
  - Added type annotations to all public functions in main.py
  - Added type annotations to all methods in capture/base.py
  - Added type annotations to all methods in trigger/engine.py
  - Added comprehensive Google-style docstrings across modules

- Documentation:
  - Added badges to README.md (CI Status, Coverage, PyPI Version, License, Python versions)
  - Added Docker usage instructions to README.md
  - Added Configuration reference section to README.md
  - Created CONTRIBUTING.md with development setup, code style, and PR process
  - Created SECURITY.md with vulnerability reporting and security best practices
  - Updated CHANGELOG.md in Keep a Changelog format

### Changed

- Updated mypy configuration to strict mode for improved type safety
- Standardized docstring format to Google-style across codebase
- Improved type hint consistency using modern Python syntax (e.g., `dict[str, Any]` vs `Dict[str, Any]`)

### Security

- Added SECURITY.md with:
  - Supported versions policy
  - Vulnerability reporting process
  - Security best practices for contributors

## [0.1.0] - 2026-02-21

### Added

- Phase 1 MVP: Video input + ROI state extraction + YAML triggers
  - Video file capture module (`src/capture/video_file.py`)
  - Screen capture module using MSS (`src/capture/screen_capture.py`)
  - ROI extraction for Fortnite HUD elements (`src/vision/roi.py`)
  - YOLO-based object detection (`src/vision/yolo_detector.py`)
  - OCR for HP/Shield values (`src/vision/ocr.py`)
  - Game state builder (`src/vision/state_builder.py`)
  - Trigger engine with YAML-based rules (`src/trigger/engine.py`)
  - Dialogue template manager (`src/dialogue/templates.py`)
  - OpenAI GPT client (`src/dialogue/openai_client.py`)

- Phase 2: Low-latency features (2026-02-19)
  - Real-time screen capture
  - WebRTC streaming support (`src/utils/webrtc.py`)
  - Realtime voice client for OpenAI Realtime API (`src/dialogue/realtime_client.py`)
  - Signaling server for WebRTC

- Phase 3: Console enhancements (2026-02-21)
  - Microphone capture with noise gate (`src/audio/capture.py`)
  - Voice Activity Detection - WebRTC/Silero/Energy (`src/audio/vad.py`)
  - Speech-to-Text via Whisper API (`src/audio/stt_client.py`)
  - Audio diagnostics - echo/crosstalk detection (`src/diagnostics/audio_check.py`)
  - System diagnostics - device/network checks (`src/diagnostics/system_check.py`)
  - Session review with statistics and scoring (`src/review/stats.py`, `src/review/scorer.py`)
  - Weakness analysis (`src/review/analyzer.py`)

- Utility modules:
  - Logger with session tracking (`src/utils/logger.py`)
  - Time utilities (`src/utils/time.py`)
  - Rate limiting (`src/utils/rate_limiter.py`)
  - Retry logic (`src/utils/retry.py`)
  - Exception hierarchy (`src/utils/exceptions.py`)

- Configuration:
  - ROI defaults configuration (`configs/roi_defaults.yaml`)
  - Triggers configuration (`configs/triggers.yaml`)
  - System prompt template (`configs/prompts/system.txt`)
  - Environment variables example (`.env.example`)

- Testing:
  - Comprehensive test suite with 51 passing tests
  - Unit and integration test structure
  - Coverage reporting with pytest-cov
  - Test fixtures for common test data

### Project Structure

```
src/
├── main.py              # CLI entry point
├── constants.py         # Application constants
├── capture/            # Video/screen capture
│   ├── base.py         # Abstract base class
│   ├── video_file.py   # Video file capture
│   └── screen_capture.py # Screen capture
├── vision/             # Computer vision modules
│   ├── roi.py          # ROI extraction
│   ├── anchors.py      # Anchor detection
│   ├── yolo_detector.py # YOLO object detection
│   ├── ocr.py          # OCR for text extraction
│   └── state_builder.py # Game state construction
├── trigger/            # Trigger engine and rules
│   ├── engine.py       # Trigger evaluation engine
│   └── rules.py        # Trigger rule definitions
├── dialogue/           # AI dialogue clients
│   ├── templates.py    # Response templates
│   ├── openai_client.py # OpenAI GPT client
│   └── realtime_client.py # Realtime API voice client
├── audio/              # Audio capture and processing
│   ├── capture.py      # Microphone capture
│   ├── vad.py          # Voice Activity Detection
│   └── stt_client.py   # Speech-to-Text
├── diagnostics/        # System diagnostics
│   ├── audio_check.py  # Audio diagnostics
│   ├── system_check.py # System diagnostics
│   └── report.py       # Diagnostic reports
├── review/             # Session review
│   ├── stats.py        # Statistics collection
│   ├── scorer.py       # Score calculation
│   ├── analyzer.py     # Weakness analysis
│   └── report.py       # Review reports
└── utils/              # Utility modules
    ├── logger.py       # Logging utilities
    ├── time.py         # Time utilities
    ├── webrtc.py       # WebRTC utilities
    ├── constants.py    # Constants
    ├── exceptions.py   # Exceptions
    ├── rate_limiter.py # Rate limiting
    └── retry.py        # Retry logic
```

[Unreleased]: https://github.com/owner/game-study/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/owner/game-study/releases/tag/v0.1.0
