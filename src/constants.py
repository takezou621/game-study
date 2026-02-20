"""Application-wide constants.

This module centralizes magic numbers and configuration values
to improve code maintainability and reduce hard-coded values.
"""

# ============================================================================
# Dialogue/Response Settings
# ============================================================================

# Voice client settings
DEFAULT_COOLDOWN_MS = 3000  # Minimum time between voice responses
DEFAULT_MAX_RESPONSE_LENGTH_MS = 10000  # Maximum voice response duration
MAX_TEXT_LENGTH = 500  # Maximum text length for TTS input

# Response length limits
MAX_RESPONSE_CHARS = 200  # Maximum characters in text response

# ============================================================================
# Logging/Frame Processing
# ============================================================================

LOG_INTERVAL_FRAMES = 100  # Log progress every N frames

# ============================================================================
# Vision/OCR Settings
# ============================================================================

# OCR confidence thresholds
OCR_MIN_CONFIDENCE = 0.5  # Minimum confidence for OCR results

# HP/Shield detection
HP_MAX_VALUE = 100
SHIELD_MAX_VALUE = 100

# ============================================================================
# Trigger System
# ============================================================================

# Priority levels (0 = highest, 3 = lowest)
PRIORITY_SURVIVAL = 0  # P0: Urgent, life-threatening situations
PRIORITY_TACTICAL = 1  # P1: Strategic suggestions
PRIORITY_LEARNING = 2  # P2: Educational content
PRIORITY_CHATTER = 3  # P3: Casual conversation

# Default cooldown for triggers (milliseconds)
DEFAULT_TRIGGER_COOLDOWN_MS = 5000

# ============================================================================
# WebRTC Settings
# ============================================================================

WEBRTC_DEFAULT_PORT = 8080
WEBRTC_TOKEN_EXPIRY_SECONDS = 3600  # 1 hour

# ============================================================================
# Session/State
# ============================================================================

MAX_STATE_BUFFER_SIZE = 100  # Maximum states to buffer for WebRTC

# ============================================================================
# File Paths (defaults)
# ============================================================================

DEFAULT_ROI_CONFIG = "./configs/roi_defaults.yaml"
DEFAULT_TRIGGERS_CONFIG = "./configs/triggers.yaml"
DEFAULT_SYSTEM_PROMPT = "./configs/prompts/system.txt"
