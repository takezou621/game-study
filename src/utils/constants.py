"""Constants for the game-study application.

This module contains all magic numbers and configuration constants
used throughout the application.
"""

# Priority Levels
PRIORITY_SURVIVAL = 0
PRIORITY_TACTICAL = 1
PRIORITY_LEARNING = 2
PRIORITY_CHATTER = 3

# Default Priority Values
DEFAULT_CURRENT_PRIORITY = 99
DEFAULT_PRIORITY = 2

# Timeouts (milliseconds)
WEBSOCKET_TIMEOUT_MS = 10000  # 10 seconds
RESPONSE_TIMEOUT_MS = 30000  # 30 seconds
EVENT_LOOP_WAIT_MS = 10  # 10 milliseconds
VOICE_RESPONSE_TIMEOUT_MS = 30000  # 30 seconds

# Cooldowns (milliseconds)
DEFAULT_COOLDOWN_MS = 3000  # 3 seconds
DEFAULT_INACTIVITY_THRESHOLD_MS = 30000  # 30 seconds

# Audio Constants
TTS_MAX_INPUT_LENGTH = 500  # Maximum text length for TTS
AUDIO_DURATION_MS_DIVISOR = 32  # For calculating audio duration from bytes

# Response Limits
MAX_RESPONSE_LENGTH_MS = 10000  # 10 seconds
MAX_RESPONSE_LENGTH_CHARS = 200  # Maximum response length in characters
DEFAULT_MAX_TOKENS = 100  # Default max tokens for OpenAI
DEFAULT_TEMPERATURE = 0.7  # Default temperature for OpenAI
SHORT_TEMPLATE_LENGTH = 50  # Characters to truncate combat templates
MIN_SENTENCES_TEMPLATE = 2  # Minimum sentences for short template

# Realtime API Configuration
DEFAULT_REALTIME_MODEL = "gpt-4o-realtime-preview-2024-12-17"
DEFAULT_VOICE = "alloy"
DEFAULT_TTS_MODEL = "tts-1"

# OpenAI Configuration
DEFAULT_MODEL = "gpt-4"
REALTIME_VAD_THRESHOLD = 0.5
REALTIME_PREFIX_PADDING_MS = 300
REALTIME_SILENCE_DURATION_MS = 500
REALTIME_MAX_RESPONSE_TOKENS = 150

# OCR Configuration
DEFAULT_MIN_CONFIDENCE = 0.7
MIN_DIGIT_CONFIDENCE = 0.3
DEFAULT_DIGIT_SIZE = (20, 30)  # Width, Height for digit template matching

# OCR Digit Recognition
ASPECT_RATIO_MIN = 0.15
ASPECT_RATIO_MAX = 1.0
MIN_DIGIT_HEIGHT = 8
MAX_DIGIT_HEIGHT_RATIO = 0.5

# OCR Value Ranges
HP_MIN = 0
HP_MAX = 100
SHIELD_MIN = 0
SHIELD_MAX = 100
AMMO_MIN = 0
AMMO_MAX = 999
MATERIALS_MIN = 0
MATERIALS_MAX = 999

# Frame Processing
PROGRESS_UPDATE_INTERVAL = 100  # Frames between progress updates

# WebRTC Configuration
DEFAULT_STUN_SERVER = "stun:stun.l.google.com:19302"
DEFAULT_PORT_RANGE = (10000, 20000)
DEFAULT_SIGNALING_HOST = "0.0.0.0"
DEFAULT_SIGNALING_PORT = 8080
MAX_STATE_BUFFER = 100

# OCR Image Processing
DEFAULT_DENOISE_STRENGTH = 10
DEFAULT_DENOISE_TEMPLATE_WINDOW = 7
DEFAULT_DENOISE_SEARCH_WINDOW = 21
DEFAULT_CLAHE_CLIP_LIMIT = 2.0
DEFAULT_CLAHE_TILE_SIZE = (4, 4)
DEFAULT_ADAPTIVE_THRESHOLD_BLOCK_SIZE = 11
DEFAULT_ADAPTIVE_THRESHOLD_C = 2

# Combat Template Priority Levels
COMBAT_TEMPLATES_PRIORITIES = (PRIORITY_SURVIVAL, PRIORITY_TACTICAL)

# Source names for logging
SOURCE_OCR_TEMPLATE = "ocr_template"
SOURCE_OCR_TEMPLATE_NO_DIGITS = "ocr_template_no_digits"
SOURCE_OCR_TEMPLATE_ERROR = "ocr_template_error"
SOURCE_OCR_TESSERACT = "ocr_tesseract"
SOURCE_OCR_TESSERACT_NO_MATCH = "ocr_tesseract_no_match"
SOURCE_OCR_TESSERACT_ERROR = "ocr_tesseract_error"
SOURCE_OCR_ERROR = "ocr_error"
SOURCE_OCR_UNAVAILABLE = "ocr_unavailable"
SOURCE_OCR_EMPTY_FRAME = "ocr_empty_frame"

# Default Tesseract confidence
TESSERACT_DEFAULT_CONFIDENCE = 0.95

# OpenAI availability check
OPENAI_AVAILABLE_CHECK_TIMEOUT = 10.0
