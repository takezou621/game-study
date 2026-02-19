"""Dialogue modules for AI coach responses."""

from .templates import DialogueTemplateManager
from .openai_client import OpenAIClient

try:
    from .realtime_client import RealtimeVoiceClient
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False

__all__ = [
    'DialogueTemplateManager',
    'OpenAIClient',
    'RealtimeVoiceClient',
]

if REALTIME_AVAILABLE:
    __all__.append('RealtimeVoiceClient')
