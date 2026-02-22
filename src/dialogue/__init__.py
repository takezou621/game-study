"""Dialogue modules for AI coach responses."""

from .openai_client import OpenAIClient
from .realtime_client import RealtimeVoiceClient, VoiceResponse, create_voice_client
from .templates import DialogueTemplateManager

__all__ = [
    'DialogueTemplateManager',
    'OpenAIClient',
    'RealtimeVoiceClient',
    'VoiceResponse',
    'create_voice_client'
]
