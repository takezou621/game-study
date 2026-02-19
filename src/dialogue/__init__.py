"""Dialogue modules for AI coach responses."""

from .templates import DialogueTemplateManager
from .openai_client import OpenAIClient
from .realtime_client import RealtimeVoiceClient, VoiceResponse, create_voice_client

__all__ = [
    'DialogueTemplateManager',
    'OpenAIClient',
    'RealtimeVoiceClient',
    'VoiceResponse',
    'create_voice_client'
]
