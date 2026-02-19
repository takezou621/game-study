"""Dialogue modules for AI coach responses."""

from .templates import DialogueTemplateManager
from .openai_client import OpenAIClient

__all__ = ['DialogueTemplateManager', 'OpenAIClient']
