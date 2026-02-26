"""Presentation layer for CLI and formatters."""

from presentation.cli.main import create_parser, main
from presentation.exceptions import (
    CLIError,
    DisplayError,
    FormatterError,
    OutputFormatError,
    PresentationError,
    UserInputError,
)
from presentation.formatters.json_formatter import JsonFormatter
from presentation.formatters.text_formatter import TextFormatter

__all__ = [
    # CLI
    "main",
    "create_parser",
    # Formatters
    "JsonFormatter",
    "TextFormatter",
    # Exceptions
    "PresentationError",
    "CLIError",
    "OutputFormatError",
    "UserInputError",
    "DisplayError",
    "FormatterError",
]
