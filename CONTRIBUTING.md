# Contributing to Game-Study

Thank you for your interest in contributing to Game-Study! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Messages](#commit-messages)
- [Reporting Issues](#reporting-issues)
- [Getting Help](#getting-help)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip or uv for package management
- Git

### Setup Steps

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/game-study.git
   cd game-study
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Set Up Pre-commit Hooks** (Optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Code Style

We use the following tools to maintain code quality:

### Formatting

- **Black**: Code formatter with 100 character line length
- **isort**: Import sorter (black-compatible profile)

```bash
# Format code
black src tests
isort src tests
```

### Linting

- **Ruff**: Fast Python linter

```bash
# Run linting
ruff check src tests
```

### Type Checking

- **mypy**: Static type checker with strict mode enabled

```bash
# Run type checking
mypy src
```

### Code Style Guidelines

1. **Type Annotations**: All public functions must have type hints
2. **Docstrings**: Use Google-style docstrings for all public modules, classes, and functions
3. **Line Length**: Maximum 100 characters
4. **Imports**: Group imports (stdlib, third-party, local) and sort alphabetically

Example docstring:

```python
def process_frame(frame: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Process a video frame and extract game state.

    Args:
        frame: Input video frame as numpy array (BGR format).
        threshold: Detection confidence threshold. Defaults to 0.5.

    Returns:
        Dictionary containing extracted game state information.

    Raises:
        ValueError: If frame is empty or has invalid dimensions.
    """
```

### Configuration Reference

All tools are configured in `pyproject.toml`:

| Tool | Configuration Key | Settings |
|------|------------------|----------|
| Black | `[tool.black]` | line_length=100 |
| isort | `[tool.isort]` | profile="black", line_length=100 |
| Ruff | `[tool.ruff]` | line_length=100, select=E,W,F,I,B,C4,UP,ARG,SIM |
| mypy | `[tool.mypy]` | strict mode enabled |

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/unit/test_capture.py -v

# Run specific test
pytest tests/unit/test_capture.py::TestScreenCapture::test_init -v

# Run by marker
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m "not slow"    # Exclude slow tests
```

### Test Structure

Tests are organized by module:

```
tests/
├── test_capture/
├── test_vision/
├── test_trigger/
├── test_dialogue/
├── test_audio/
├── test_diagnostics/
└── test_review/
```

### Writing Tests

1. Place unit tests in `tests/unit/` or the appropriate module directory
2. Name test files `test_<module_name>.py`
3. Name test classes `Test<Feature>`
4. Name test methods `test_<scenario>`

```python
class TestTriggerEngine:
    """Tests for TriggerEngine class."""

    def test_evaluate_with_matching_condition(self):
        """Test that matching conditions trigger correctly."""
        # Arrange
        engine = TriggerEngine("configs/triggers.yaml")
        state = {"hp": 50}

        # Act
        result = engine.evaluate(state)

        # Assert
        assert len(result) > 0
```

### Coverage Requirements

- Minimum coverage: 80%
- All new code should include tests
- Critical paths (main.py, realtime_client.py) should have 80%+ coverage

### Test Markers

We use pytest markers for categorizing tests:

- `unit`: Unit tests (fast, isolated)
- `integration`: Integration tests (slower, may use external resources)
- `slow`: Slow-running tests

## Pull Request Process

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number-description
   ```

2. **Make Changes**
   - Write code following our style guidelines
   - Add/update tests
   - Update documentation if needed

3. **Run Checks**
   ```bash
   # Format code
   black src tests
   isort src tests

   # Lint
   ruff check src tests

   # Type check
   mypy src

   # Test
   pytest --cov=src
   ```

4. **Commit Changes**
   - Use conventional commit messages
   - Reference issues when applicable

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   - Create PR on GitHub
   - Fill out PR template
   - Link related issues

6. **CI Checks**
   - All CI checks must pass
   - Coverage must not decrease
   - No new security vulnerabilities

7. **Code Review**
   - Address review feedback
   - Keep PR focused and reasonably sized

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

### Examples

```
feat(capture): add support for custom screen regions

fix(ocr): handle edge case with empty HP bar

docs(readme): add Docker deployment instructions

test(dialogue): add tests for realtime client
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- **Description**: Clear description of the bug
- **Steps to reproduce**: Minimal steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, relevant dependencies
- **Logs**: Relevant error messages or logs

### Feature Requests

When suggesting features, please include:

- **Use case**: Describe the problem you're trying to solve
- **Proposed solution**: How you think this could be implemented
- **Alternatives**: Any alternative solutions you've considered

## Getting Help

- Open a [Discussion](https://github.com/owner/game-study/discussions) for questions
- Open an [Issue](https://github.com/owner/game-study/issues) for bugs or feature requests
- Check the [Documentation](README.md) for usage information

Thank you for contributing!
