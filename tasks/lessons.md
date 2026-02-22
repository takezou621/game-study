# Lessons Learned

This file captures patterns and lessons learned during development to prevent repeated mistakes.

---

## 2026-02-22: PR #20 Codex CLI Reviews

### Lesson 1: Config Key Consistency
**Issue**: Code expected `templates` key but config used `template`
**Fix**: Support both keys with fallback
```python
templates=trigger.get('templates') or trigger.get('template', {})
```
**Rule**: Always check YAML config keys match code expectations

### Lesson 2: API Key Handling
**Issue**: `_get_api_key()` only checked env var, ignored constructor arg
**Fix**: Store resolved key and check it first
```python
self._resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
def _get_api_key(self):
    key = getattr(self, '_resolved_api_key', None)
    if key:
        return key
    return os.getenv("OPENAI_API_KEY")
```
**Rule**: Constructor args should override env vars

### Lesson 3: Docker Compose Default Command
**Issue**: Service inherited `CMD ["--help"]` from Dockerfile, exited immediately
**Fix**: Add explicit command to compose
```yaml
command: ["tail", "-f", "/dev/null"]
```
**Rule**: Always specify command in docker-compose for long-running services

### Lesson 4: Health Check Optional Dependencies
**Issue**: Health check required API key, but app works without it
**Fix**: Exclude optional checks from overall health
```python
results["healthy"] = all([
    results["config"],
    results["directories"],  # API key is optional
])
```
**Rule**: Health checks should only require essential dependencies

### Lesson 5: Build System Configuration
**Issue**: `python -m build` failed without `[build-system]` in pyproject.toml
**Fix**: Add setuptools configuration
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```
**Rule**: Always include build-system config for packaging

### Lesson 6: Log Sanitization
**Issue**: `log_error()` wrote raw messages, bypassing SensitiveFormatter
**Fix**: Sanitize before writing
```python
formatter = SensitiveFormatter()
sanitized_message = formatter._mask_sensitive(message)
```
**Rule**: All log paths must go through sanitization

---

## 2026-02-22: Type Annotation Fixes

### Lesson 7: Python 3.10 Compatibility
**Issue**: `typing.Self` only available in Python 3.11+
**Fix**: Use TYPE_CHECKING conditional import
```python
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Self
```
**Rule**: Use `from __future__ import annotations` for 3.10 compatibility

### Lesson 8: Pydantic Validator Type Hints
**Issue**: Validators missing type annotations for strict mypy
**Fix**: Add complete signatures
```python
@field_validator('field')
@classmethod
def validate_field(cls, v: str) -> str:
    ...
```
**Rule**: All Pydantic validators need `@classmethod` and type hints

---

## Testing Patterns

### Lesson 9: Config Factory Fixture
**Pattern**: Use factory fixture for complex test configs
```python
@pytest.fixture
def config_factory():
    def _create_config(tmpdir, triggers_override=None):
        # Create config with overrides
        ...
    return _create_config
```
**Rule**: Prefer factory fixtures over fixed fixtures for flexibility

### Lesson 10: State Builder Updates
**Issue**: Test expected no trigger, but inactivity triggered P3
**Fix**: Create isolated config with conditions that can't be met
**Rule**: For testing "no trigger", use impossible conditions

---

## 2026-02-22: Test Assertion Fix

### Lesson 11: Mock Thread Assertion
**Issue**: `mock_thread.assert_called_once()` failed because mock instance was not callable
**Fix**: Use `mock_thread.start.assert_called_once()` to verify method called on instance
```python
# Wrong
mock_thread.assert_called_once()  # mock_thread is instance, not class mock

# Correct
mock_thread.start.assert_called_once()  # Check start() was called
```
**Rule**: When mocking Thread with `return_value=mock_instance`, assert on instance methods not the instance itself

---

## Summary of Rules

1. ✅ Check config keys match code expectations
2. ✅ Constructor args override env vars
3. ✅ Specify command in docker-compose for services
4. ✅ Health checks only require essential dependencies
5. ✅ Include build-system config in pyproject.toml
6. ✅ All log paths must sanitize sensitive data
7. ✅ Use `from __future__ import annotations` for 3.10
8. ✅ Pydantic validators need complete type hints
9. ✅ Use factory fixtures for complex configs
10. ✅ Use impossible conditions to test "no trigger"
11. ✅ Mock instance methods, not instance itself
