# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security vulnerability in game-study, please report it responsibly.

### How to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of the following methods:

1. **GitHub Security Advisories** (Preferred)
   - Go to the [Security Advisories](https://github.com/kawai/game-study/security/advisories) page
   - Click "Report a vulnerability"
   - Fill out the form with details about the vulnerability

2. **Email**
   - Send an email to: security@kawai.dev
   - Include "SECURITY" in the subject line
   - Provide a detailed description of the vulnerability

### What to Include

Please include the following information:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any possible mitigations you've identified
- Your contact information for follow-up

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 7 days
- **Fix Development**: Depends on severity and complexity
- **Disclosure**: After fix is released

## Security Best Practices

When using game-study, please follow these security best practices:

### API Keys and Secrets

1. **Never commit secrets to version control**
   - Use `.env` files (already in `.gitignore`)
   - Use environment variables in production

2. **Protect your OpenAI API key**
   ```bash
   # In .env file
   OPENAI_API_KEY=sk-your-key-here
   ```

3. **Use strong secrets for WebRTC**
   ```bash
   # Generate a strong secret
   WEBRTC_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
   ```

### Environment Configuration

1. **Use `.env.example` as a template**
   ```bash
   cp .env.example .env
   # Edit .env with your actual values
   ```

2. **Never share `.env` files**

3. **In production, use secure secret management**
   - AWS Secrets Manager
   - Azure Key Vault
   - HashiCorp Vault
   - Kubernetes Secrets

### Docker Security

1. **The container runs as non-root user** (`appuser`)

2. **Mount only necessary volumes**
   ```yaml
   volumes:
     - ./configs:/app/configs:ro  # Read-only
     - ./output:/app/output
   ```

3. **Don't expose unnecessary ports**

### Logging

- Sensitive data is automatically masked in logs using SensitiveFormatter
- API keys, tokens, and passwords are redacted
- Log level should be `INFO` or higher in production

### Dependencies

We use automated tools to check for vulnerabilities:

- **Dependabot**: Weekly dependency updates
- **Safety**: Checks for known vulnerabilities
- **Bandit**: Static security analysis

Run security checks:
```bash
pip-audit -r requirements.txt
safety check -r requirements.txt
bandit -r src/
```

## Security Features

game-study includes several built-in security features:

### SensitiveFormatter

Logs are automatically sanitized to mask:
- API keys (sk-*, api_key patterns)
- Bearer tokens
- Passwords
- Base64-encoded strings

### Input Validation

Pydantic models validate:
- Trigger conditions
- Game state data
- Configuration values

### Rate Limiting

The rate limiter utility (`src/utils/rate_limiter.py`) helps prevent:
- API abuse
- Resource exhaustion

### API Key Handling

- API keys are never stored in instance variables
- Keys are loaded from environment variables at runtime
- Sensitive logging is masked automatically

## Security Updates

Security updates are released as patch versions and announced via:

1. GitHub Security Advisories
2. Release notes
3. CHANGELOG.md

## Project Security Compliance

This project is designed for educational purposes (English learning coach) and:

- **Does NOT automate game actions** (no aim assist, botting, or cheating)
- **Only analyzes HUD elements** for learning feedback
- **Stores logs locally** by default with user consent required for cloud upload
- **Follows fair use guidelines** for game analytics

## Contact

For general security questions (non-vulnerability):
- Open a GitHub Discussion
- Email: security@kawai.dev

Thank you for helping keep game-study secure!
