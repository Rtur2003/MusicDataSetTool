# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

1. **Do NOT** open a public issue
2. Email the maintainer directly (see profile)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Status Updates**: Every week
- **Resolution**: Depends on severity

### Security Best Practices for Users

#### API Credentials

1. **Never commit credentials** to version control
2. Use `.env` file for credentials (included in `.gitignore`)
3. Rotate credentials regularly
4. Use environment-specific credentials

#### Audio Files

1. Validate audio file sources
2. Scan files for malware before processing
3. Use sandboxed environments for untrusted files

#### Dependencies

1. Keep dependencies up to date
2. Review dependency security advisories
3. Use `pip-audit` to check for vulnerabilities:
   ```bash
   pip install pip-audit
   pip-audit
   ```

## Known Security Considerations

### API Rate Limiting

- Implement rate limiting for API calls
- Use retry logic with exponential backoff
- Cache API responses when appropriate

### Input Validation

- All file paths are validated
- Audio formats are checked
- API responses are validated

### Data Privacy

- No user data is collected by default
- API credentials are not logged
- Audio files are processed locally

## Security Updates

Security updates will be released as patch versions (e.g., 1.0.1) and announced in:
- GitHub Releases
- README.md changelog
- Security advisories (for critical issues)

## Acknowledgments

We appreciate security researchers who responsibly disclose vulnerabilities. Contributors will be acknowledged in release notes (unless they prefer to remain anonymous).
