# Contributing to Music Dataset Tool

Thank you for your interest in contributing to Music Dataset Tool! This document provides guidelines and instructions for contributing.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/MusicDataSetTool.git
cd MusicDataSetTool
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

## Code Style

We follow PEP 8 style guidelines with some modifications:

- Maximum line length: 100 characters
- Use meaningful variable names
- Add docstrings to all public functions and classes
- Type hints are encouraged but not mandatory

### Formatting

We use `black` for code formatting:

```bash
black src/
```

### Linting

We use `flake8` for linting:

```bash
flake8 src/
```

## Testing

### Running Tests

```bash
pytest tests/ -v
```

### Test Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Name test functions as `test_*`
- Use fixtures from `conftest.py`
- Aim for high test coverage (>80%)

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write clean, documented code
- Follow the code style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
pytest tests/ -v
black src/
flake8 src/
```

### 4. Commit Changes

Use clear, descriptive commit messages:

```bash
git commit -m "Add feature: description of feature"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Commit Message Guidelines

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
- `feat: Add support for FLAC audio format`
- `fix: Resolve memory leak in feature extraction`
- `docs: Update API documentation`
- `refactor: Improve error handling in analyzer`

## Code Review Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] No breaking changes (or documented if necessary)
- [ ] Commit messages are clear
- [ ] Branch is up to date with main

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Detailed steps to reproduce the problem
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: OS, Python version, package versions
6. **Logs**: Relevant error messages or logs

## Feature Requests

Feature requests are welcome! Please include:

1. **Use Case**: Why this feature would be useful
2. **Description**: Detailed description of the feature
3. **Examples**: Examples of how it would work
4. **Alternatives**: Alternative solutions you've considered

## Documentation

- Update README.md for user-facing changes
- Update docstrings for code changes
- Add examples for new features
- Keep documentation clear and concise

## Questions?

- Open an issue with the `question` label
- Check existing issues and documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!
