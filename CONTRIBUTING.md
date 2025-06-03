# Contributing to fma_ions

Thank you for your interest in contributing to fma_ions! We welcome contributions from the community to help improve this project.

## How to Contribute

### Reporting Issues
If you find a bug or have a feature request, please open an issue on our [GitHub Issues](https://github.com/ewaagaard/fma_ions/issues) page. When reporting a bug, please include:

- A clear description of the issue
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your operating system and Python version
- Any error messages or stack traces

### Setting Up the Development Environment

1. Fork the repository and clone it to your local machine:
   ```bash
   git clone https://github.com/your-username/fma_ions.git
   cd fma_ions
   ```

2. Create a new virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for static type checking

Before submitting a pull request, please run:

```bash
black .
isort .
flake8
mypy fma_ions/
```

### Testing

We use `pytest` for testing. To run the tests:

```bash
pytest tests/ -v
```

To run tests with coverage:

```bash
pytest tests/ -v --cov=fma_ions --cov-report=term-missing
```

### Documentation

Documentation is built using Sphinx. To build the documentation locally:

```bash
cd docs
make html
```

The built documentation will be available in `docs/_build/html/index.html`.

### Submitting a Pull Request

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them with a descriptive commit message:
   ```bash
   git commit -m "Add feature: your feature description"
   ```

3. Push your changes to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Open a pull request against the `main` branch of the main repository.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing to fma_ions, you agree that your contributions will be licensed under the [MIT License](LICENSE).
