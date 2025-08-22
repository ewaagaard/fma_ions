Contributing
============

We welcome contributions to fma_ions! This guide will help you get started.

Development Setup
-----------------

1. **Fork and clone the repository:**

.. code-block:: bash

   git clone https://github.com/yourusername/fma_ions.git
   cd fma_ions

2. **Create a development environment:**

.. code-block:: bash

   conda create --name fma_dev python=3.11
   conda activate fma_dev
   pip install -e ".[dev]"

3. **Install pre-commit hooks:**

.. code-block:: bash

   pre-commit install

Code Style
----------

We follow Python best practices:

- **PEP 8** compliance for code formatting
- **Type hints** for function signatures
- **Docstrings** following NumPy style
- **Black** for automatic code formatting

**Format your code:**

.. code-block:: bash

   black fma_ions/
   isort fma_ions/

**Check style:**

.. code-block:: bash

   flake8 fma_ions/
   mypy fma_ions/

Testing
-------

Run the test suite:

.. code-block:: bash

   pytest tests/ -v

**Add tests for new features:**

- Place tests in `tests/` directory
- Follow existing test structure
- Include both unit tests and integration tests
- Test edge cases and error handling

Documentation
-------------

**Building documentation locally:**

.. code-block:: bash

   cd docs/
   make html

**Documentation standards:**

- Update docstrings for all public functions
- Add examples to docstrings when helpful
- Update this documentation for significant changes
- Include mathematical formulas using Sphinx math syntax

Pull Request Process
--------------------

1. **Create a feature branch:**

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. **Make your changes:**
   - Write clean, tested code
   - Update documentation
   - Add tests for new functionality

3. **Commit your changes:**

.. code-block:: bash

   git add .
   git commit -m "Add feature: brief description"

4. **Push and create PR:**

.. code-block:: bash

   git push origin feature/your-feature-name

5. **PR requirements:**
   - Clear description of changes
   - Reference any related issues
   - Pass all tests
   - Maintain or improve test coverage

Reporting Issues
----------------

**Before reporting an issue:**

- Check existing issues for duplicates
- Try to reproduce the problem
- Gather relevant information (OS, Python version, error messages)

**Good issue reports include:**

- Clear problem description
- Steps to reproduce
- Expected vs actual behavior
- System information
- Minimal code example (if applicable)

Feature Requests
----------------

We welcome feature requests! Please:

- Check if the feature already exists
- Explain the use case clearly
- Consider implementation complexity
- Be open to alternative solutions

Development Guidelines
----------------------

**Code Organization:**

- Keep modules focused and cohesive
- Use clear, descriptive names
- Follow existing patterns
- Document complex algorithms

**Performance Considerations:**

- Profile performance-critical code
- Consider GPU compatibility
- Use vectorized operations when possible
- Test with realistic problem sizes

**Physics Accuracy:**

- Validate against known results
- Include references for algorithms
- Test with different accelerator configurations
- Document assumptions and limitations

**Accelerator-Specific Code:**

- Support multiple CERN accelerators (SPS, PS, LEIR)
- Keep accelerator parameters configurable
- Validate against measurements when available

Release Process
---------------

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

**Release checklist:**

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Build documentation
5. Create release on GitHub
6. Publish to PyPI (maintainers only)

Community
---------

**Getting Help:**

- Start discussions on GitHub
- Join CERN accelerator physics communities
- Attend relevant workshops and conferences

**Code of Conduct:**

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers learn
- Acknowledge contributions

Recognition
-----------

Contributors are recognized in:

- `CONTRIBUTORS.md` file
- Release notes
- Academic publications (when appropriate)

Thank you for contributing to fma_ions!
