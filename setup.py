#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements-docs.txt") as f:
    docs_require = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setuptools.setup(
    name="fma_ions",
    version="0.1.0",
    author="Elias Waagaard",
    author_email="elias.walter.waagaard@cern.ch",
    description="Python package for Frequency Map Analysis of ion beams in particle accelerators",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/ewaagaard/fma_ions",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "cpymad>=1.9.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "NAFFlib>=0.1.0",
        "xtrack>=0.40.0",
        "xfields>=0.16.0",
        "xpart>=0.36.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.900",
            "flake8>=4.0.0",
        ],
        "docs": docs_require,
    },
    include_package_data=True,
    zip_safe=False,
)
