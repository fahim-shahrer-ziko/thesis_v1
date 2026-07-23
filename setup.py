"""
setup.py
========
Minimal setup so the project's src/ package can be installed in editable
mode inside the virtual environment:

  pip install -e .

This makes `import config` and `import utils` work from any directory
within the project without needing PYTHONPATH tricks.
"""

from setuptools import find_packages, setup

setup(
    name="apollo-mode-choice",
    version="1.0.0",
    description="Transport mode choice prediction with LLaMA personas and LoRA fine-tuning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "openai>=1.30.0",
        "python-dotenv>=1.0.0",
    ],
)
