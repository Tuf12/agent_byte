#!/usr/bin/env python3
"""
Setup script to create the Agent Byte v3.0 project structure.

Run this script to create all necessary directories and __init__.py files.
"""

import os
from pathlib import Path


def create_project_structure():
    """Create the complete project directory structure."""

    # Define the structure
    structure = {
        'agent_byte': {
            '__init__.py': '"""Agent Byte v3.0 - Modular Transfer Learning AI Agent"""\n\n__version__ = "3.0.0"\n',
            'core': {
                '__init__.py': '"""Core agent functionality."""\n',
            },
            'storage': {
                '__init__.py': '"""Storage backend implementations."""\n\nfrom .base import StorageBase\nfrom .json_numpy_storage import JsonNumpyStorage\n\n__all__ = ["StorageBase", "JsonNumpyStorage"]\n',
            },
            'knowledge': {
                '__init__.py': '"""Knowledge management and transfer learning."""\n',
            },
            'analysis': {
                '__init__.py': '"""Environment analysis and state interpretation."""\n',
            },
            'utils': {
                '__init__.py': '"""Utility functions and helpers."""\n',
            },
        },
        'examples': {
            '__init__.py': '',
        },
        'tests': {
            '__init__.py': '',
        },
    }

    # Additional root-level files
    root_files = {
        'README.md': """# Agent Byte v3.0

A modular, transferable AI agent with neural-symbolic dual brain architecture.

## Features

- **Environment Agnostic**: Works with any environment implementing the standard interface
- **Transfer Learning**: Transfers knowledge between different environments
- **Dual Brain Architecture**: Combines neural learning with symbolic reasoning
- **Flexible Storage**: Supports JSON, databases, and vector stores
- **No Hard-coding**: Completely modular and configurable

## Installation

```bash
pip install agent-byte
```

## Quick Start

```python
from agent_byte import AgentByte, JsonNumpyStorage
from agent_byte.adapters import GymnasiumAdapter
import gymnasium as gym

# Create agent
agent = AgentByte(
    agent_id="my_agent",
    storage=JsonNumpyStorage("./agent_data")
)

# Train on any Gymnasium environment
env = gym.make("CartPole-v1")
adapted_env = GymnasiumAdapter(env)
agent.train(adapted_env, episodes=1000)

# Transfer to new environment
env2 = gym.make("MountainCar-v0")
adapted_env2 = GymnasiumAdapter(env2)
agent.transfer_to(adapted_env2)
```

## License

MIT License
""",
        '.gitignore': """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Agent Data
agent_data/
*.npz
*.npy

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/
.mypy_cache/
""",
        'pyproject.toml': """[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-byte"
version = "3.0.0"
description = "A modular, transferable AI agent with neural-symbolic dual brain architecture"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[tool.black]
line-length = 100
target-version = ['py38', 'py39', 'py310', 'py311']

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
""",
    }

    def create_structure(base_path, structure):
        """Recursively create directory structure."""
        for name, content in structure.items():
            path = base_path / name

            if isinstance(content, dict):
                # It's a directory
                path.mkdir(exist_ok=True)
                print(f"Created directory: {path}")
                create_structure(path, content)
            else:
                # It's a file
                with open(path, 'w') as f:
                    f.write(content)
                print(f"Created file: {path}")

    # Create the structure
    base = Path('.')
    create_structure(base, structure)

    # Create root files
    for filename, content in root_files.items():
        with open(base / filename, 'w') as f:
            f.write(content)
        print(f"Created file: {filename}")

    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Create a virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate (or venv\\Scripts\\activate on Windows)")
    print("3. Install development dependencies: pip install -e .[dev]")
    print("4. Start implementing the core agent functionality!")


if __name__ == "__main__":
    create_project_structure()