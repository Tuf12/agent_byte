"""
Setup script for Agent Byte v3.0
"""

from setuptools import setup, find_packages
import os

# Read README if it exists
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Agent Byte v3.0 - Neural-Symbolic RL Agent"

# Read requirements if it exists
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        # Basic requirements
        return [
            "numpy>=1.20.0",
            "torch>=1.9.0",
            "scikit-learn>=1.0.0",
            "tqdm>=4.60.0",
            "pathlib",
        ]

setup(
    name="agent_byte",
    version="3.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural-Symbolic RL Agent with Transfer Learning",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/agent_byte",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
        "vector_db": [
            "faiss-cpu>=1.7.0",
            "chromadb>=0.3.0",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "plotly>=5.0.0",
        ],
        "gym": [
            "gymnasium>=0.26.0",
            "pygame>=2.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "agent-byte=agent_byte.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agent_byte": [
            "examples/*.py",
            "configs/*.json",
        ],
    },
    zip_safe=False,
)