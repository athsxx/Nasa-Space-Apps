"""Setup script for Air Quality Monitoring Agent."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="aqi-monitoring-agent",
    version="1.0.0",
    author="NASA Air Quality Monitoring Team",
    author_email="aqi-team@nasa.gov",
    description="Air Quality Monitoring Agent with TEMPO satellite and OpenAQ integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/aqi-monitoring-agent",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aqi-agent=cli:run_cli",
            "aqi-server=main:app",
        ],
    },
    include_package_data=True,
    package_data={
        "aqi_agent": ["*.py"],
        "config": ["*.py"],
    },
)