"""
DocVAL: Visual Answer Localization for Document VQA
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="docval",
    version="1.0.0",
    author="Ahmad Shirazi",
    author_email="your.email@example.com",
    description="Visual Answer Localization for Document VQA with Asymmetric Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/docval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pytest-cov>=4.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "docval-phase-a=docval.scripts.run_phase_a:main",
            "docval-train-b1=docval.training.train_phase_b1:main",
            "docval-train-b2=docval.training.train_phase_b2:main",
            "docval-evaluate=docval.inference.evaluate_phase_c:main",
            "docval-monitor=docval.scripts.monitor_phase_a:main",
        ],
    },
    include_package_data=True,
    package_data={
        "docval": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    keywords="document-understanding visual-qa vqa vlm vision-language transformers",
    project_urls={
        "Bug Reports": "https://github.com/your-username/docval/issues",
        "Source": "https://github.com/your-username/docval",
        "Documentation": "https://github.com/your-username/docval/tree/main/docs",
    },
)
