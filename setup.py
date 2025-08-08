# setup.py
from setuptools import setup, find_packages

# Read core requirements
with open("requirements-core.txt") as f:
    core_requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read pretraining requirements
with open("requirements-pretraining.txt") as f:
    pretraining_requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ecog_foundation_model",
    version="0.1.0",
    packages=find_packages(),
    install_requires=core_requirements,
    extras_require={
        'pretraining': pretraining_requirements,
        'all': core_requirements + pretraining_requirements,
    },
    description="Shared ECoG foundation model for pretraining and finetuning.",
    author="Hasson Lab",
    license="MIT",
)
