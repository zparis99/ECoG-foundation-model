# setup.py
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    install_requires = f.read().splitlines()

setup(
    name="ecog_foundation_model",
    version="1.0.3",
    packages=find_packages(),
    install_requires=install_requires,
    description="Shared ECoG foundation model for pretraining and finetuning.",
    author="Hasson Lab",
    license="MIT",
)
