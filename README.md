# ECoG foundation model

In progress - this repo is under active development in the MedARC discord server.

## Installation

This repository contains both the ECoG foundation model and pretraining code in a monorepo structure.

### Option 1: Foundation Model Only (Recommended for most users)
For just using the foundation model without heavy GPU/pretraining dependencies:

```bash
pip install -e .
```

This installs only the core dependencies needed to use the foundation model.

### Option 2: With Pretraining Capabilities
For running pretraining (requires GPU and CUDA support):

```bash
pip install -e .[pretraining]
```

This includes all the heavy ML dependencies, GPU libraries, and Jupyter environments needed for pretraining.

### Option 3: Everything
To install all dependencies:

```bash
pip install -e .[all]
```

### Requirements Files
- `requirements-core.txt` - Minimal dependencies for the foundation model
- `requirements-pretraining.txt` - Additional dependencies for pretraining  
- `requirements.txt` - All dependencies combined (for reference)

## Run training on local machine

1. Run "sh setup.sh" to create virtual environment

2. Activate the virtual environment with "source ecog/bin/activate"

3. Specify your huggingface [user access token](https://huggingface.co/docs/hub/en/security-tokens) in the makefile to authenticate and run "make download-data" to fetch data from huggingface hub 

4. Specify model training parameters in makefile and run "make model-train"

## Run training on Google Colab

If you don't have access to GPU's try uploading notebooks/ECoG_Model_Training.ipynb to
Colab and use the free tier GPU's. This is a good way to play around with the training
code and try training a model to get onboarded, although training is very slow on the
T4 GPU's.
