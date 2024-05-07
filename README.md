# ECoG foundation model

In progress - this repo is under active development in the MedARC discord server. 

1. Run "sh setup.sh" to create virtual environment

2. Activate the virtual environment with "source ecog/bin/activate"

3. Specify your huggingface [user access token] (https://huggingface.co/docs/hub/en/security-tokens) in the makefile to authenticate and run "make download-data" to fetch data from huggingface hub 

4. Specify model training parameters in makefile and run "make model-train"
