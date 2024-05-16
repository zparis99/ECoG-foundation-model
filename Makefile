# Run all commands in one shell
.ONESHELL:

USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d")

ACCESS_TOKEN := "your huggingface access token"

download-data:
	$ python ECoG-MAE/download_data.py \
		--access-token $(ACCESS_TOKEN)

PREFIX =
NORM = batch
DATA_SIZE = 0.25
BATCH_SIZE = 256
NEW_FS = 20
SAMPLE_LENGTH = 2
PATCH_SIZE =  1 2 2
PATCH_SIZE_STR = 2
FRAME_PATCH_SIZE = 4
TUBE_MASK_RATIO = 0.75
DECODER_MASK_RATIO = 0
BANDS = "[[4,8],[8,13],[13,30],[30,55],[70,200]]"
BANDS_STR = all
NUM_EPOCHS = 10
LEARNING_RATE = 0
# 0 -> using learning rate scheduler 
DIM = 0
# 0 -> using patch dimensions, no projection to wider embeddings
MLP_DIM = 0
# 0 -> using patch dimensions, no projection to wider embeddings
JOB_NAME = "$(USR)-$(DT)-$(PREFIX)-ds-$(DATA_SIZE)-bs-$(BATCH_SIZE)-norm-$(NORM)-fs-$(NEW_FS)-sl-$(SAMPLE_LENGTH)-ps-$(PATCH_SIZE_STR)-fps-$(FRAME_PATCH_SIZE)-dmr-$(DECODER_MASK_RATIO)-b-$(BANDS_STR)-ep-$(NUM_EPOCHS)-lr-$(LEARNING_RATE)"
CMD = sbatch --job-name=$(JOB_NAME) submit.sh

# to debug, request interactive gpu node via salloc and select this option:
# CMD = python

# for commands debug, use-contrastive-loss, use-cls-token: add to arguments = True, leave out = False
# --debug -> just enables verbose print out for debugging
# --env -> compute power envelope
# --sandbox -> use sandbox data
# --use-cls-token (not implemented yet!)
# --use-contrastive-loss (not implemented yet!)
# --running-cell-masking -> specific type of decoder masking (not properly tested yet!)

model-train:
	mkdir -p logs
	$(CMD) ECoG-MAE/main.py \
		--job-name $(JOB_NAME) \
		--data-size $(DATA_SIZE) \
		--sandbox \
		--batch-size $(BATCH_SIZE) \
		--env \
		--norm $(NORM) \
		--new-fs $(NEW_FS) \
		--sample-length $(SAMPLE_LENGTH) \
		--patch-size $(PATCH_SIZE) \
		--frame-patch-size $(FRAME_PATCH_SIZE) \
		--tube-mask-ratio $(TUBE_MASK_RATIO) \
		--decoder-mask-ratio $(DECODER_MASK_RATIO) \
		--bands $(BANDS) \
		--num-epochs $(NUM_EPOCHS) \
		--learning-rate $(LEARNING_RATE) \
		--dim $(DIM) \
		--mlp-dim $(MLP_DIM);
