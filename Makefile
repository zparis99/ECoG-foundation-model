# Run all commands in one shell
.ONESHELL:

USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d-%H%M")
DT := ${USR}

DEBUG = False
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
USE_CONTRASTIVE_LOSS = False
USE_CLS_TOKEN = True
JOB_NAME = "ds-$(DATA_SIZE)-bs-$(BATCH_SIZE)-fs-$(NEW_FS)-sl-$(SAMPLE_LENGTH)-ps-$(PATCH_SIZE_STR)-fps-$(FRAME_PATCH_SIZE)-dmr-$(DECODER_MASK_RATIO)-b-$(BANDS_STR)-ne-$(NUM_EPOCHS)"
CMD = sbatch --job-name=$(JOB_NAME) submit.sh

model-train:
	mkdir -p logs
	$(CMD) ECoG-MAE/main.py \
		--job-name $(JOB_NAME) \
		--debug $(DEBUG) \
		--data-size $(DATA_SIZE) \
		--batch-size $(BATCH_SIZE) \
		--new-fs $(NEW_FS) \
		--sample-length $(SAMPLE_LENGTH) \
		--patch-size $(PATCH_SIZE) \
		--frame-patch-size $(FRAME_PATCH_SIZE) \
		--tube-mask-ratio $(TUBE_MASK_RATIO) \
		--decoder-mask-ratio $(DECODER_MASK_RATIO) \
		--bands $(BANDS) \
		--num-epochs $(NUM_EPOCHS) \
		--use-contrastive-loss $(USE_CONTRASTIVE_LOSS) \
		--use-cls-token $(USE_CLS_TOKEN);
