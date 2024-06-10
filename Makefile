# Run all commands in one shell
.ONESHELL:

USR := $(shell whoami | head -c 2)
DT := $(shell date +"%Y%m%d")

ACCESS_TOKEN := "your huggingface access token"

download-data:
	$ python ECoG-MAE/download_data.py \
		--access-token $(ACCESS_TOKEN)

PREFIX = test-long-run
NORM = hour
DATA_SIZE = 1
BATCH_SIZE = 64
NEW_FS = 20
SAMPLE_LENGTH = 2
PATCH_DIMS = 1 1 1
PATCH_SIZE = 1
FRAME_PATCH_SIZE = 4
TUBE_MASK_RATIO = 0.15
DECODER_MASK_RATIO = 0
BANDS = "[[4,8],[8,13],[13,30],[30,55],[70,200]]"
BANDS_STR = all
NUM_EPOCHS = 25
LOSS = patch
LEARNING_RATE = 0
# 0 -> using learning rate scheduler 
DIM = 128
# 0 -> using patch dimensions, no projection to wider embeddings
MLP_DIM = 128
# 0 -> using patch dimensions, no projection to wider embeddings
DATASET_PATH = dataset_full
TRAIN_DATA_PROPORTION = 0.9
JOB_NAME = "$(USR)-$(DT)-$(PREFIX)-ds-$(DATA_SIZE)-bs-$(BATCH_SIZE)-norm-$(NORM)-fs-$(NEW_FS)-sl-$(SAMPLE_LENGTH)-ps-$(PATCH_SIZE)-fps-$(FRAME_PATCH_SIZE)-tmr-$(TUBE_MASK_RATIO)-dmr-$(DECODER_MASK_RATIO)-b-$(BANDS_STR)-ep-$(NUM_EPOCHS)-loss-$(LOSS)-lr-$(LEARNING_RATE)-dim-$(DIM)"

CMD = sbatch --job-name=$(JOB_NAME) submit.sh
# to debug, request interactive gpu node via salloc and select this option:
# CMD = python

# for commands debug, use-contrastive-loss, use-cls-token: add to arguments = True, leave out = False
# --debug -> just enables verbose print out for debugging
# --env -> compute power envelope
# --dataset-path="dataset" -> Sets where training looks for the dataset. Should be a relative path.
# --train-data-proportion=0.8 -> Sets proportion of data assigned to train split. All remaining data is assigned to test.
# --use-cls-token (not implemented yet!)
# --use-contrastive-loss (not implemented yet!)
# --running-cell-masking -> specific type of decoder masking (not properly tested yet!)

model-train:
	mkdir -p logs
	$(CMD) ECoG-MAE/main.py \
		--job-name $(JOB_NAME) \
		--data-size $(DATA_SIZE) \
		--batch-size $(BATCH_SIZE) \
		--env \
		--norm $(NORM) \
		--new-fs $(NEW_FS) \
		--sample-length $(SAMPLE_LENGTH) \
		--patch-dims $(PATCH_DIMS) \
		--patch-size $(PATCH_SIZE) \
		--frame-patch-size $(FRAME_PATCH_SIZE) \
		--tube-mask-ratio $(TUBE_MASK_RATIO) \
		--decoder-mask-ratio $(DECODER_MASK_RATIO) \
		--bands $(BANDS) \
		--num-epochs $(NUM_EPOCHS) \
		--loss $(LOSS) \
		--learning-rate $(LEARNING_RATE) \
		--dim $(DIM) \
		--mlp-dim $(MLP_DIM);
