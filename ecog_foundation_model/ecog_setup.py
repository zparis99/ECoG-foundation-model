from dataclasses import asdict
import torch
import subprocess
from importlib.metadata import version, PackageNotFoundError

from ecog_foundation_model.config import VideoMAEExperimentConfig
import ecog_foundation_model.constants
from ecog_foundation_model.mae_st_util.models_mae import MaskedAutoencoderViT
import ecog_foundation_model.mae_st_util.logging as logging

logger = logging.get_logger(__name__)


def create_model(config: VideoMAEExperimentConfig):
    model_config = config.video_mae_task_config.vit_config
    num_frames = int(
        config.ecog_data_config.sample_length * config.ecog_data_config.new_fs
    )
    model = MaskedAutoencoderViT(
        img_size=ecog_foundation_model.constants.GRID_SIZE,
        patch_size=model_config.patch_size,
        in_chans=len(config.ecog_data_config.bands),
        embed_dim=model_config.dim,
        depth=model_config.depth,
        num_heads=model_config.num_heads,
        decoder_embed_dim=model_config.decoder_embed_dim,
        decoder_depth=model_config.decoder_depth,
        decoder_num_heads=model_config.decoder_num_heads,
        mlp_ratio=model_config.mlp_ratio,
        num_frames=num_frames,
        t_patch_size=model_config.frame_patch_size,
        no_qkv_bias=model_config.no_qkv_bias,
        sep_pos_embed=model_config.sep_pos_embed,
        trunc_init=model_config.trunc_init,
        cls_embed=model_config.use_cls_token,
        # TODO: Make this configurable.
        pred_t_dim=num_frames,
        img_mask=None,
        pct_masks_to_decode=config.video_mae_task_config.pct_masks_to_decode,
        proj_drop=model_config.proj_drop,
        drop_path=model_config.drop_path,
    )
    return model


def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        branch = (
            subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
            .decode()
            .strip()
        )
        try:
            tag = (
                subprocess.check_output(["git", "describe", "--exact-match", "--tags"])
                .decode()
                .strip()
            )
        except subprocess.CalledProcessError:
            tag = None
        return {"commit": commit, "branch": branch, "tag": tag}
    except Exception:
        return {"commit": "unknown", "branch": "unknown", "tag": None}


def get_ecog_model_version():
    try:
        return version("ecog_foundation_model")
    except PackageNotFoundError:
        return "unknown"


class CheckpointManager:
    def __init__(self, model, optimizer=None, config=None, lr_scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.lr_scheduler = lr_scheduler

    def save(self, path, tags=None):
        # Remove any left over image masks before saving.
        self.model.initialize_mask(None)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "ecog_model_version": get_ecog_model_version(),
        }

        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.config:
            checkpoint["model_config"] = asdict(self.config)
        if self.lr_scheduler:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        if tags:
            checkpoint["tags"] = tags

        torch.save(checkpoint, path)
        logger.debug(f"[CheckpointManager] ✔ Saved to {path}")
        if tags:
            logger.debug(f"[CheckpointManager] 🏷 Tags: {tags}")

    def load(self, path, strict=True, required_tags=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # 1. Load model weights
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # 2. Optimizer (optional)
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        # 3. Version check
        saved_version = checkpoint.get("ecog_model_version", "unknown")
        current_version = get_ecog_model_version()
        if saved_version != current_version:
            msg = (
                f"[CheckpointManager] ❗ Version mismatch: "
                f"checkpoint={saved_version}, installed={current_version}"
            )
            if strict:
                raise RuntimeError(msg)
            else:
                logger.warning(msg)

        # 4. User tag check
        checkpoint_tags = checkpoint.get("tags", {})
        if required_tags:
            mismatches = []
            for key, expected in required_tags.items():
                actual = checkpoint_tags.get(key, None)
                if actual != expected:
                    mismatches.append(f"{key}: expected {expected}, got {actual}")
            if mismatches:
                msg = f"[CheckpointManager] ❗ Tag mismatch:\n  " + "\n  ".join(
                    mismatches
                )
                if strict:
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        # 5. Log what we loaded
        logger.debug(f"[CheckpointManager] 🔍 ecog_model_version: {saved_version}")
        if checkpoint_tags:
            logger.debug(f"[CheckpointManager] 🏷 Loaded tags: {checkpoint_tags}")

        return checkpoint
