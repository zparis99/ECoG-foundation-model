import torch
import subprocess

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


class CheckpointManager:
    def __init__(self, model, optimizer=None, config=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config

    def save(self, path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "git_info": get_git_info(),
        }
        if self.optimizer:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.config:
            checkpoint["model_config"] = self.config.__dict__

        torch.save(checkpoint, path)
        logger.debug(f"[CheckpointManager] Saved to {path}")

    def load(self, path, strict=True, validate_git=True):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if validate_git and "git_info" in checkpoint:
            self._validate_git_info(checkpoint["git_info"], strict)

        return checkpoint

    def _validate_git_info(self, saved_info, strict):
        current = get_git_info()

        warnings = []
        non_strict_warnings = []
        if saved_info["commit"] != current["commit"]:
            warnings.append(
                f"Commit mismatch: {saved_info['commit']} vs {current['commit']}"
            )
        if saved_info["tag"] and saved_info["tag"] != current["tag"]:
            warnings.append(f"Tag mismatch: {saved_info['tag']} vs {current['tag']}")

        # Non strict warnings will not crash on strict.
        if saved_info["branch"] != current["branch"]:
            non_strict_warnings.append(
                f"Branch mismatch: {saved_info['branch']} vs {current['branch']}"
            )

        if warnings:
            logger.warning("[CheckpointManager] ⚠️ Git mismatch detected:")
            for w in warnings:
                logger.warning("  •", w)
            if strict:
                raise RuntimeError(
                    "Git metadata mismatch — checkpoint may be incompatible."
                )
        if non_strict_warnings:
            logger.warning("[CheckpointManager] ⚠️ Non-crucial Git mismatch detected:")
            for w in non_strict_warnings:
                logger.warning("  •", w)
        else:
            logger.debug("[CheckpointManager] ✅ Git version matches checkpoint.")
