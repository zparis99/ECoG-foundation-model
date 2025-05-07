# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# MAE-ST: https://github.com/facebookresearch/mae_st
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import copy
from mae_st_util import video_vit


def pearson_correlation(x1, x2):
    """Compute pearson correlation between x1 and x2.

    Args:
        x1 (Tensor): shape [N]
        x2 (Tensor): shape [N]
    """
    return F.cosine_similarity(
        x1 - x1.mean(),
        x2 - x2.mean(),
        dim=0,
    )


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        num_frames=16,
        t_patch_size=2,
        patch_embed=video_vit.PatchEmbed,
        no_qkv_bias=False,
        sep_pos_embed=True,
        trunc_init=False,
        cls_embed=True,
        pred_t_dim=8,
        img_mask=None,
        pct_masks_to_decode=1,
        proj_drop=0.0,
        drop_path=0.0,
        **kwargs,
    ):
        """Initialize a Masked Autoencoder with Vision Transformer backbone for video processing.

        This class implements a masked autoencoder that operates on video data using a Vision
        Transformer architecture. It supports both spatial and temporal masking/encoding of video frames.

        Args:
            img_size (int, optional): Size of input image (assuming square). Defaults to 224.
            patch_size (int, optional): Size of patches in pixels to be extracted from image. Defaults to 16.
            in_chans (int, optional): Number of input channels. Defaults to 3.
            embed_dim (int, optional): Dimension of encoder embedding. Defaults to 1024.
            depth (int, optional): Number of transformer blocks in encoder. Defaults to 24.
            num_heads (int, optional): Number of attention heads in encoder. Defaults to 16.
            decoder_embed_dim (int, optional): Dimension of decoder embedding. Defaults to 512.
            decoder_depth (int, optional): Number of transformer blocks in decoder. Defaults to 8.
            decoder_num_heads (int, optional): Number of attention heads in decoder. Defaults to 16.
            mlp_ratio (float, optional): Ratio of MLP hidden dim to embedding dim. Defaults to 4.0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
            num_frames (int, optional): Number of frames in input video. Defaults to 16.
            t_patch_size (int, optional): Size of temporal patches. Defaults to 2.
            patch_embed (nn.Module, optional): Module for patch embedding. Defaults to video_vit.PatchEmbed.
            no_qkv_bias (bool, optional): If True, disable bias in attention. Defaults to False.
            sep_pos_embed (bool, optional): If True, use separate spatial and temporal position embeddings. Defaults to True.
            trunc_init (bool, optional): If True, use truncated normal initialization. Defaults to False.
            cls_embed (bool, optional): If True, use classification token. Defaults to True.
            pred_t_dim (int, optional): Number of time predictions to make (i.e. if t_patch_size is 4 and num_frames
                is 20 a pred_t_dim=5 will predict 5 total frames one for each patch, if pred_t_dim is 10 it will predict
                10 frames). Essentially it is a way to increase efficiency of training if neighboring frames are highly
                correlated which is not always true with neural data. Defaults to 8.
            img_mask (torch.Tensor, optional): Mask indicating which pixels are present/absent. True means present, false means
                absent. Can also be instantiated using initialize_mask(). Defaults to None.
            pct_masks_to_decode (float, optional): Percentage of masked patches to decode. Defaults to 1.
            proj_drop (float, optional): Probability of drop out in projection layer of attention blocks.
            drop_path (float, optional): Probability of drop path in attention blocks.
            **kwargs: Additional arguments passed to parent class.

        The model architecture consists of:
            1. A patch embedding layer that processes both spatial and temporal dimensions
            2. An encoder with transformer blocks
            3. A decoder that reconstructs masked patches
            4. Optional separate position embeddings for spatial and temporal dimensions
            5. Support for masked regions in input images/videos

        Note:
            - The model assumes square input images (img_size x img_size)
            - Position embeddings can be either separate for spatial/temporal dimensions or combined
            - The decoder reconstructs only a portion of masked patches when pct_masks_to_decode < 1
        """
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames
        self.embed_dim = embed_dim
        self.pct_masks_to_decode = pct_masks_to_decode
        self.patch_size = patch_size

        self.masked_input_norm = video_vit.MaskedBatchNorm(in_chans)

        self.patch_embed = patch_embed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames,
            t_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                video_vit.Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop=proj_drop,
                    drop_path=drop_path,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size**2 * in_chans,
            bias=True,
        )

        self.initialize_mask(img_mask)
        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_mask(self, img_mask):
        """Provide a mask of which pixels in the image are not present. True means value is masked out, False means it is present.

        img_mask: bool tensor of shape [H, W]
        """
        if img_mask is not None:
            img_mask = torch.as_tensor(img_mask > 0).float()

            H, W = img_mask.shape
            img_mask_patches = self.patchify(
                img_mask.view(1, 1, 1, H, W).repeat(
                    1, self.patch_embed.in_chans, self.pred_t_dim, 1, 1
                )
            )

            patch_mask = (
                rearrange(
                    img_mask,
                    "(h ph) (w pw) -> (h w) (ph pw)",
                    ph=self.patch_embed.patch_size[0],
                    pw=self.patch_embed.patch_size[1],
                )
                .any(dim=1)
                .float()
            )
            (patch_mask_indices,) = patch_mask.nonzero(as_tuple=True)

            self.register_buffer("img_mask", img_mask)
            self.register_buffer("img_mask_patches", img_mask_patches)
            self.register_buffer("patch_mask", patch_mask)
            self.register_buffer("patch_mask_indices", patch_mask_indices)
            self.n_mask_patches = int(
                len(patch_mask_indices) * self.pct_masks_to_decode
            )
        else:
            self.register_buffer("img_mask", None)
            self.register_buffer("img_mask_patches", None)
            self.register_buffer("patch_mask", None)
            self.register_buffer("patch_mask_indices", None)
            self.n_mask_patches = None

    def patchify(self, imgs):
        """
        imgs: (N, C, T, H, W)
        x: (N, L, patch_size**2 *C)
        """
        N, C, T, H, W = imgs.shape
        ph, pw = self.patch_embed.patch_size
        u = self.t_pred_patch_size
        assert H % ph == 0 and W % pw == 0 and T % u == 0
        h = H // ph
        w = W // pw
        t = T // u

        x = imgs.reshape(shape=(N, C, t, u, h, ph, w, pw))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * ph * pw * C))
        self.patch_info = (N, C, T, H, W, ph, pw, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *C)
        imgs: (N, C, H, W)
        """
        N, C, T, H, W, ph, pw, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, ph, pw, C))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, C, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio, use_contrastive_loss=False):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size
        assert L == T * H * W

        # adjust number to keep relative to image mask
        if self.img_mask is not None:
            len_keep = int(T * self.n_mask_patches * (1 - mask_ratio))
        else:
            len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # shift missing patches to not be selected
        if self.img_mask is not None:
            noise = noise.view(N, T, H * W)
            noise = noise + (1.0 - self.patch_mask)
            noise = noise.view(N, L)

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        if not use_contrastive_loss:
            x_masked = torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
            )
        else:
            x_masked1 = torch.gather(
                x,
                dim=1,
                index=ids_keep[:, : len_keep // 2].unsqueeze(-1).repeat(1, 1, D),
            )
            x_masked2 = torch.gather(
                x,
                dim=1,
                index=ids_keep[:, len_keep // 2 : len_keep]
                .unsqueeze(-1)
                .repeat(1, 1, D),
            )

        if not use_contrastive_loss:
            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([N, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)
        else:
            # generate the binary mask: 0 is keep, 1 is remove
            mask1 = torch.ones([N, L], device=x.device)
            mask2 = torch.ones([N, L], device=x.device)
            mask1[:, : len_keep // 2] = 0
            mask2[:, len_keep // 2 : len_keep] = 0
            # unshuffle to get the binary mask
            mask1 = torch.gather(mask1, dim=1, index=ids_restore)
            mask2 = torch.gather(mask2, dim=1, index=ids_restore)

        if not use_contrastive_loss:
            return x_masked, mask, ids_restore, ids_keep
        else:
            return [x_masked1, x_masked2], [mask1, mask2], ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio, use_contrastive_loss=False):
        x = self.patch_embed(x)

        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        if not use_contrastive_loss:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
            x = x.view(N, -1, C)
        else:
            [x1, x2], [mask1, mask2], ids_restore, ids_keep = self.random_masking(
                x, mask_ratio, use_contrastive_loss=use_contrastive_loss
            )
            x1 = x1.view(len(x1), -1, C)
            x2 = x2.view(len(x2), -1, C)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            if not use_contrastive_loss:
                x = torch.cat((cls_tokens, x), dim=1)
            else:
                x1 = torch.cat((cls_tokens, x1), dim=1)
                x2 = torch.cat((cls_tokens, x2), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        if not use_contrastive_loss:
            x = x.view([N, -1, C]) + pos_embed
        else:
            x1 = x1.view([len(x1), -1, C]) + pos_embed[:, : x1.shape[1]]
            x2 = x2.view([len(x2), -1, C]) + torch.cat(
                (pos_embed[:, :1], pos_embed[:, x1.shape[1] :]), dim=1
            )

        if not use_contrastive_loss:
            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
        else:
            # apply Transformer blocks
            for blk in self.blocks:
                x1 = blk(x1)
                x2 = blk(x2)
            x1 = self.norm(x1)
            x2 = self.norm(x2)

        if not use_contrastive_loss:
            if self.cls_embed:
                # remove cls token
                x = x[:, 1:, :]

            return x, mask, ids_restore
        else:
            if self.cls_embed:
                # remove cls token
                x1 = x1[:, 1:, :]
                x2 = x2[:, 1:, :]

            return [x1, x2], [mask1, mask2], ids_restore

    def forward_encoder_with_mask(self, x, ids_keep):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)
        # mask out tokens
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, C))

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        x = x.view([N, -1, C]) + pos_embed

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_decoder(self, x, ids_restore, use_contrastive_loss=False):
        N = x.shape[0]
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn

        # drop patches outside image mask (and then only keep a subset a la VideoMAE2)
        if self.img_mask is not None:
            if self.cls_embed:
                decoder_cls_tokens, x = x[:, :1, :], x[:, 1:, :]
            x = x.view([N, T, H * W, C])
            # x = x[:, :, self.patch_mask_indices]

            # drop patches randomly to preserve memory (VideoMAE2 approach)
            included_patches = self.patch_mask_indices
            num_to_select = int(self.pct_masks_to_decode * len(included_patches))
            selected_idx = torch.randperm(len(included_patches))[:num_to_select]
            included_patches = included_patches[selected_idx]
            x = x[:, :, included_patches]

            x = x.view([N, T * self.n_mask_patches, C])
            if self.cls_embed:
                x = torch.cat((decoder_cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]

        # fill outside mask with zeros
        if self.img_mask is not None:
            C = x.shape[-1]
            x = x.view([N, T, self.n_mask_patches, C])
            x_ = torch.zeros([N, T, H * W, C], dtype=x.dtype, device=x.device)
            x = x_.scatter(
                2,
                included_patches.view(1, 1, -1, 1).expand(N, T, self.n_mask_patches, C),
                x,
            )
            x = x.view([N, T * H * W, C])

        return x

    def forward_loss(self, imgs, pred, mask, alpha):
        """
        imgs: [N, C, T, H, W]
        pred: [N, t*h*w, u*p*p*C]
        mask: [N, t*h*w], 0 is keep, 1 is remove,
        alpha: Loss weighting between correlation and MSE given by alpha * -correlation + (1 - alpha) * mse
        """
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )
        target = self.patchify(_imgs)

        # Calculate correlation of masked patches
        B, L, C = target.shape
        expanded_mask = mask.repeat_interleave(C, axis=1).view(B, L, C).bool()
        masked_imgs = target.masked_fill(expanded_mask, torch.nan)
        masked_pred = pred.masked_fill(expanded_mask, torch.nan)
        correlation = pearson_correlation(
            masked_imgs[~masked_imgs.isnan()], masked_pred[~masked_pred.isnan()]
        )

        # Calculate mse
        mse = (pred - target) ** 2
        if self.img_mask is not None:
            # exclude missing pixels from loss
            mask = mask.unsqueeze(-1) * self.img_mask_patches
        else:
            mse = mse.mean(dim=-1)  # [N, L], mean loss per patch

        mse = (mse * mask).sum() / mask.sum()  # mean loss on removed patches

        # Loss is weighted sum of mse and correlation
        loss = alpha * (1 - correlation) / 2 + (1 - alpha) * mse

        return loss, mse, correlation

    def forward_cyclic_loss(self, pred1, pred2, mask):
        """
        mask1 and mask2 encoder outputs should be the same since they are predicting the same held-out true mask
        """
        loss = (pred1 - pred2) ** 2
        if self.img_mask is not None:
            # exclude missing pixels from loss
            mask = mask.unsqueeze(-1) * self.img_mask_patches
        else:
            loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_input_norm(self, x):
        return self.masked_input_norm(x, self.img_mask)

    def forward(
        self,
        imgs,
        mask_ratio=0.75,
        use_contrastive_loss=False,
        forward_features=False,
        global_pool=True,
        cls_forward=False,
        alpha=0.5,
    ):
        imgs = self.masked_input_norm(imgs, self.img_mask)
        # TODO: Break this out and test.
        if forward_features:
            # embed patches
            x = self.patch_embed(imgs)
            N, T, L, C = x.shape  # T: temporal; L: spatial

            x = x.reshape([N, T * L, C])

            # append cls token
            if self.cls_embed:
                cls_token = self.cls_token
                cls_tokens = cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)

            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.input_size[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.input_size[1] * self.input_size[2],
                    dim=1,
                )
                if self.cls_embed:
                    pos_embed = torch.cat(
                        [
                            self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                            pos_embed,
                        ],
                        1,
                    )
            else:
                pos_embed = self.pos_embed[:, :, :]
            x = x + pos_embed

            # drop patches outside image mask
            if self.img_mask is not None:
                if self.cls_embed:
                    cls_tokens, x = x[:, :1, :], x[:, 1:, :]
                x = x.view([N, T, L, C])
                x = x[:, :, self.patch_mask_indices]
                x = x.reshape([N, T * self.n_mask_patches, C])
                if self.cls_embed:
                    x = torch.cat((cls_tokens, x), dim=1)

            # apply Transformer blocks
            for blk in self.blocks:
                x = blk(x)

            if global_pool:
                if self.cls_embed:
                    # remove cls token
                    x = x[:, 1:, :]
                x = x.mean(dim=1)
            elif cls_forward:
                x = x[:, :1, :]
            return x
        else:
            latent, mask, ids_restore = self.forward_encoder(
                imgs, mask_ratio, use_contrastive_loss=use_contrastive_loss
            )
            if not use_contrastive_loss:
                pred = self.forward_decoder(
                    latent, ids_restore, use_contrastive_loss=use_contrastive_loss
                )  # [N, L, p*p*C]
                loss, mse, correlation = self.forward_loss(imgs, pred, mask, alpha)
                return loss, mse, pred, mask, latent, correlation

    def forward_head(self, x):
        # classifier
        x = self.norm(x)
        # x = self.fc_norm(x)
        x = self.dropout(x)
        x = self.head(x)

        return x

    def mask_fill(self, x):
        N, L, C = x.shape
        T = self.patch_embed.t_grid_size
        H, W = self.patch_embed.grid_size
        assert L == T * self.n_mask_patches

        x = x.view(N, T, -1, C)
        x_ = torch.zeros([N, T, H * W, C], dtype=x.dtype, device=x.device)
        x = x_.scatter(
            2,
            self.patch_mask_indices.view(1, 1, -1, 1).expand(N, T, -1, C),
            x,
        )
        return x
