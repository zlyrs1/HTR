import os
import time
from gym import spaces
import numpy as np
from typing import Dict
from pathlib import Path

import torch
import open_clip
from torch import Tensor
from torch import distributed as distrib
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from types import SimpleNamespace

from Model.utils import ddppo_resnet_utils as resnet
from utils.logger import logger
from src.common.param import args
from RGBD_CL.model import MAFF
from RGBD_CL.modelTool.extractor import E_SPA as RGBDCL_E_SPA, E_SPEC as RGBDCL_E_SPEC


class RunningMeanAndVar(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))
        self.register_buffer("_count", torch.zeros(()))
        self._mean: torch.Tensor = self._mean
        self._var: torch.Tensor = self._var
        self._count: torch.Tensor = self._count

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            n = x.size(0)
            # We will need to do reductions (mean) over the channel dimension,
            # so moving channels to the first dimension and then flattening
            # will make those faster.  Further, it makes things more numerically stable
            # for fp16 since it is done in a single reduction call instead of
            # multiple
            x_channels_first = (
                x.transpose(1, 0).contiguous().view(x.size(1), -1)
            )
            new_mean = x_channels_first.mean(-1, keepdim=True)
            new_count = torch.full_like(self._count, n)

            if distrib.is_initialized():
                distrib.all_reduce(new_mean)
                distrib.all_reduce(new_count)
                new_mean /= distrib.get_world_size()

            new_var = (
                (x_channels_first - new_mean).pow(2).mean(dim=-1, keepdim=True)
            )

            if distrib.is_initialized():
                distrib.all_reduce(new_var)
                new_var /= distrib.get_world_size()

            new_mean = new_mean.view(1, -1, 1, 1)
            new_var = new_var.view(1, -1, 1, 1)

            m_a = self._var * (self._count)
            m_b = new_var * (new_count)
            M2 = (
                m_a
                + m_b
                + (new_mean - self._mean).pow(2)
                * self._count
                * new_count
                / (self._count + new_count)
            )

            self._var = M2 / (self._count + new_count)
            self._mean = (self._count * self._mean + new_count * new_mean) / (
                self._count + new_count
            )

            self._count += new_count

        inv_stdev = torch.rsqrt(
            torch.max(self._var, torch.full_like(self._var, 1e-2))
        )
        # This is the same as
        # (x - self._mean) * inv_stdev but is faster since it can
        # make use of addcmul and is more numerically stable in fp16
        return torch.addcmul(-self._mean * inv_stdev, x, inv_stdev)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
    ):
        super().__init__()

        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
            spatial_size = observation_space.spaces["rgb"].shape[0] // 2
        else:
            self._n_input_rgb = 0

        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
            spatial_size = observation_space.spaces["depth"].shape[0] // 2
        else:
            self._n_input_depth = 0

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial = int(
                spatial_size * self.backbone.final_spatial_compress
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(after_compression_flat_size / (final_spatial ** 2))
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial,
                final_spatial,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB
            cnn_input.append(rgb_observations)

        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)

            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class VlnResnetDepthEncoder(nn.Module):
    def __init__(self, observation_space):
        super().__init__()

        if args.policy_type in ['seq2seq']:
            output_size = 128; self._output_size = output_size
            checkpoint = str(Path(args.project_prefix) / 'DATA/models/ddppo-models/gibson-2plus-resnet50.pth')
            backbone = "resnet50"
            resnet_baseplanes = 32
            normalize_visual_inputs = False
            spatial_output = False
        elif args.policy_type in ['cma']:
            output_size = 128; self._output_size = output_size
            checkpoint = str(Path(args.project_prefix) / 'DATA/models/ddppo-models/gibson-2plus-resnet50.pth')
            backbone = "resnet50"
            resnet_baseplanes = 32
            normalize_visual_inputs = False
            spatial_output = True
        else:
            raise NotImplementedError

        self.visual_encoder = ResNetEncoder(
            spaces.Dict({"depth": observation_space.spaces["depth"]}),
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
        )

        for param in self.visual_encoder.parameters():
            param.requires_grad_(False)

        if checkpoint != "NONE":
            ddppo_weights = torch.load(checkpoint, map_location=torch.device('cpu'))
            weights_dict = {}
            for k, v in ddppo_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue
                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v
            del ddppo_weights
            self.visual_encoder.load_state_dict(weights_dict, strict=True)

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(np.prod(self.visual_encoder.output_shape), output_size),
                nn.ReLU(True),
            )
        else:
            self.spatial_embeddings = nn.Embedding(
                self.visual_encoder.output_shape[1]
                * self.visual_encoder.output_shape[2],
                64,
            )

            self.output_shape = list(self.visual_encoder.output_shape)
            self.output_shape[0] += self.spatial_embeddings.embedding_dim
            self.output_shape = tuple(self.output_shape)

    #
    @property
    def output_size(self):
        return self._output_size

    #
    def forward(self, observations):
        """
        Args:
            observations: [BATCH, HEIGHT, WIDTH, CHANNEL]
        Returns:
            [BATCH, OUTPUT_SIZE]
        """
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            x = self.visual_encoder(observations)

        if self.spatial_output:
            b, c, h, w = x.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=x.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([x, spatial_features], dim=1)
        else:
            return self.visual_fc(x)


class TorchVisionResNet50(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    #
    def __init__(
        self,
        observation_space,
        device,
    ):
        super().__init__()

        if args.policy_type in ['seq2seq']:
            output_size = 256; self._output_size = output_size
            spatial_output = False
        elif args.policy_type in ['cma']:
            output_size = 256; self._output_size = output_size
            spatial_output = True
        else:
            raise NotImplementedError

        self.resnet_layer_size = 2048
        self.device = device
        linear_layer_input_size = 0

        self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        obs_size_0 = observation_space.spaces["rgb"].shape[0]
        obs_size_1 = observation_space.spaces["rgb"].shape[1]
        if obs_size_0 != 224 or obs_size_1 != 224:
            logger.warn(
                "TorchVisionResNet50: observation size is not conformant to expected ResNet input size [3x224x224]"
            )
        linear_layer_input_size += self.resnet_layer_size

        self.resize_transform = transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        self.cnn = models.resnet50(pretrained=True)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            # 如果是req2req，输出1个全局特征
            self.output_shape = (output_size,)
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:
            # 在cma中修改了resnet50在avgpool的实现，不再是1*1的全局池化
            # 希望得到4*4的空间大小
            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))
                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (self.resnet_layer_size + self.spatial_embeddings.embedding_dim, 4, 4)

        self.layer_extract = self.cnn._modules.get("avgpool")

    #
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    #
    @property
    def output_size(self):
        return self._output_size

    #
    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(1, dtype=torch.float32, device=self.device)

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        if "rgb_features" in observations:
            # train
            resnet_output = observations["rgb_features"]
        else:
            # collect [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            rgb_observations = self.resize_transform(rgb_observations)
            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()
            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.activation(
                self.fc(torch.flatten(resnet_output, 1))
            )


class TorchVisionCLIP(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """
    def __init__(self, observation_space, device):
        super().__init__()

        self.device = device
        self._output_size = 512

        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent
        clip_weights_path = project_root / 'src' / 'vlnce_src' / 'laion' / 'CLIP-ViT-B-32-laion2B-s34B-b79K.bin'
        clip_image_size = 448

        self.CLIPEncoder, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32',
                                                                             pretrained=str(clip_weights_path),
                                                                             force_image_size=clip_image_size)

        for param in self.CLIPEncoder.parameters():
            param.requires_grad = False
        self.CLIPEncoder.eval()

        clip_normalize = None
        for transform in self.preprocess.transforms:
            if isinstance(transform, transforms.Normalize):
                clip_normalize = transform
                break

        self.preprocess_tensor = transforms.Compose([
            transforms.Resize((clip_image_size, clip_image_size),
                              interpolation=transforms.InterpolationMode.BICUBIC,
                              antialias=True),
            clip_normalize,
        ])

        self.clip_surgery_depth = int(getattr(args, "clip_surgery_depth", 6))
        self.layer_extract = nn.Identity()

        patch_size = self.CLIPEncoder.visual.patch_size
        patch_size = patch_size[0] if isinstance(patch_size, tuple) else patch_size
        clip_grid_size = clip_image_size // patch_size
        self.output_shape = (512, clip_grid_size, clip_grid_size)

    @property
    def is_blind(self):
        return self.rgb_c == 0

    #
    @property
    def output_size(self):
        return self._output_size

    @staticmethod
    def _attention_block_forward(block, x: torch.Tensor):
        if hasattr(block, "attention"):
            return block.attention(q_x=block.ln_1(x))
        return block.ln_attn(block.attn(block.ln_1(x)))

    @staticmethod
    def _ffn_block_forward(block, x: torch.Tensor):
        return block.mlp(block.ln_2(x))

    @staticmethod
    def _consistent_attention(block, x: torch.Tensor):
        attn = block.attn
        if isinstance(attn, nn.MultiheadAttention):
            embed_dim = attn.embed_dim
            num_heads = attn.num_heads
            head_dim = embed_dim // num_heads
            _, _, value_weight = attn.in_proj_weight.chunk(3, dim=0)
            value_bias = None
            if attn.in_proj_bias is not None:
                _, _, value_bias = attn.in_proj_bias.chunk(3, dim=0)
            v = F.linear(x, value_weight, value_bias)
            batch_size, seq_len, _ = v.shape
            v = v.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
            attn_map = torch.matmul(v, v.transpose(-1, -2)) * (head_dim ** -0.5)
            attn_map = attn_map.softmax(dim=-1)
            out = torch.matmul(attn_map, v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
            return attn.out_proj(out)

        value_weight = attn.in_proj_weight.chunk(3, dim=0)[2]
        value_bias = attn.in_proj_bias.chunk(3, dim=0)[2] if attn.in_proj_bias is not None else None
        v = F.linear(x, value_weight, value_bias)
        batch_size, seq_len, channels = v.shape
        num_heads = attn.num_heads
        head_dim = channels // num_heads
        v = v.reshape(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        if getattr(attn, "logit_scale", None) is not None:
            attn_map = torch.matmul(F.normalize(v, dim=-1), F.normalize(v, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(attn.logit_scale, max=attn.logit_scale_max).exp()
            attn_map = attn_map * logit_scale.view(1, num_heads, 1, 1)
        else:
            attn_map = torch.matmul(v, v.transpose(-1, -2)) * (head_dim ** -0.5)
        attn_map = attn_map.softmax(dim=-1)
        out = torch.matmul(attn_map, v).permute(0, 2, 1, 3).reshape(batch_size, seq_len, channels)
        out = attn.out_proj(out)
        return attn.out_drop(out)

    def _vanilla_ln_post_tokens(self, rgb_image: torch.Tensor):
        token_holder = {}

        def hook(_module, _inputs, output):
            token_holder["tokens"] = output

        h = self.CLIPEncoder.visual._modules.get("ln_post").register_forward_hook(hook)
        self.CLIPEncoder(rgb_image)
        h.remove()
        return self.layer_extract(token_holder["tokens"])

    def _clip_surgery_ln_post_tokens(self, rgb_image: torch.Tensor):
        visual = self.CLIPEncoder.visual
        required_attrs = ["conv1", "transformer", "ln_post", "proj"]
        if not all(hasattr(visual, attr) for attr in required_attrs):
            return self._vanilla_ln_post_tokens(rgb_image)

        x = visual.conv1(rgb_image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        class_embedding = visual.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype)
        x = torch.cat([class_embedding, x], dim=1)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.patch_dropout(x)
        x = visual.ln_pre(x)

        blocks = list(visual.transformer.resblocks)
        surgery_depth = max(0, min(self.clip_surgery_depth, len(blocks)))
        surgery_start = len(blocks) - surgery_depth

        for block in blocks[:surgery_start]:
            x = block(x)

        if surgery_depth > 0:
            x_new = x.clone()
            x_ori = x
            for block in blocks[surgery_start:]:
                norm_ori = block.ln_1(x_ori)
                x_ori = x_ori + self._attention_block_forward(block, x_ori)
                x_ori = x_ori + self._ffn_block_forward(block, x_ori)
                x_new = x_new + self._consistent_attention(block, norm_ori)
            x = x_new + x_ori

        x = visual.ln_post(x)
        return self.layer_extract(x)

    #
    def forward(self, observations):
        r"""Sends RGB observations through CLIP and returns CLS and patch tokens."""

        if "rgb_features" in observations:
            rgb_CLIP = observations["rgb_features"]
        else:
            # collect [BATCH x CHANNEL x (HEIGHT x WIDTH + 1)]
            # AirSim observations arrive in BGR order; CLIP expects RGB.
            rgb_observations = observations["rgb"].flip(dims=[-1]).permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0
            rgb_observations = self.preprocess_tensor(rgb_observations)
            rgb_CLIP = self._clip_surgery_ln_post_tokens(rgb_observations.contiguous())

        all_rgb_CLIP = rgb_CLIP @ self.CLIPEncoder.visual.proj
        cls_rgb_CLIP = all_rgb_CLIP[:, 0, :]
        patch_rgb_CLIP = all_rgb_CLIP[:, 1:, :].permute(0, 2, 1)

        return cls_rgb_CLIP, patch_rgb_CLIP



class CtVisMAFF(nn.Module):
    def __init__(self, fea_dims, vis_dims):
        super(CtVisMAFF, self).__init__()

        self.f_fusion = MAFF(fea_dims)
        self.FC = nn.Linear(fea_dims, vis_dims)

    def forward(self, fea):
        # MAFF
        out = self.f_fusion(fea)
        # FC
        out = out.view(out.size(0), -1)
        output = self.FC(out)

        return output


class RgbdCLEncoder(nn.Module):
    """
    - Split RGB into patches (7x7, aligned by center).
    - For each RGB patch center, take the aligned 21x21 depth patch.
    - Feed RGB patches to encoder_spec.pth (E_SPEC), depth patches to encoder_spa.pth (E_SPA).
    - Average patch features to form global spectral + spatial features, then concat.
    """

    def __init__(self, observation_space, device: torch.device):
        super().__init__()
        self.device = device

        self.patch_rgb = 32
        self.patch_depth = 64

        self.half_rgb = (self.patch_rgb - 1) // 2
        self.half_depth = (self.patch_depth - 1) // 2

        # 加载预训练好的基于对比学习的多模态空间/光谱编码器
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent
        spa_path = project_root / 'RGBD_CL' / 'checkpoints' / 'RGBD' / 'learning_rate_1.2' / 'encoder_spa.pth'
        spec_path = project_root / 'RGBD_CL' / 'checkpoints' / 'RGBD' / 'learning_rate_1.2' / 'encoder_spec.pth'

        spa_channels = [16, 32, 64, 64]
        spec_channels = [64, 128, 256, 256]
        rgbdcl_cfg = SimpleNamespace(
            en_spa_input_channel=1,
            en_spa_channels=spa_channels,
            en_spec_input_channel=3,
            en_spec_channels=spec_channels,
        )

        # 创建原始的、未经编译的模型实例
        self.encoder_spa = RGBDCL_E_SPA(rgbdcl_cfg)
        self.encoder_spec = RGBDCL_E_SPEC(rgbdcl_cfg)

        # 加载模型权重到原始模型中
        self.encoder_spa.load_state_dict(torch.load(spa_path, map_location="cpu"), strict=True)
        self.encoder_spec.load_state_dict(torch.load(spec_path, map_location="cpu"), strict=True)

        # 使用torch.compile进行优化
        if hasattr(torch, 'compile'):
            self.encoder_spa = torch.compile(self.encoder_spa)
            self.encoder_spec = torch.compile(self.encoder_spec)

        # 预训练好的模型参数固定
        for p in self.encoder_spa.parameters():
            p.requires_grad_(False)
        for p in self.encoder_spec.parameters():
            p.requires_grad_(False)

        self.encoder_spa.eval()
        self.encoder_spec.eval()

        fea_dims = int(spa_channels[-1]) + int(spec_channels[-1])
        self.CtvisMAFFNet = CtVisMAFF(fea_dims, 128)
        self._output_size = 128

        # 分块推理的块大小：控制单次编码器输入的最大样本数，避免 B*P 过大时显存溢出
        # 训练时 B=500, P=121 → B*P=60500；chunk_size=64 则分 ~945 块，显存峰值固定
        self.enc_chunk_size = 500

    @property
    def output_size(self) -> int:
        return self._output_size

    def _make_centers(self, H: int, W: int, device: torch.device) -> Tensor:
        """
        Patch-center construction logic (no padding):
        - Use depth patch (21x21) boundary to define valid center range:
          min_c* = half_depth, max_c* = (H/W - 1 - half_depth),
          so any 21x21 crop is guaranteed in-bounds.
        - Inside that valid range, place centers on a regular grid with step=patch_rgb (7),
          so RGB 7x7 patches are as non-overlapping as possible and cover most of the image.
          Depth 21x21 patches aligned by these centers will naturally have overlap, which is acceptable.
        """
        min_cy = self.half_depth
        max_cy = H - 1 - self.half_depth
        min_cx = self.half_depth
        max_cx = W - 1 - self.half_depth

        ys = torch.arange(min_cy, max_cy + 1, step=self.patch_rgb, device=device, dtype=torch.long)
        xs = torch.arange(min_cx, max_cx + 1, step=self.patch_rgb, device=device, dtype=torch.long)
        if ys.numel() == 0:
            ys = torch.tensor([int((min_cy + max_cy) // 2)], device=device, dtype=torch.long)
        if xs.numel() == 0:
            xs = torch.tensor([int((min_cx + max_cx) // 2)], device=device, dtype=torch.long)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)  # (P, 2)
        return centers

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        rgb = observations["rgb"]
        depth = observations["depth"]

        B, H, W, _ = rgb.shape

        rgb_t = rgb.permute(0, 3, 1, 2).float() / 255.0  # [B,3,H,W]
        depth_t = depth.permute(0, 3, 1, 2).float()      # [B,1,H,W]

        centers = self._make_centers(H, W, rgb_t.device)  # (P, 2)
        P = centers.shape[0]

        start_time = time.time()

        spec_feats = []
        spa_feats = []
        with torch.no_grad(), torch.cuda.amp.autocast():
            for cy, cx in centers.tolist():
                rgb_patch = rgb_t[:, :, cy - self.half_rgb: cy + self.half_rgb + 1,
                            cx - self.half_rgb: cx + self.half_rgb + 1]
                depth_patch = depth_t[:, :, cy - self.half_depth: cy + self.half_depth + 1,
                              cx - self.half_depth: cx + self.half_depth + 1]

                spec_feats.append(self.encoder_spec(rgb_patch))
                spa_feats.append(self.encoder_spa(depth_patch))

        elapse_time = time.time() - start_time
        print('encoder eps_time %2.2f s,' % elapse_time)

        spec_mean = torch.stack(spec_feats, dim=0).mean(dim=0)
        spa_mean = torch.stack(spa_feats, dim=0).mean(dim=0)
        fused = torch.cat([spec_mean.float() , spa_mean.float() ], dim=1)  # [B, fea_dims]

        feature_context = self.CtvisMAFFNet(fused.unsqueeze(2))  # [B, 128]

        elapse_time = time.time() - start_time
        print('total eps_time %2.2f s,' % elapse_time)

        return feature_context


class TorchVisionResNet50Place365(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.
    Based on PLACE 365 Pretrained Model

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self,
        observation_space,
        device,
    ):
        super().__init__()

        if args.policy_type in ['seq2seq']:
            output_size = 256; self._output_size = output_size
            spatial_output = False
        elif args.policy_type in ['cma']:
            output_size = 256; self._output_size = output_size
            spatial_output = True
        else:
            raise NotImplementedError

        self.device = device
        self.resnet_layer_size = 2048
        linear_layer_input_size = 0

        self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        obs_size_0 = observation_space.spaces["rgb"].shape[0]
        obs_size_1 = observation_space.spaces["rgb"].shape[1]
        if obs_size_0 != 224 or obs_size_1 != 224:
            logger.warn(
                "TorchVisionResNet50: observation size is not conformant to expected ResNet input size [3x224x224]"
            )
        linear_layer_input_size += self.resnet_layer_size

        if self.is_blind:
            self.cnn = nn.Sequential()
            return

        from scripts.get_feature import BuildModel_ResNet50_raw, model_param_convertor_raw
        place365_model = torch.load(
            str(Path(args.project_prefix) / 'DATA/models/resnet50/resnet50_places365.pth'),
            map_location=torch.device('cpu')
        )
        param = model_param_convertor_raw(place365_model['state_dict'])
        self.cnn = BuildModel_ResNet50_raw(param)

        # disable gradients for resnet, params frozen
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            self.fc = nn.Linear(linear_layer_input_size, output_size)
            self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.cnn.avgpool = SpatialAvgPool()
            self.cnn.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.layer_extract = self.cnn._modules.get("avgpool")

    #
    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    #
    @property
    def output_size(self):
        return self._output_size

    #
    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        def resnet_forward(observation):
            resnet_output = torch.zeros(
                1, dtype=torch.float32, device=self.device
            )

            def hook(m, i, o):
                resnet_output.set_(o)

            # output: [BATCH x RESNET_DIM]
            h = self.layer_extract.register_forward_hook(hook)
            self.cnn(observation)
            h.remove()
            return resnet_output

        if "rgb_features" in observations:
            resnet_output = observations["rgb_features"]
        else:
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
            rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
            rgb_observations = rgb_observations / 255.0  # normalize RGB
            resnet_output = resnet_forward(rgb_observations.contiguous())

        if self.spatial_output:
            b, c, h, w = resnet_output.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=resnet_output.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([resnet_output, spatial_features], dim=1)
        else:
            return self.activation(
                self.fc(torch.flatten(resnet_output, 1))
            )
