import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space

from Model.policy import ILPolicy
from Model.encoders.instruction_encoder import InstructionEncoder, InstructionBertEncoder
from Model.encoders.resnet_encoders import VlnResnetDepthEncoder, TorchVisionCLIP, RgbdCLEncoder
from Model.encoders.rnn_state_encoder import build_rnn_state_encoder
from Model.aux_losses import AuxLosses
from Model.utils.CN import CN

from src.common.param import args


class LandmarkSlotAttention(nn.Module):
    """Iterative landmark slot refinement with local visual patch grounding."""

    def __init__(self, feature_dim: int, proj_dim: int, num_iterations: int = 5, num_slots: int = 5):
        super().__init__()
        self.background_slots = nn.Parameter(torch.randn(1, num_slots, feature_dim) * 0.02)
        self.text_proj = nn.Linear(feature_dim, proj_dim)
        self.visual_proj = nn.Linear(feature_dim, proj_dim)
        self.visual_value = nn.Linear(feature_dim, feature_dim)
        self.slot_gru = nn.GRUCell(feature_dim, feature_dim)
        self.residual_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.text_norm = nn.LayerNorm(feature_dim)
        self.visual_norm = nn.LayerNorm(feature_dim)
        self.scale = proj_dim ** -0.5
        self.num_iterations = num_iterations
        self.eps = 1e-6

    def forward(
        self,
        text_slots: torch.Tensor,
        visual_tokens: torch.Tensor,
        slot_mask: torch.Tensor = None,
        return_attention: bool = False,
    ):
        # text_slots: [B, D, K], visual_tokens: [B, D, N]
        slots = text_slots.transpose(1, 2).contiguous()  # [B, K, D]
        visual = visual_tokens.transpose(1, 2).contiguous()  # [B, N, D]
        slots = F.normalize(slots, dim=-1)
        visual = F.normalize(visual, dim=-1)
        if slot_mask is not None:
            slot_mask = slot_mask.bool()
            background_slots = self.background_slots[:, : slots.size(1), :].expand(slots.size(0), -1, -1)
            slots = torch.where(slot_mask.unsqueeze(-1), slots, background_slots)

        visual_norm = self.visual_norm(visual)
        k = self.visual_proj(visual_norm)  # [B, N, C]
        v = self.visual_value(visual_norm)  # [B, N, D]

        attn_for_vis = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.text_proj(self.text_norm(slots))  # [B, K, C]

            logits = torch.bmm(q, k.transpose(1, 2)) * self.scale  # [B, K, N]
            # Slot attention normalizes over slots first, forcing landmarks to
            # compete for each visual token instead of independently lighting up.
            attn = F.softmax(logits, dim=1) + self.eps
            attn = torch.nan_to_num(attn, nan=self.eps, posinf=1.0, neginf=self.eps)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)

            updates = torch.bmm(attn, v)  # [B, K, D]
            slots = self.slot_gru(
                updates.reshape(-1, updates.size(-1)),
                slots_prev.reshape(-1, slots_prev.size(-1)),
            ).view_as(slots_prev)
            slots = slots + self.residual_mlp(slots)

            # For visualization, expose the final slot-to-token distribution.
            attn_for_vis = attn / attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        
        if return_attention:
            return slots.transpose(1, 2).contiguous(), attn_for_vis  # [B, D, K], [B, K, N]
        return slots.transpose(1, 2).contiguous()  # [B, D, K]


class CMAPolicy(ILPolicy):
    def __init__(
            self,
            observation_space: Space,
            action_space: Space,
            out_model_config=None,
            device=torch.device("cpu"),
    ):
        super().__init__(
            CMAOurNet(
                observation_space=observation_space,
                num_actions=action_space.n,
                out_model_config=out_model_config,
                device=device,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
            cls, observation_space: Space, action_space: Space, out_model_config=None,
            device=torch.device("cpu"),
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            out_model_config=out_model_config,
            device=device,
        )


class CMAOurNet(nn.Module):
    r"""A cross-modal attention (CMA) network that contains:
    Instruction encoder
    Depth encoder
    RGB encoder
    CMA state encoder
    """

    def __init__(self, observation_space: Space, num_actions, out_model_config=None, device=torch.device("cpu")):
        super().__init__()
        self.device = device

        # 参数设置
        model_config = CN.clone()
        model_config.STATE_ENCODER_hidden_size = 512
        model_config.STATE_ENCODER_rnn_type = 'GRU'
        model_config.PROGRESS_MONITOR_use = args.PROGRESS_MONITOR_use
        model_config.PROGRESS_MONITOR_alpha = args.PROGRESS_MONITOR_alpha
        self.model_config = model_config

        # instruction encoder
        self.instruction_output_size = 512

        # depth encoder
        self.depth_encoder = VlnResnetDepthEncoder(observation_space)

        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                self.depth_encoder.output_size,
            ),
            nn.ReLU(True),
        )

        # rgb encoder
        self.rgb_encoder = TorchVisionCLIP(observation_space, device)

        # RGBD-CL encoder (pretrained)
        # self.rgbd_cl_encoder = RgbdCLEncoder(observation_space, device)

        # action encoder
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # state encoder
        hidden_size = model_config.STATE_ENCODER_hidden_size
        self._hidden_size = hidden_size

        # the attn state decoder
        rnn_input_size = self.depth_encoder.output_size
        rnn_input_size += self.rgb_encoder.output_size
        # rnn_input_size += self.rgbd_cl_encoder.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER_rnn_type,
            num_layers=1,
        )

        # 01计算加权后的文本特征
        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(self.instruction_output_size, hidden_size // 2, 1)

        # 02分层任务推理中的细粒度动作/地标模块
        self.component_state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.component_action_k = nn.Linear(self.instruction_output_size, hidden_size // 2)
        self.component_landmark_k = nn.Linear(self.instruction_output_size, hidden_size // 2)

        self.action_elem_q = nn.Linear(hidden_size, hidden_size // 2)
        self.action_elem_k = nn.Conv1d(self.instruction_output_size, hidden_size // 2, 1)

        self.landmark_elem_q = nn.Linear(hidden_size, hidden_size // 2)
        self.landmark_elem_k = nn.Conv1d(self.instruction_output_size, hidden_size // 2, 1)
        self.landmark_slot_attention = LandmarkSlotAttention(
            feature_dim=self.instruction_output_size,
            proj_dim=hidden_size // 2,
        )

        ###
        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        self.hierarchy_rnn2_input_size = (
                self._hidden_size
                + self.instruction_output_size
                + self.instruction_output_size
                + self.prev_action_embedding.embedding_dim
        )

        self.second_state_compress_hierarchy = nn.Sequential(
            nn.Linear(
                self.hierarchy_rnn2_input_size,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER_rnn_type,
            num_layers=1,
        )
        self._output_size = self._hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self):
        return self._output_size

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self):
        if self.model_config.PROGRESS_MONITOR_use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc, nci -> ni", q, k)

        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1.0, neginf=0.0)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        return torch.einsum("ni, nci -> nc", attn, v), attn

    @staticmethod
    def _gather_subtask_vector(embedding: torch.Tensor, subtask_index: torch.Tensor) -> torch.Tensor:
        index = subtask_index.view(-1, 1, 1).expand(-1, embedding.size(1), 1)
        return torch.gather(embedding, dim=2, index=index).squeeze(2)

    @staticmethod
    def _gather_subtask_elements(embedding: torch.Tensor, subtask_index: torch.Tensor) -> torch.Tensor:
        index = subtask_index.view(-1, 1, 1, 1).expand(
            -1, embedding.size(1), embedding.size(2), 1
        )
        return torch.gather(embedding, dim=3, index=index).squeeze(3)

    def forward(self, observations, rnn_states, prev_actions, masks):
        r"""
        instruction_embedding: [batch_size x INSTRUCTION_ENCODER.output_size]
        depth_embedding: [batch_size x DEPTH_ENCODER.output_size]
        rgb_embedding: [batch_size x RGB_ENCODER.output_size]
        """
        # action编码器
        prev_actions = self.prev_action_embedding(((prev_actions.float() + 1) * masks).long().view(-1))  # [B * 32]

        # depth图像编码器
        depth_embedding = self.depth_encoder(observations)  # B * (128+64) * 4 * 4
        depth_embedding = torch.flatten(depth_embedding, 2)  # B * (128+64) * 16
        depth_in = self.depth_linear(depth_embedding)  # B * 128

        # rgb图像编码器
        rgb_in, rgb_embedding = self.rgb_encoder(observations)

        # # ------------   基于对比学习的编码器   ------------
        # # Resize depth from 256x256 to 224x224
        # depth_bchw = observations["depth"].permute(0, 3, 1, 2).float()
        # depth_resized = F.interpolate(depth_bchw, size=(224, 224), mode="bilinear", align_corners=False)
        # observations["depth"] = depth_resized.permute(0, 2, 3, 1).contiguous()
        #
        # rgbd_cl_features = self.rgbd_cl_encoder(observations)  # [B, fea_dim]
        # # -----------------------------------------------

        subtask_embedding = observations['subtask_embedding']

        # attn state
        # 更新包state_in的特征维度800
        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0: self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0: self.state_encoder.num_recurrent_layers],
            masks,
        )

        # 利用注意力机制在hidden_size // 2的维度计算相似性并获取加权后的文本特征text_embedding
        text_state_q = self.state_q(state)
        text_state_k = self.text_k(subtask_embedding)
        text_mask = (subtask_embedding == 0.0).all(dim=1)
        _, attn_mask = self._attn(text_state_q, text_state_k, subtask_embedding, text_mask)
        subtask_index = torch.argmax(attn_mask, dim=1)

        if subtask_embedding.size(2) > 0:
            subtask_index = subtask_index.clamp(min=0, max=subtask_embedding.size(2) - 1)
        invalid_subtask = text_mask.all(dim=1)
        subtask_index = torch.where(invalid_subtask, torch.zeros_like(subtask_index), subtask_index)

        has_hierarchy_inputs = all(
            k in observations for k in ["nA_embedding", "nL_embedding", "Ae_embedding", "Le_embedding"]
        )
        if not has_hierarchy_inputs:
            missing_keys = [
                k for k in ["nA_embedding", "nL_embedding", "Ae_embedding", "Le_embedding"]
                if k not in observations
            ]
            raise KeyError(
                "AirVLN03 requires hierarchical task observation keys; "
                f"missing observation keys: {missing_keys}"
            )

        nA_embedding = observations["nA_embedding"]
        nL_embedding = observations["nL_embedding"]
        Ae_embedding = observations["Ae_embedding"]
        Le_embedding = observations["Le_embedding"]

        gA_comp = self._gather_subtask_vector(nA_embedding, subtask_index)
        gL_comp = self._gather_subtask_vector(nL_embedding, subtask_index)
        gA_elem = self._gather_subtask_elements(Ae_embedding, subtask_index)
        gL_elem = self._gather_subtask_elements(Le_embedding, subtask_index)
        Le_mask = observations.get("Le_mask", None)
        if Le_mask is not None:
            Le_mask = self._gather_subtask_vector(Le_mask, subtask_index)
        else:
            empty_landmark = Le_embedding[:, :, 0, -1].unsqueeze(2)
            Le_mask = (gL_elem - empty_landmark).abs().mean(dim=1) > 1e-5

        comp_q = self.component_state_q(state)
        score_a = torch.einsum("nc,nc->n", comp_q, self.component_action_k(gA_comp))
        score_l = torch.einsum("nc,nc->n", comp_q, self.component_landmark_k(gL_comp))
        component_alpha = F.softmax(torch.stack([score_a, score_l], dim=1), dim=1)
        component_alpha = torch.nan_to_num(component_alpha, nan=0.5, posinf=1.0, neginf=0.0)
        component_alpha = component_alpha / component_alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
        no_landmark_component = ~Le_mask.bool().any(dim=1, keepdim=True)
        action_only_alpha = torch.cat(
            [
                torch.ones_like(component_alpha[:, 0:1]),
                torch.zeros_like(component_alpha[:, 1:2]),
            ],
            dim=1,
        )
        component_alpha = torch.where(
            no_landmark_component.expand_as(component_alpha),
            action_only_alpha,
            component_alpha,
        )
        alpha_a = component_alpha[:, 0:1]
        alpha_l = component_alpha[:, 1:2]

        action_q = self.action_elem_q(state)
        action_k = self.action_elem_k(gA_elem)
        gA_tilde, _ = self._attn(action_q, action_k, gA_elem)

        gL_slots = self.landmark_slot_attention(
            gL_elem,
            rgb_embedding,
            slot_mask=Le_mask,
        )

        landmark_q = self.landmark_elem_q(state)
        landmark_k = self.landmark_elem_k(gL_slots)
        landmark_slot_mask = ~Le_mask.bool()
        no_landmark_slot = no_landmark_component.squeeze(1)
        landmark_slot_mask = landmark_slot_mask.masked_fill(no_landmark_slot.unsqueeze(1), False)
        gL_tilde, _ = self._attn(landmark_q, landmark_k, gL_slots, landmark_slot_mask)
        gL_tilde = torch.where(no_landmark_slot.unsqueeze(1), torch.zeros_like(gL_tilde), gL_tilde)

        action_refined = torch.nan_to_num(alpha_a * gA_tilde, nan=0.0, posinf=1.0, neginf=-1.0)
        landmark_refined = torch.nan_to_num(alpha_l * gL_tilde, nan=0.0, posinf=1.0, neginf=-1.0)

        # 训练稳定性监控（可在调试时读取）
        self.last_component_alpha_mean = component_alpha.mean().detach()
        self.last_component_alpha_var = component_alpha.var(unbiased=False).detach()

        x = torch.cat(
            [
                state,
                action_refined,
                landmark_refined,
                prev_actions,
            ],
            dim=1,
        )
        x = self.second_state_compress_hierarchy(x)
        (
            x,
            rnn_states_out[:, self.state_encoder.num_recurrent_layers:],
        ) = self.second_state_encoder(
            x,
            rnn_states[:, self.state_encoder.num_recurrent_layers:],
            masks,
        )

        if self.model_config.PROGRESS_MONITOR_use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR_alpha,
            )

        return x, rnn_states_out, subtask_index
