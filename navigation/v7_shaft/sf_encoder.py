"""
Custom multi-input observation encoder for Sample Factory (SF v2).

WHY THIS FILE EXISTS
--------------------
SB3's MultiInputPolicy handled the dict obs space automatically — it detected
the depth key and routed it through a built-in NatureCNN, and the state key
through a small MLP, then concatenated the two. Sample Factory has no
equivalent automatic multi-input policy. You must write the encoder yourself
as a subclass of SF's Encoder base class.

This file re-implements exactly the same architecture SB3 used, so the learning
dynamics should be similar even though the training framework is different.

ARCHITECTURE
------------
  depth (1, H, W)    ->  DepthCNN   ->  512-D feature
  state (9,)         ->  MLP(64,64) ->  128-D feature
                                          └─ concatenate -> 576-D
                                          -> SF's shared actor-critic head

The 576-D vector goes into SF's built-in policy/value heads (linear layers).
We do not touch those — only this encoder is custom.

DEPTH SHAPE NOTE
----------------
ScopeColonEnv stores depth channel-first: obs["depth"] shape = (1, H, W).
This is non-standard for gym (gym uses H, W, C) but matches PyTorch conv
convention. When SF batches observations the tensor becomes (B, 1, H, W) —
exactly what nn.Conv2d expects. No permutation needed.
"""

import torch
import torch.nn as nn

# Sample Factory runs learner/inference in child processes. On this transfer
# environment cuDNN can fail to initialize in those subprocesses, while plain
# CUDA kernels work. Disable cuDNN so GPU mode remains usable instead of
# crashing during the first convolution.
torch.backends.cudnn.enabled = False

from sample_factory.model.encoder import Encoder
from sample_factory.algo.utils.context import global_model_factory


# ---------------------------------------------------------------------------
# Sub-network: NatureCNN
# ---------------------------------------------------------------------------

class _DepthCNN(nn.Module):
    """
    Convolutional encoder for single-channel depth images (currently 64×64).

    Uses stride-2 3×3 convolutions throughout. n_flatten is computed
    dynamically via a torch.zeros probe so this works at any input resolution
    without code changes.

    Input:  (B, 1, H, W) float32    — normalised depth image, channel-first.
    Output: (B, 512) float32        — flat feature vector.

    Spatial sizes at 64×64 input:
      Conv1 (k=3, s=2): 64 → 31
      Conv2 (k=3, s=2): 31 → 15
      Conv3 (k=3, s=1): 15 → 13
      Flatten: 13×13×64 = 10816 → Linear → 512
    """

    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super().__init__()
        conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = conv(torch.zeros(1, *input_shape)).shape[1]

        self.net = nn.Sequential(
            conv,
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        torch.backends.cudnn.enabled = False
        with torch.backends.cudnn.flags(enabled=False):
            return self.net(x)


# ---------------------------------------------------------------------------
# Main encoder
# ---------------------------------------------------------------------------

class ScopeColonEncoder(Encoder):
    """
    Multi-input encoder for ScopeColonEnv's dict observation space.

    Sample Factory calls encoder.forward(obs_dict) where obs_dict is a dict
    of {key: tensor} with the batch dimension prepended. The encoder must
    return a single flat feature tensor of size get_out_size().

    SF then passes this feature vector to the shared policy head (outputs
    the action distribution mean/std) and value head (outputs V(s)).
    """

    STATE_DIM = 9   # must match scope_colon_env.py STATE_OBS_DIM (cmd_x, cmd_y, force_norm, net_fx, net_fz, last_action[3], tip_contact)
    CNN_OUT   = 512
    MLP_OUT   = 128

    def __init__(self, cfg, obs_space) -> None:
        # cfg  : SF DictConfig object — passed through; we read it if needed
        # obs_space : gymnasium.spaces.Dict — the env's observation_space
        super().__init__(cfg)

        self.depth_cnn = _DepthCNN(tuple(obs_space["depth"].shape))

        # MLP for the 9-D low-dimensional state vector.
        self.state_mlp = nn.Sequential(
            nn.Linear(self.STATE_DIM, 64),   # STATE_DIM=9
            nn.ReLU(),
            nn.Linear(64, self.MLP_OUT),
            nn.ReLU(),
        )

        self._out_size = self.CNN_OUT + self.MLP_OUT  # 640

    # ------------------------------------------------------------------

    def forward(self, obs_dict: dict) -> torch.Tensor:
        """
        obs_dict["depth"] : (B, 1, H, W) float32  — already channel-first
        obs_dict["state"] : (B, 9)       float32
        returns           : (B, 640)     float32  — 512 CNN + 128 MLP
        """
        depth = obs_dict["depth"].float()

        # Guard: if SF ever delivers depth channel-last (B, H, W, 1), fix it.
        # This should not happen given our obs_space definition but is cheap.
        if depth.ndim == 4 and depth.shape[1] != 1 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2).contiguous()

        state = obs_dict["state"].float()

        cnn_features   = self.depth_cnn(depth)   # (B, 512)
        state_features = self.state_mlp(state)    # (B,  64)

        return torch.cat([cnn_features, state_features], dim=-1)  # (B, 576)

    def get_out_size(self) -> int:
        return self._out_size


# ---------------------------------------------------------------------------
# Registration helpers
# ---------------------------------------------------------------------------

def make_scope_encoder(cfg, obs_space) -> ScopeColonEncoder:
    """
    Factory function — SF calls this to construct the encoder.
    Signature must be (cfg, obs_space) -> Encoder.
    """
    return ScopeColonEncoder(cfg, obs_space)


def register_scope_encoder() -> None:
    """
    Register our custom encoder with SF's global model factory.

    SF builds the actor-critic network at training startup. When it creates
    the encoder it calls every registered factory in order; our factory
    replaces the default conv/mlp encoder.

    Must be called BEFORE parse_sf_args() so the factory is in place when
    SF constructs the network.
    """
    global_model_factory().register_encoder_factory(make_scope_encoder)
