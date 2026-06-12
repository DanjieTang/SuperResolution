import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def zero_out(layer: nn.Module) -> nn.Module:
    """
    Zero out the parameters of a layer.

    :param layer: The layer to zero out.
    :return: The zeroed layer.
    """
    for p in layer.parameters():
        p.detach().zero_()
    return layer


class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding of the flow matching timestep, followed by an MLP."""

    def __init__(self, hidden_dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half_dim = self.frequency_dim // 2
        frequency = torch.exp(-math.log(10000) * torch.arange(half_dim, device=timestep.device) / half_dim)
        angle = timestep[:, None].float() * frequency[None, :]
        embedding = torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
        return self.mlp(embedding)


def modulate(tokens: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return tokens * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """Transformer block with adaLN-zero conditioning on the timestep."""

    def __init__(self, hidden_dim: int, num_head: int, mlp_ratio: int = 4):
        super().__init__()
        self.num_head = num_head

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.attention_output = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

        # Produces shift/scale/gate for the attention and MLP branches.
        # Zero-initialized so every block starts as the identity function
        self.modulation = zero_out(nn.Linear(hidden_dim, hidden_dim * 6))

    def attention(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, num_token, hidden_dim = tokens.shape

        query, key, value = torch.chunk(self.qkv_proj(tokens), 3, dim=-1)
        query = query.view(batch_size, num_token, self.num_head, -1).transpose(1, 2)
        key = key.view(batch_size, num_token, self.num_head, -1).transpose(1, 2)
        value = value.view(batch_size, num_token, self.num_head, -1).transpose(1, 2)

        # Fused kernel never materializes the full attention matrix
        tokens = F.scaled_dot_product_attention(query, key, value)

        tokens = tokens.transpose(1, 2).reshape(batch_size, num_token, hidden_dim)
        return self.attention_output(tokens)

    def forward(self, tokens: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = torch.chunk(
            self.modulation(F.silu(conditioning)), 6, dim=-1
        )

        tokens = tokens + gate_attn[:, None] * self.attention(modulate(self.norm1(tokens), shift_attn, scale_attn))
        tokens = tokens + gate_mlp[:, None] * self.mlp(modulate(self.norm2(tokens), shift_mlp, scale_mlp))
        return tokens


class FinalLayer(nn.Module):
    """Modulated projection from tokens back to latent patches."""

    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.modulation = zero_out(nn.Linear(hidden_dim, hidden_dim * 2))
        self.projection = zero_out(nn.Linear(hidden_dim, patch_size ** 2 * out_channels))

    def forward(self, tokens: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        shift, scale = torch.chunk(self.modulation(F.silu(conditioning)), 2, dim=-1)
        tokens = modulate(self.norm(tokens), shift, scale)
        return self.projection(tokens)


class DiT(nn.Module):
    """
    Diffusion transformer for outpainting, working directly in pixel space.

    Predicts the flow matching velocity for the noisy image, conditioned on
    the timestep, the masked (known pixels only) image, and the known-region
    mask. Inputs are concatenated along channels:
    3 noisy image + 3 masked image + 1 mask = 7 in pixel space, or
    4 noisy latent + 4 masked latent + 1 mask = 9 with --use_vae.

    The patch embedding folds each patch_size x patch_size tile into one
    token, the same tiling trick as in the super resolution attention.
    """

    def __init__(
        self,
        image_size: int = 512,
        patch_size: int = 16,
        in_channels: int = 7,
        out_channels: int = 3,
        hidden_dim: int = 512,
        depth: int = 12,
        num_head: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.token_per_side = image_size // patch_size

        self.patch_embed = nn.Conv2d(in_channels, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, self.token_per_side ** 2, hidden_dim) * 0.02)
        self.timestep_embedding = TimestepEmbedding(hidden_dim)

        self.blocks = nn.ModuleList([DiTBlock(hidden_dim, num_head) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_dim, patch_size, out_channels)

    def unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = tokens.shape[0]
        side = self.token_per_side
        tokens = tokens.view(batch_size, side, side, self.patch_size, self.patch_size, self.out_channels)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4)
        return tokens.reshape(batch_size, self.out_channels, side * self.patch_size, side * self.patch_size)

    def forward(
        self,
        noisy_image: torch.Tensor,
        timestep: torch.Tensor,
        known_image: torch.Tensor,
        known_mask: torch.Tensor,
    ) -> torch.Tensor:
        tensor = torch.cat([noisy_image, known_image, known_mask], dim=1)

        tokens = self.patch_embed(tensor).flatten(2).transpose(1, 2)
        tokens = tokens + self.positional_encoding

        conditioning = self.timestep_embedding(timestep)
        for block in self.blocks:
            tokens = block(tokens, conditioning)

        tokens = self.final_layer(tokens, conditioning)
        return self.unpatchify(tokens)
