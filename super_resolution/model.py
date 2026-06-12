import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, up: bool = False):
        super().__init__()

        # Upsampling or downsampling only for skip connection
        self.up = up

        # Normalization layers
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)

        # Convolution layers
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.conv2 = self.zero_out(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))

        # Skip connection
        if in_channel != out_channel or up:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
                nn.Upsample(scale_factor=2) if up else nn.Identity(),
            )
        else:
            self.skip_connection = nn.Identity()

    def zero_out(self, layer: nn.Module) -> nn.Module:
        """
        Zero out the parameters of a layer.

        :param layer: The layer to zero out.
        :return: The zeroed layer.
        """
        for p in layer.parameters():
            p.detach().zero_()
        return layer

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        skip_tensor = self.skip_connection(tensor)

        # Main path
        tensor = self.norm1(tensor)
        tensor = F.relu(tensor)
        if self.up:
            tensor = F.interpolate(tensor, scale_factor=2)
        tensor = self.conv1(tensor)
        tensor = self.norm2(tensor)
        tensor = F.relu(tensor)
        tensor = self.conv2(tensor)

        tensor += skip_tensor
        return tensor


class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, image_size: int, patch_size: int = 4, head_dim: int = 64, channel_per_group: int = 16):
        super().__init__()
        # Each patch_size x patch_size tile becomes one token, so the
        # attention matrix shrinks by patch_size^4
        self.patch_size: int = patch_size
        self.token_dim: int = embedding_dim * patch_size ** 2
        self.head_dim: int = head_dim
        self.num_head: int = self.token_dim // head_dim
        self.scale: float = head_dim ** -0.5
        self.num_token = (image_size // patch_size) ** 2
        self.gnorm1 = nn.GroupNorm(embedding_dim // channel_per_group, embedding_dim)
        self.gnorm2 = nn.GroupNorm(embedding_dim // channel_per_group, embedding_dim)

        # QKV projection
        self.qkv_proj = nn.Linear(self.token_dim, self.token_dim * 3)

        # Output layer
        self.output = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)

        # Learned positional embedding for patches
        self.positional_encoding = nn.Parameter(torch.randn(embedding_dim, image_size, image_size))

        # Feed Forward Layer
        self.ffn1 = nn.Conv2d(embedding_dim, embedding_dim * 8, kernel_size=1)
        self.ffn2 = nn.Conv2d(embedding_dim * 8, embedding_dim, kernel_size=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        skip_tensor = tensor

        tensor = self.gnorm1(tensor)

        # Reshape for self attention
        batch_size, _, height, width = tensor.shape
        tensor = tensor + self.positional_encoding

        # Fold each patch_size x patch_size tile into the channel dimension,
        # turning every tile into a single token
        tensor = F.pixel_unshuffle(tensor, self.patch_size)
        tensor = tensor.view(batch_size, self.token_dim, self.num_token)
        tensor = tensor.permute(0, 2, 1)

        tensor = self.qkv_proj(tensor)

        query, key, value = torch.chunk(tensor, 3, dim=-1)
        query = query.view(batch_size, self.num_token, self.num_head, self.head_dim)
        key = key.view(batch_size, self.num_token, self.num_head, self.head_dim)
        value = value.view(batch_size, self.num_token, self.num_head, self.head_dim)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Self attention
        attention_raw = torch.matmul(query, key.transpose(2, 3))
        attention_scaled = attention_raw * self.scale
        attention_score = torch.softmax(attention_scaled, dim=-1)
        value = torch.matmul(attention_score, value)

        # Reshape for self attention output
        tensor = value.transpose(1, 2).contiguous()
        tensor = tensor.view(batch_size, self.num_token, self.token_dim)
        tensor = tensor.permute(0, 2, 1)
        tensor = tensor.reshape(batch_size, self.token_dim, height // self.patch_size, width // self.patch_size)

        # Unfold tokens back to the full resolution feature map
        tensor = F.pixel_shuffle(tensor, self.patch_size)
        tensor = self.output(tensor)

        tensor = tensor + skip_tensor

        # Feed Forward Layer
        tensor = self.gnorm2(tensor)
        tensor = self.ffn1(tensor)
        tensor = F.relu(tensor)
        tensor = self.ffn2(tensor)

        return tensor

class SuperResolution(nn.Module):
    # Attention only at low resolutions: token count grows with
    # resolution^2, so high-res stages stay convolution-only
    MAX_ATTENTION_RESOLUTION = 64

    def __init__(self, embedding_dim: list[int] = [3, 128, 256], input_image_size: int = 64):
        super().__init__()
        self.module_list = nn.ModuleList()

        for i in range(len(embedding_dim) - 1):
            resolution = input_image_size * 2 ** i
            self.module_list.append(ResBlock(in_channel=embedding_dim[i], out_channel=embedding_dim[i+1]))
            if resolution <= self.MAX_ATTENTION_RESOLUTION:
                self.module_list.append(SelfAttentionBlock(embedding_dim=embedding_dim[i+1], image_size=resolution))
            self.module_list.append(ResBlock(in_channel=embedding_dim[i+1], out_channel=embedding_dim[i+1], up=True))
        self.module_list.append(ResBlock(in_channel=embedding_dim[-1], out_channel=3))

    def forward(self, tensor):
        for module in self.module_list:
            tensor = module(tensor)
        return tensor
