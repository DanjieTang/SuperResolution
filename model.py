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
    def __init__(self, embedding_dim: int, image_size: int, head_dim: int = 64, channel_per_group: int = 16):
        super().__init__()
        self.head_dim: int = head_dim
        self.num_head: int = embedding_dim // head_dim
        self.scale: float = head_dim ** -0.5
        self.num_pixel = image_size ** 2
        self.gnorm1 = nn.GroupNorm(embedding_dim // channel_per_group, embedding_dim)
        self.gnorm2 = nn.GroupNorm(embedding_dim // channel_per_group, embedding_dim)

        # QKV projection
        self.qkv_proj = nn.Linear(embedding_dim, embedding_dim * 3)

        # Output layer
        self.output = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)

        # Positional embedding for patches, registered as a buffer so it follows model.to(device)
        self.register_buffer(
            "positional_encoding",
            self.sinusoidal_positional_encoding_2d(image_size, image_size, embedding_dim),
            persistent=False,
        )

        # Feed Forward Layer
        self.ffn1 = nn.Conv2d(embedding_dim, embedding_dim * 8, kernel_size=1)
        self.ffn2 = nn.Conv2d(embedding_dim * 8, embedding_dim, kernel_size=1)

    def sinusoidal_positional_encoding_2d(self, height: int, width: int, channel: int) -> torch.Tensor:
        """
        Generate a 2D sinusoidal positional encoding.

        :param height: The height of the encoding.
        :param width: The width of the encoding.
        :param channel: The number of channels in the encoding.
        :return: A tensor of shape (channel, width, height) containing the 2D positional encoding.
        """
        if channel % 2 != 0:
            raise ValueError("The 'channel' dimension must be an even number.")

        # First, build in (height, width, channel) format
        pe = torch.zeros(height, width, channel)

        half_ch = channel // 2

        # Precompute the exponent for row and column
        row_div_term = torch.exp(
            -math.log(10000.0) * (torch.arange(0, half_ch, 2).float() / half_ch)
        )
        col_div_term = torch.exp(
            -math.log(10000.0) * (torch.arange(0, half_ch, 2).float() / half_ch)
        )

        for h in range(height):
            for w in range(width):
                # Encode row index (h) into the first half of the channels
                for i in range(0, half_ch, 2):
                    pe[h, w, i]     = math.sin(h * row_div_term[i // 2])
                    pe[h, w, i + 1] = math.cos(h * row_div_term[i // 2])

                # Encode column index (w) into the second half of the channels
                for j in range(0, half_ch, 2):
                    pe[h, w, half_ch + j]     = math.sin(w * col_div_term[j // 2])
                    pe[h, w, half_ch + j + 1] = math.cos(w * col_div_term[j // 2])

        # Permute to get the shape (channel, width, height).
        # Currently pe is (height, width, channel) = (H, W, C)
        # We want (C, W, H), so we do permute(2, 1, 0).
        pe = pe.permute(2, 1, 0)  # => (channel, width, height)

        return pe

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        skip_tensor = tensor

        tensor = self.gnorm1(tensor)

        # Reshape for self attention
        batch_size, channel, height, width = tensor.shape
        tensor = tensor + self.positional_encoding
        tensor = tensor.view(batch_size, channel, self.num_pixel)
        tensor = tensor.permute(0, 2, 1)

        tensor = self.qkv_proj(tensor)

        query, key, value = torch.chunk(tensor, 3, dim=-1)
        query = query.view(batch_size, self.num_pixel, self.num_head, self.head_dim)
        key = key.view(batch_size, self.num_pixel, self.num_head, self.head_dim)
        value = value.view(batch_size, self.num_pixel, self.num_head, self.head_dim)

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
        tensor = tensor.view(batch_size, self.num_pixel, channel)
        tensor = tensor.permute(0, 2, 1)
        tensor = tensor.reshape(batch_size, channel, height, width)
        tensor = self.output(tensor)

        tensor = tensor + skip_tensor

        # Feed Forward Layer
        tensor = self.gnorm2(tensor)
        tensor = self.ffn1(tensor)
        tensor = F.relu(tensor)
        tensor = self.ffn2(tensor)

        return tensor

class SuperResolution(nn.Module):
    def __init__(self, embedding_dim: list[int] = [3, 128, 256], input_image_size: int = 64):
        super().__init__()
        self.module_list = nn.ModuleList()

        for i in range(len(embedding_dim) - 1):
            self.module_list.append(ResBlock(in_channel=embedding_dim[i], out_channel=embedding_dim[i+1]))
            if i == 0:
                self.module_list.append(SelfAttentionBlock(embedding_dim=embedding_dim[i+1], image_size=input_image_size))
            self.module_list.append(ResBlock(in_channel=embedding_dim[i+1], out_channel=embedding_dim[i+1], up=True))
        self.module_list.append(ResBlock(in_channel=embedding_dim[-1], out_channel=3))

    def forward(self, tensor):
        for module in self.module_list:
            tensor = module(tensor)
        return tensor
