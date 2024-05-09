import torch
import torch.nn as nn

class Projection(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.proj = nn.Sequential(
      nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        padding=1)
    )

  def forward(self, x):
    x = self.proj(x) # B C H W -> B out_channels H W
    return x

class Downsample(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.conv = nn.Sequential(
      nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=2,
        padding=1)
    )

  def forward(self, x):
    x = self.conv(x) # B C H W -> B out_channels H//2 W//2
    return x

class Upsample(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.deconv = nn.Sequential(
      nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=2)
    )

  def forward(self, x):
    x = self.deconv(x) # B C H W -> B out_channels H*2 W*2
    return x

class Feature(nn.Module):
  def __init__(self, in_channels, kernel_size, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.conv_mix = nn.Sequential(
      nn.Conv2d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=kernel_size,
        padding=(kernel_size - 1) // 2,
        groups=in_channels),
      nn.GELU(),
      nn.BatchNorm2d(in_channels),
      nn.Conv2d(
        in_channels=in_channels,
        out_channels=in_channels,
        kernel_size=1),
      nn.GELU(),
      nn.BatchNorm2d(in_channels)
    )

  def forward(self, x):
    x = self.conv_mix(x) # B, C, H, W -> B, out_channels, H, W
    return x

class Deblur(nn.Module):
  def __init__(self, in_channels, embed_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # Input / Output
    self.input_proj = Projection(
      in_channels=in_channels,
      out_channels=embed_channels)
    self.output_proj = Projection(
      in_channels=embed_channels*2,
      out_channels=in_channels)

    # Encoder
    self.encode_0 = Feature(
      in_channels=embed_channels,
      kernel_size=9)
    self.downsample_00 = Downsample(
      in_channels=embed_channels,
      out_channels=embed_channels*2)
    self.encode_1 = Feature(
      in_channels=embed_channels*2,
      kernel_size=9)
    self.downsample_10 = Downsample(
      in_channels=embed_channels*2,
      out_channels=embed_channels*4)
    self.encode_2 = Feature(
      in_channels=embed_channels*4,
      kernel_size=9)
    self.downsample_20 = Downsample(
      in_channels=embed_channels*4,
      out_channels=embed_channels*8)

    # Bottleneck
    self.bottleneck = Feature(
      in_channels=embed_channels*8,
      kernel_size=9)

    # Decoder
    self.upsample_0 = Upsample(
      in_channels=embed_channels*8,
      out_channels=embed_channels*4)
    self.decode_0 = Feature(
      in_channels=embed_channels*8,
      kernel_size=9)
    self.upsample_1 = Upsample(
      in_channels=embed_channels*8,
      out_channels=embed_channels*2)
    self.decode_1 = Feature(
      in_channels=embed_channels*4,
      kernel_size=9)
    self.upsample_2 = Upsample(
      in_channels=embed_channels*4,
      out_channels=embed_channels)
    self.decode_2 = Feature(
      in_channels=embed_channels*2,
      kernel_size=9)

  def forward(self, x):
    # Input projection
    blur0 = self.input_proj(x)

    # Initialise kernel and sharp features
    b, c, h, w = blur0.shape
    kern0 = torch.zeros((b, c, h, w), device=x.device)
    kern0[:, :, h//2, w//2] = 1
    sharp0 = torch.zeros((b, c, h, w), device=x.device)

    # Encoder
    blur0 = self.encode_0(blur0)

    blur1 = self.downsample_00(blur0)
    blur1 = self.encode_1(blur1)

    blur2 = self.downsample_10(blur1)
    blur2 = self.encode_2(blur2)

    blur3 = self.downsample_20(blur2)

    # Bottleneck
    blur3 = self.bottleneck(blur3)

    # Decoder
    up0 = self.upsample_0(blur3)
    deconv0 = torch.cat((up0, blur2), dim=1)
    deconv0 = self.decode_0(deconv0)

    up1 = self.upsample_1(deconv0)
    deconv1 = torch.cat((up1, blur1), dim=1)
    deconv1 = self.decode_1(deconv1)

    up2 = self.upsample_2(deconv1)
    deconv2 = torch.cat((up2, blur0), dim=1)
    deconv2 = self.decode_2(deconv2)

    # Output projection
    output = x + self.output_proj(deconv2)

    return output