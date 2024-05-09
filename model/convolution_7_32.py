import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

from utils.utils_deblur import psf2otf, otf2psf

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

class Kernel(nn.Module):
  def __init__(self, in_channels, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.eps = 1e-8
    self.beta_scale = 1e-2
    self.kern_scale = 1e2
    self.zeta_scale = 10
    self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
    self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1))
    self.zeta = nn.Parameter(torch.ones(1, in_channels, 1, 1))
    self.lamda = nn.Parameter(torch.full((1, in_channels, 1, 1), 0.02))

  def forward(self, kern, blur, sharp):
    # Kernel initialization
    delta = otf2psf(psf2otf(torch.ones((kern.shape[0], kern.shape[1], 1, 1), device=blur.device), shape=(blur.shape[-2], blur.shape[-1])))
    Fkern = psf2otf(kern, shape=(blur.shape[-2], blur.shape[-1]))

    # Blur feature
    Fblur = fft.fft2(blur)

    # Intermediate sharp image
    Fsharp = fft.fft2(sharp)
    
    numerator = self.zeta*self.zeta_scale*Fkern.conj()*Fblur+Fsharp
    denominator = self.zeta*self.zeta_scale*Fkern.abs().square()+1

    sharp = torch.real(fft.ifft2(numerator/denominator))
    sharp = F.relu(sharp-self.lamda)-F.relu(-sharp-self.lamda)
    Fsharp = fft.fft2(sharp)

    # Kernel
    numerator = self.gamma*Fsharp.conj()*Fblur+Fkern
    denominator = self.gamma*Fsharp.abs().square()+1

    kern = torch.real(otf2psf(numerator/denominator))
    kernmax = torch.logsumexp(kern*self.kern_scale, dim=(-2, -1), keepdim=True)/self.kern_scale
    kern = F.relu(kern-(self.beta*self.beta_scale*kernmax))
    kernsum = torch.sum(kern, dim=(-2, -1), keepdim=True)
    kern = (kern+delta*self.eps)/(kernsum+self.eps)

    return kern, sharp

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
    self.kernel_0 = Kernel(in_channels=embed_channels)
    self.downsample_00 = Downsample(
      in_channels=embed_channels,
      out_channels=embed_channels*2)
    self.downsample_01 = Downsample(
      in_channels=embed_channels,
      out_channels=embed_channels*2)
    self.downsample_02 = Downsample(
      in_channels=embed_channels,
      out_channels=embed_channels*2)
    
    self.encode_1 = Feature(
      in_channels=embed_channels*2,
      kernel_size=9)
    self.kernel_1 = Kernel(in_channels=embed_channels*2)
    self.downsample_10 = Downsample(
      in_channels=embed_channels*2,
      out_channels=embed_channels*4)
    self.downsample_11 = Downsample(
      in_channels=embed_channels*2,
      out_channels=embed_channels*4)
    self.downsample_12 = Downsample(
      in_channels=embed_channels*2,
      out_channels=embed_channels*4)
    
    self.encode_2 = Feature(
      in_channels=embed_channels*4,
      kernel_size=9)
    self.kernel_2 = Kernel(in_channels=embed_channels*4)
    self.downsample_20 = Downsample(
      in_channels=embed_channels*4,
      out_channels=embed_channels*8)
    self.downsample_21 = Downsample(
      in_channels=embed_channels*4,
      out_channels=embed_channels*8)
    self.downsample_22 = Downsample(
      in_channels=embed_channels*4,
      out_channels=embed_channels*8)
    
    # Bottleneck
    self.bottleneck = Feature(
      in_channels=embed_channels*8,
      kernel_size=9)
    self.kernel_bottleneck = Kernel(in_channels=embed_channels*8)
    
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
    kern0, sharp0 = self.kernel_0(kern0, blur0, sharp0)

    kern1 = self.downsample_00(kern0)
    blur1 = self.downsample_01(blur0)
    sharp1 = self.downsample_02(sharp0)

    blur1 = self.encode_1(blur1)
    kern1, sharp1 = self.kernel_1(kern1, blur1, sharp1)

    kern2 = self.downsample_10(kern1)
    blur2 = self.downsample_11(blur1)
    sharp2 = self.downsample_12(sharp1)

    blur2 = self.encode_2(blur2)
    kern2, sharp2 = self.kernel_2(kern2, blur2, sharp2)
    
    kern3 = self.downsample_20(kern2)
    blur3 = self.downsample_21(blur2)
    sharp3 = self.downsample_22(sharp2)

    # Bottleneck
    blur3 = self.bottleneck(blur3)
    kern3, sharp3 = self.kernel_bottleneck(kern3, blur3, sharp3)

    # Decoder
    up0 = self.upsample_0(sharp3)
    deconv0 = torch.cat((up0, sharp2), dim=1)
    deconv0 = self.decode_0(deconv0)

    up1 = self.upsample_1(deconv0)
    deconv1 = torch.cat((up1, sharp1), dim=1)
    deconv1 = self.decode_1(deconv1)

    up2 = self.upsample_2(deconv1)
    deconv2 = torch.cat((up2, sharp0), dim=1)
    deconv2 = self.decode_2(deconv2)

    # Output projection
    output = x + self.output_proj(deconv2)

    return output