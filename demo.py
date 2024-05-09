import math

import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F

from torchvision.utils import save_image

from einops import rearrange

from matplotlib import pyplot as plt
from matplotlib import image as mpimg

from model.convolution_7_32 import Deblur

target = torch.from_numpy(mpimg.imread("./sample/x.png"))
target = rearrange(target, "h w c -> 1 c h w")

input = torch.from_numpy(mpimg.imread("./sample/y.png"))
input = rearrange(input, "h w c -> 1 c h w")
input_psnr = 10*math.log10(1/F.mse_loss(input.clamp(min=0, max=1), target))

state = torch.load("./checkpoint/convolution_7_32.pth", map_location=torch.device('cpu'))
model = Deblur(in_channels=3, embed_channels=32)
model.load_state_dict(state["model_state"])
model.eval()

with torch.no_grad():
  output = model(input)
  output_psnr = 10*math.log10(1/F.mse_loss(output.clamp(min=0, max=1), target))
  print('PSNR: %.3f, ISNR: %.3f' % (output_psnr, output_psnr - input_psnr))
  save_image(output[0, :, :, :], './sample/xhat.png')
  plt.imshow(rearrange(output[0, :, :, :], 'c h w -> h w c'))
  plt.show()