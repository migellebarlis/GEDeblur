
import torch
import torch.fft as fft

from matplotlib import image
from matplotlib import pyplot

from torchvision.utils import save_image

from einops import rearrange

from utils.utils_deblur import psf2otf, otf2psf

from model.convolution_7_32 import *

activation = {}
def get_activation(name):
  def hook(model, input, output):
    if name in ['kernel_0','kernel_1','kernel_2','kernel_3']:
      activation[name] = output[0].detach(), output[1].detach()
    else:
      activation[name] = output.detach()
  return hook

state = torch.load("./checkpoint/convolution_7_32.pth", map_location=torch.device('cpu'))

model = Deblur(in_channels=3, embed_channels=32)
model.load_state_dict(state["model_state"])
model.eval()

model.input_proj.register_forward_hook(get_activation('input_proj'))
model.encode_0.register_forward_hook(get_activation('encode_0'))
model.kernel_0.register_forward_hook(get_activation('kernel_0'))
model.downsample_00.register_forward_hook(get_activation('downsample_00'))
model.encode_1.register_forward_hook(get_activation('encode_1'))
model.kernel_1.register_forward_hook(get_activation('kernel_1'))
model.downsample_10.register_forward_hook(get_activation('downsample_10'))
model.encode_2.register_forward_hook(get_activation('encode_2'))
model.kernel_2.register_forward_hook(get_activation('kernel_2'))
model.downsample_20.register_forward_hook(get_activation('downsample_20'))
model.bottleneck.register_forward_hook(get_activation('encode_3'))
model.kernel_bottleneck.register_forward_hook(get_activation('kernel_3'))
model.decode_0.register_forward_hook(get_activation('decode_0'))
model.decode_1.register_forward_hook(get_activation('decode_1'))
model.decode_2.register_forward_hook(get_activation('decode_2'))
model.output_proj.register_forward_hook(get_activation('output_proj'))

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Total params: ', total_params)

min, max = model.input_proj.proj[0].weight.min(), model.input_proj.proj[0].weight.max()
weight_input_proj = (model.input_proj.proj[0].weight - min) / (max - min)
ij = 1
for i in range(8):
  for j in range(4):
    for k in range(3):
      ax = pyplot.subplot(8, 12, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      pyplot.imshow(weight_input_proj[i*4+j, k, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
pyplot.title("Input projection weights")
# pyplot.show()

# min, max = model.encode_0.conv_mix[0].weight.min(), model.encode_0.conv_mix[0].weight.max()
# weight_encode_0 = (model.encode_0.conv_mix[0].weight - min) / (max - min)
weight_encode_0 = model.encode_0.conv_mix[0].weight
ij = 1
for i in range(4):
  for j in range(8):
    ax = pyplot.subplot(4, 8, ij)
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.imshow(weight_encode_0[i*8+j, 0, :, :].detach().numpy(), extent=[0, 1, 0, 1])
    ij += 1
pyplot.title("Blur feature extractor 0 weights")
# pyplot.show()

input = torch.from_numpy(image.imread("09.png"))
input = rearrange(input, "h w c -> 1 c h w")
with torch.no_grad():
  sharp = model(input)
  
  input_proj = activation["input_proj"]
  
  ij = 1
  for i in range(4):
    for j in range(8):
      ax = pyplot.subplot(4, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(input_proj[0, i*8+j, :, :], f"./activations/input_proj_{ij:03d}.png")
      pyplot.imshow(input_proj[0, i*8+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("input_proj")
  # pyplot.show()

  blur_0 = activation["encode_0"]

  ij = 1
  for i in range(4):
    for j in range(8):
      ax = pyplot.subplot(4, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(blur_0[0, i*8+j, :, :], f"./activations/blur_0_{ij:03d}.png")
      pyplot.imshow(blur_0[0, i*8+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("blur_0")
  # pyplot.show()

  _, sharp_0 = activation["kernel_0"]

  ij = 1
  for i in range(4):
    for j in range(8):
      ax = pyplot.subplot(4, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(sharp_0[0, i*8+j, :, :], f"./activations/sharp_0_{ij:03d}.png")
      pyplot.imshow(sharp_0[0, i*8+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("sharp_0")
  # pyplot.show()

  blur_1 = activation["encode_1"]

  ij = 1
  for i in range(8):
    for j in range(8):
      ax = pyplot.subplot(8, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(blur_1[0, i*8+j, :, :], f"./activations/blur_1_{ij:03d}.png")
      pyplot.imshow(blur_1[0, i*8+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("blur_1")
  # pyplot.show()

  kernel_1 = activation["downsample_00"]
  _, sharp_1 = activation["kernel_1"]
  ij = 1
  for i in range(8):
    for j in range(8):
      ax = pyplot.subplot(16, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(kernel_1[0, i*8+j, :, :], f"./activations/kernel_1_{ij:03d}.png")
      pyplot.imshow(kernel_1[0, i*8+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  for i in range(8):
    for j in range(8):
      ax = pyplot.subplot(16, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(fft.fftshift(psf2otf(kernel_1[0, i*8+j, :, :]).abs()), f"./activations/kernel_fft_1_{(ij-64):03d}.png")
      pyplot.imshow(fft.fftshift(psf2otf(kernel_1[0, i*8+j, :, :]).abs()).detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("kern_1")
  # pyplot.show()

  ij = 1
  for i in range(8):
    for j in range(8):
      ax = pyplot.subplot(8, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(sharp_1[0, i*8+j, :, :], f"./activations/sharp_1_{ij:03d}.png")
      pyplot.imshow(sharp_1[0, i*8+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("sharp_1")
  # pyplot.show()

  blur_2 = activation["encode_2"]

  ij = 1
  for i in range(8):
    for j in range(16):
      ax = pyplot.subplot(8, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(blur_2[0, i*16+j, :, :], f"./activations/blur_2_{ij:03d}.png")
      pyplot.imshow(blur_2[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("blur_2")
  # pyplot.show()

  kernel_2 = activation["downsample_10"]
  _, sharp_2 = activation["kernel_2"]

  ij = 1
  for i in range(8):
    for j in range(16):
      ax = pyplot.subplot(16, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(kernel_2[0, i*16+j, :, :], f"./activations/kernel_2_{ij:03d}.png")
      pyplot.imshow(kernel_2[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  for i in range(8):
    for j in range(16):
      ax = pyplot.subplot(16, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(fft.fftshift(psf2otf(kernel_2[0, i*16+j, :, :]).abs()), f"./activations/kernel_fft_2_{(ij-128):03d}.png")
      pyplot.imshow(fft.fftshift(psf2otf(kernel_2[0, i*16+j, :, :]).abs()).detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("kern_2")
  # pyplot.show()

  ij = 1
  for i in range(8):
    for j in range(16):
      ax = pyplot.subplot(8, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(sharp_2[0, i*16+j, :, :], f"./activations/sharp_2_{ij:03d}.png")
      pyplot.imshow(sharp_2[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("sharp_2")
  # pyplot.show()

  blur_3 = activation["encode_3"]

  ij = 1
  for i in range(16):
    for j in range(16):
      ax = pyplot.subplot(16, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(blur_3[0, i*16+j, :, :], f"./activations/blur_3_{ij:03d}.png")
      pyplot.imshow(blur_3[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("blur_3")
  # pyplot.show()

  kernel_3 = activation["downsample_20"]
  _, sharp_3 = activation["kernel_3"]

  ij = 1
  for i in range(16):
    for j in range(16):
      ax = pyplot.subplot(32, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(kernel_3[0, i*16+j, :, :], f"./activations/kernel_3_{ij:03d}.png")
      pyplot.imshow(kernel_3[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  for i in range(16):
    for j in range(16):
      ax = pyplot.subplot(32, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(fft.fftshift(psf2otf(kernel_3[0, i*16+j, :, :]).abs()), f"./activations/kernel_fft_3_{(ij-256):03d}.png")
      pyplot.imshow(fft.fftshift(psf2otf(kernel_3[0, i*16+j, :, :]).abs()).detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("kern_3")
  # pyplot.show()

  ij = 1
  for i in range(16):
    for j in range(16):
      ax = pyplot.subplot(16, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(sharp_3[0, i*16+j, :, :], f"./activations/sharp_3_{ij:03d}.png")
      pyplot.imshow(sharp_3[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("sharp_3")
  # pyplot.show()

  sharp_4 = activation["decode_0"]

  ij = 1
  for i in range(16):
    for j in range(16):
      ax = pyplot.subplot(16, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(sharp_4[0, i*16+j, :, :], f"./activations/sharp_4_{ij:03d}.png")
      pyplot.imshow(sharp_4[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("sharp_4")
  # pyplot.show()

  sharp_5 = activation["decode_1"]

  ij = 1
  for i in range(8):
    for j in range(16):
      ax = pyplot.subplot(8, 16, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(sharp_5[0, i*16+j, :, :], f"./activations/sharp_5_{ij:03d}.png")
      pyplot.imshow(sharp_5[0, i*16+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("sharp_5")
  # pyplot.show()

  sharp_6 = activation["decode_2"]

  ij = 1
  for i in range(8):
    for j in range(8):
      ax = pyplot.subplot(8, 8, ij)
      ax.set_xticks([])
      ax.set_yticks([])
      save_image(sharp_6[0, i*8+j, :, :], f"./activations/sharp_6_{ij:03d}.png")
      pyplot.imshow(sharp_6[0, i*8+j, :, :].detach().numpy(), extent=[0, 1, 0, 1])
      ij += 1
  pyplot.title("sharp_6")
  # pyplot.show()
  
  output_proj = activation["output_proj"]
  
  ij = 1
  for i in range(3):
    ax = pyplot.subplot(1, 3, ij)
    ax.set_xticks([])
    ax.set_yticks([])
    save_image(output_proj[0, i, :, :], f"./activations/output_proj_{ij:03d}.png")
    pyplot.imshow(output_proj[0, i, :, :].detach().numpy(), extent=[0, 1, 0, 1])
    ij += 1
  pyplot.title("output_proj")
  # pyplot.show()