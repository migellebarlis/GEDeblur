import torch
import torch.fft as fft
import torch.nn.functional as F

def dctmtx(n, device):
  cc, rr = torch.meshgrid(torch.arange(1, n+1, device=device), torch.arange(1, n+1, device=device))

  c = ((2/n)**0.5) * torch.cos(torch.pi * (2 * cc + 1) * rr / (2 * n))
  c[0, :] = c[0, :] / (2**0.5)

  return c

def dct2(input, s=None):
  device = input.device

  n = input.shape[-2] if s == None else s[0]
  m = dctmtx(n, device)

  return m @ input @ torch.transpose(m, -2, -1)

def idct2(input, s=None):
  device = input.device

  n = input.shape[-2] if s == None else s[0]
  m = torch.transpose(dctmtx(n, device), -2, -1)

  return m @ input @ torch.transpose(m, -2, -1)

def psf2otf(psf, shape=None):
  h, w = psf.shape[-2], psf.shape[-1]

  if shape is not None:
    psf = F.pad(psf, (0, shape[-1]-w, 0, shape[-2]-h))

  if torch.all(psf == 0):
    return torch.zeros(psf.shape)

  psf = torch.roll(psf, shifts=-h//2, dims=-2)
  psf = torch.roll(psf, shifts=-w//2, dims=-1)

  otf = fft.fft2(psf)

  n_ops = torch.sum(torch.prod(torch.tensor(psf.shape)) * torch.log2(torch.tensor(psf.shape)))

  if torch.max(torch.abs(torch.imag(otf)))/torch.max(torch.abs(otf)) <= n_ops*torch.finfo(torch.float32).eps:
    otf = torch.real(otf)

  return otf

def otf2psf(otf, shape=None):
  h, w = otf.shape[-2], otf.shape[-1]

  if torch.all(otf == 0):
    if shape is not None:
      return torch.zeros(*otf.shape[0:-2], shape[-2], shape[-1])
    else:
      return torch.zeros(otf.shape)
  
  psf = fft.ifft2(otf)

  n_ops = torch.sum(torch.prod(torch.tensor(otf.shape)) * torch.log2(torch.tensor(otf.shape)))

  if torch.max(torch.abs(torch.imag(psf)))/torch.max(torch.abs(psf)) <= n_ops*torch.finfo(torch.float32).eps:
    psf = torch.real(psf)

  psf = torch.roll(psf, shifts=h//2, dims=-2)
  psf = torch.roll(psf, shifts=w//2, dims=-1)

  if shape is not None:
    psf = psf[..., 0:shape[-2], 0:shape[-1]]

  return psf