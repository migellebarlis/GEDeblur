import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

def dctmtx(n):
    with torch.no_grad():
        cc, rr = torch.meshgrid(torch.arange(1, n+1), torch.arange(1, n+1))

        c = ((2/n)**0.5) * torch.cos(torch.pi * (2 * cc + 1) * rr / (2 * n))
        c[0, :] = c[0, :] / (2**0.5)

        return c
    
def init_gauss_mask(head, size):
    with torch.no_grad():
        gauss = torch.unsqueeze(torch.signal.windows.gaussian(5*size, std=10), dim=0)
        gauss = torch.t(gauss)@gauss
        mask = torch.zeros(1, 1, size*size, 1, head)
        factor1 = head
        factor2 = 1
        for i in range(math.floor(head**0.5), 0, -1):
            if (head%i == 0):
                factor1 = i
                factor2 = math.floor(head/i)
                break

        step1 = math.floor(size/factor1)
        offset1 = math.floor(step1/2)
        step2 = math.floor(size/factor2)
        offset2 = math.floor(step2/2)
        k = 0
        for i in range(factor1):
            for j in range(factor2):
                temp = torch.zeros(size, size)
                temp[0+(i*step1)+offset1,0+(j*step2)+offset2] = 1
                conv = F.conv2d(temp.unsqueeze(0).unsqueeze(0), gauss.unsqueeze(0).unsqueeze(0), padding="same").squeeze()
                conv = conv / torch.sum(conv)
                conv = conv.reshape(-1)
                mask[0, 0, :, 0, k] = conv
        return mask

class BlurSimilarity(nn.Module):
    def __init__(self, pat_size, in_channels, num_feats, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Linear layer
        # in_features=patch dimension
        # out_features=number of features * number of heads
        self.df = num_feats**0.5

        self.num_feats = num_feats
        self.num_heads = num_heads
        self.linear = nn.Linear(in_features=in_channels*pat_size*pat_size, out_features=num_feats*num_heads)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        # Wb*x
        x = self.linear(x)

        # Softmax(BB'/sqrt(dB))
        x = rearrange(x, 'b n (h cf) -> b (n h) (cf)', h=self.num_heads)
        xt = torch.transpose(x, -1, -2)
        x = self.softmax((x@xt)/self.df)
        
        return x
    
class SharpBlurFeature(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_heads, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels*num_heads, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels*num_heads),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels*num_heads, out_channels=mid_channels*num_heads, kernel_size=1, groups=num_heads, padding=0),
            nn.BatchNorm2d(mid_channels*num_heads),
            nn.GELU(),
            nn.Conv2d(in_channels=mid_channels*num_heads, out_channels=out_channels*num_heads, kernel_size=1, groups=num_heads, padding=0),
            nn.BatchNorm2d(out_channels*num_heads)
        )

    def forward(self, x):

        x = self.block(x)

        return x
    
class KernelEstimate(nn.Module):
    def __init__(self, pat_size=16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.eps = 1e-3
        self.beta = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU()
        self.transform = nn.Parameter(rearrange(dctmtx(pat_size), 'h w -> 1 1 h w'))
        self.itransform = nn.Parameter(rearrange(dctmtx(pat_size).permute((1, 0)), 'h w -> 1 1 h w'))

    def forward(self, blur, sharp):

        fblur = self.transform @ blur @ self.transform.permute((0, 1, 3, 2))
        fsharp = self.transform @ sharp @ self.transform.permute((0, 1, 3, 2))

        fkern = torch.sum(fsharp.conj() * fblur, dim=1) / (torch.sum(fsharp.abs().square(), dim=1) + self.eps)
        kern = self.itransform @ fkern @ self.itransform.permute((0, 1, 3, 2))

        kern = self.relu(kern - self.beta * torch.logsumexp(kern, dim=(-2, -1), keepdim=True))
        kerndiv = torch.sum(kern, dim=(-2, -1), keepdim=True)
        kerndiv[kerndiv == 0] = self.eps
        kern = kern / kerndiv

        return kern
    
class ImageEstimate(nn.Module):
    def __init__(self, pat_size=16, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gamma = 1e-3
        self.pat_size = pat_size
        self.transform3 = nn.Parameter(rearrange(dctmtx(3*pat_size), 'h w -> 1 1 1 1 h w'))
        self.itransform3 = nn.Parameter(rearrange(dctmtx(3*pat_size).permute((1, 0)), 'h w -> 1 1 1 1 h w'))

    def forward(self, blur, kern):

        fkern = self.transform3 @ F.pad(kern, (16, 16, 16, 16), "constant", 0) @ self.transform3.permute((0, 1, 2, 3, 5, 4))
        fblur = self.transform3 @ blur @ self.transform3.permute((0, 1, 2, 3, 5, 4))

        fsharp = (fkern.conj() * fblur)/(fkern.abs().square() + self.gamma)
        sharp = self.itransform3 @ fsharp @ self.itransform3.permute((0, 1, 2, 3, 5, 4))

        sharp = sharp[:, :, :, :, self.pat_size:2*self.pat_size, self.pat_size:2*self.pat_size]

        return sharp

class Xformer(nn.Module):
    def __init__(self, pat_size=16, in_channels=3, in_sharp_blur_feats=16, out_sharp_blur_feats=2, blur_sim_feats=8, num_heads=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.pat_size = pat_size
        self.out_sharp_blur_feats = out_sharp_blur_feats
        self.mask = nn.Parameter(init_gauss_mask(num_heads, pat_size))
        self.blurSim = BlurSimilarity(pat_size=pat_size, in_channels=in_channels, num_feats=blur_sim_feats, num_heads=num_heads)
        self.sharpBlurFeat = SharpBlurFeature(in_channels=in_channels, mid_channels=in_sharp_blur_feats, out_channels=out_sharp_blur_feats, num_heads=num_heads)
        self.kernelEstimate = KernelEstimate(pat_size=pat_size)
        self.imageEstimate = ImageEstimate(pat_size=pat_size)

    def forward(self, x):
        # Size
        b, c, h, w = x.size()
        py = h//self.pat_size # py = h/ph
        px = w//self.pat_size # px = w/pw

        # Clamp the mask and window values
        mask = self.mask.clamp(0, 1)

        # Unfold input image
        x_uf = F.pad(x, pad=(self.pat_size, self.pat_size, self.pat_size, self.pat_size), mode='replicate')
        x_uf = x_uf.unfold(dimension=-2, size=3*self.pat_size, step=self.pat_size)
        x_uf = x_uf.unfold(dimension=-2, size=3*self.pat_size, step=self.pat_size)
        x_ctr = x_uf[:, :, :, :, self.pat_size:2*self.pat_size, self.pat_size:2*self.pat_size]
        
        # Flatten
        sim = rearrange(x_ctr, 'b c py px ph pw -> b (py px) (c ph pw)')

        # Blur similarity
        sim = self.blurSim(sim)
        sim = sim.unsqueeze(1)

        # Sharp blur features
        feat = rearrange(x_ctr, 'b c py px ph pw -> (b py px) c ph pw')

        feat = self.sharpBlurFeat(feat)

        feat = rearrange(feat, '(b n) (h c) ph pw -> b c (ph pw) n h', n=py*px, c=self.out_sharp_blur_feats)

        # Masked sharp blur features
        masked = rearrange(feat * mask, 'b c f n h -> b c f (n h)')

        blur, sharp = torch.tensor_split(masked, 2, dim=1)

        # Compound sharp blur features
        blur = rearrange(blur@sim, 'b c (ph pw) nh -> b c nh ph pw', ph=self.pat_size, pw=self.pat_size)
        sharp = rearrange(sharp@sim, 'b c (ph pw) nh -> b c nh ph pw', ph=self.pat_size, pw=self.pat_size)

        # Kernel estimates
        kern = self.kernelEstimate(blur, sharp)
        
        # Image estimates
        kern = rearrange(kern, 'b (n h) ph pw -> b 1 n h ph pw', n=py*px)
        x_uf = rearrange(x_uf, 'b c py px ph pw -> b c (py px) 1 ph pw')

        est = self.imageEstimate(x_uf, kern)

        # Unmask image estimates
        mask = rearrange(self.mask, 'b c (ph pw) n h -> b c n h ph pw', ph=self.pat_size, pw=self.pat_size)
        est = torch.sum(est*mask, dim=-3)/torch.sum(mask, dim=-3)
        est = rearrange(est, 'b c (py px) ph pw -> b c (py ph) (px pw)', py=py, px=px)

        return est
    
class XformerN(nn.Module):
    def __init__(self, pat_size=16, in_channels=3, in_sharp_blur_feats=16, out_sharp_blur_feats=2, blur_sim_feats=8, num_heads=4, num_stages=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.stages = nn.ModuleList()

        for l in range(num_stages):
            stage_pat_size = pat_size

            self.stages.append(Xformer(
                pat_size=stage_pat_size,
                in_channels=in_channels,
                in_sharp_blur_feats=in_sharp_blur_feats,
                out_sharp_blur_feats=out_sharp_blur_feats,
                blur_sim_feats=blur_sim_feats,
                num_heads=num_heads))

    def forward(self, x):
        
        for stage in self.stages:
            x = stage(x)

        return x