import os,sys,math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'..'))

from model.convolution_7_32 import *

from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

import cv2

from dataset.gopro import GoProDataset

dir_dataset = 'G:/GOPRO_Large'
dir_result = './result/convolution_7_32/GOPRO_Large'

test_dataset = GoProDataset(path=dir_dataset,train=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

state = torch.load("./checkpoint/convolution_7_32.pth", map_location=torch.device('cpu'))

model_restoration = Deblur(
    in_channels=3,
    embed_channels=32)
model_restoration.load_state_dict(state["model_state"])
model_restoration.eval()

def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask

with torch.no_grad():

    psnr_val_rgb = []
    isnr_val_rgb = []
    ssim_val_rgb = []
    progress_bar = tqdm(test_loader)

    for ii, data_test in enumerate(progress_bar, 0):
        rgb_gt = data_test[1].numpy().squeeze().transpose((1,2,0))
        rgb_noisy_ = data_test[0].numpy().squeeze().transpose((1,2,0))
        rgb_noisy, mask = expand2square(data_test[0], factor=128)
        path_gt = data_test[3]
        path_noisy = data_test[2]

        rgb_restored = model_restoration(rgb_noisy)
        rgb_restored = torch.masked_select(rgb_restored,mask.bool()).reshape(1,3,rgb_gt.shape[0],rgb_gt.shape[1])
        rgb_restored = torch.clamp(rgb_restored,0,1).numpy().squeeze().transpose((1,2,0))

        psnr = psnr_loss(rgb_restored, rgb_gt)
        isnr = psnr - psnr_loss(rgb_noisy_, rgb_gt)
        ssim = ssim_loss(rgb_restored, rgb_gt, data_range=1., channel_axis=-1)
        psnr_val_rgb.append(psnr)
        isnr_val_rgb.append(isnr)
        ssim_val_rgb.append(ssim)

        progress_bar.set_postfix_str('PSNR: %.4f, ISNR:%.4f, SSIM: %.4f'% (psnr, isnr, ssim))

        path_restored = os.path.join(dir_result, path_noisy[0])
        dir_restored = os.path.split(path_restored)[0]

        if not os.path.exists(dir_restored):
            os.makedirs(dir_restored)

        cv2.imwrite(path_restored, cv2.cvtColor(img_as_ubyte(rgb_restored), cv2.COLOR_RGB2BGR))

        with open(os.path.join(dir_result,'psnr_isnr_ssim.txt'),'a') as f:
            f.write(path_noisy[0]+' ----> '+"PSNR: %.4f, ISNR: %.4f, SSIM: %.4f"% (psnr, isnr, ssim)+'\n')

        with open(os.path.join(dir_result,'psnr_isnr_ssim.csv'),'a') as f:
            f.write('"'+path_noisy[0]+'",%.4f,%.4f,%.4f'% (psnr, isnr, ssim)+'\n')

    psnr_val_rgb = sum(psnr_val_rgb)/len(test_dataset)
    isnr_val_rgb = sum(isnr_val_rgb)/len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb)/len(test_dataset)

    print("PSNR: %f, ISNR:%.4f, SSIM: %f " %(psnr_val_rgb, isnr_val_rgb, ssim_val_rgb))

    with open(os.path.join(dir_result,'psnr_isnr_ssim.txt'),'a') as f:
        f.write("Arch:convolution_7_32 ----> PSNR: %.4f, ISNR: %.4f, SSIM: %.4f"% (psnr_val_rgb, isnr_val_rgb, ssim_val_rgb)+'\n')

    with open(os.path.join(dir_result,'psnr_isnr_ssim.csv'),'a') as f:
        f.write('"Arch:convolution_7_32",%.4f,%.4f,%.4f'% (psnr_val_rgb, isnr_val_rgb, ssim_val_rgb)+'\n')