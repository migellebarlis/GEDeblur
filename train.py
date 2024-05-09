import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from dataset.dataloader import DeblurringDataLoader
from model.convolution_7_32 import Deblur

class Trainer():
  def __init__(self, model : nn.Module, data_loader, optimiser, scheduler, train_loss, valid_loss, best_loss):
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.model = model.to(self.device)

    self.data_loader = data_loader
    self.optimiser = optimiser
    self.scheduler = scheduler
    self.train_loss = train_loss
    self.valid_loss = valid_loss

    self.best_loss = best_loss

  def train_epoch(self, epoch):
    print('\nEpoch %d' % (epoch+1))
    self.model.train()

    mse = 0
    mse_blur = 0
    epoch_loss = 0
    progress_bar = tqdm(self.data_loader.train_dl, desc='Training  ')

    for batch_idx, (inputs, targets) in enumerate(progress_bar):
      inputs, targets = inputs.to(self.device), targets.to(self.device)

      self.optimiser.zero_grad()

      outputs = self.model(inputs)

      loss = self.train_loss(outputs, targets)
      loss.backward()

      self.optimiser.step()

      if torch.cuda.is_available():
        torch.cuda.synchronize()

      epoch_loss += loss.item()

      mse_blur += F.mse_loss(inputs.clamp(min=0, max=1), targets).item()
      mse += F.mse_loss(outputs.clamp(min=0, max=1), targets).item()
      psnr_blur = 10*math.log10(1/(mse_blur/(batch_idx+1)))
      psnr = 10*math.log10(1/(mse/(batch_idx+1)))
      progress_bar.set_postfix_str('loss: %.10f, psnr: %.10f, isnr: %.10f' % (epoch_loss, psnr, psnr - psnr_blur))

    self.scheduler.step()

    return epoch_loss

  def valid_epoch(self, epoch, state_path):
    mse = 0
    mse_blur = 0
    valid_loss = 0
    self.model.eval()
    with torch.no_grad():
      progress_bar = tqdm(self.data_loader.valid_dl, desc='Validation')
      for batch_idx, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        outputs = self.model(inputs)
        loss = self.valid_loss(outputs, targets)
        valid_loss += loss.item()

        mse_blur += F.mse_loss(inputs.clamp(min=0, max=1), targets).item()
        mse += F.mse_loss(outputs.clamp(min=0, max=1), targets).item()
        psnr_blur = 10*math.log10(1/(mse_blur/(batch_idx+1)))
        psnr = 10*math.log10(1/(mse/(batch_idx+1)))
        progress_bar.set_postfix_str('loss: %.10f, psnr: %.10f, isnr: %.10f' % (valid_loss, psnr, psnr - psnr_blur))

    if not os.path.isdir(os.path.dirname(state_path)):
      os.mkdir(os.path.dirname(state_path))

    if valid_loss < self.best_loss:
      self.best_loss = valid_loss
      state = {
        'model_state' : self.model.state_dict(),
        'optim_state' : self.optimiser.state_dict(),
        'sched_state' : self.scheduler.state_dict(),
        'accuracy': self.best_loss,
        'epoch': epoch,
      }
      torch.save(state, state_path)

    return valid_loss

  def train(self, num_epochs=3000, start_epoch=0, state_path=None):
    self.optimiser.zero_grad()

    for epoch in range(start_epoch, num_epochs):
      self.train_epoch(epoch)
      self.valid_epoch(epoch, state_path)

if __name__ == "__main__":
  torch.manual_seed(1234)
  torch.autograd.set_detect_anomaly(True)

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  state_path = './checkpoint/convolution_7_32.pth'
  total_epochs = 2000

  print('===== Configuring dataset =====')
  dls = DeblurringDataLoader(
      set_name='gopro',
      batch_size=20,
      image_size=256,
      num_workers=2,
  )

  print('===== Building model =====')
  net = Deblur(in_channels=3, embed_channels=32)
  net = net.to(device)

  print('===== Configuring optimiser =====')
  optimiser = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_epochs, eta_min=1e-6)

  best_loss = float("inf")
  start_epoch = 0
  if os.path.exists(state_path):
    print('===== Loading state =====')
    state = torch.load(state_path, map_location=torch.device(device))
    net.load_state_dict(state["model_state"])
    optimiser.load_state_dict(state["optim_state"])
    scheduler.load_state_dict(state["sched_state"])
    best_loss = state["accuracy"]
    start_epoch = state["epoch"]

  print('===== Creating trainer =====')
  train_loss = nn.MSELoss()
  valid_loss = nn.MSELoss()
  t = Trainer(net, dls, optimiser, scheduler, train_loss, valid_loss, best_loss)

  print('===== Start trainer =====')
  t.train(total_epochs, start_epoch, state_path)