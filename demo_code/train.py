from DngDataloader import DngDataLoader,ValDngDataLoader
from torch.utils.data import DataLoader
from models.unetTorch import Unet
import random
import numpy as np
import torch
import losses
import time
from tqdm import tqdm
import utils

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
import torch.nn as nn
#from models.net_torch import Network
#from models.restormer import Denoiser
#from models.NBNet import NBNet
from models.Uformer import Uformer
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
#new_lr = 1e-6
new_lr = 5e-7


#########model##############
#model = Unet()
#model = Network()
#model = Denoiser()
#model = Denoiser()
#model.load_state_dict(torch.load('save_model/th_model.pth'))
model = torch.load('save_model/model1.pth')


#torch.save(model, 'save_model/model.pth')
if torch.cuda.is_available():
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=new_lr, betas=(0.9, 0.999))

######### Scheduler ###########
warmup_epochs = 3
NUM_EPOCHS = 1000
LR_MIN = 1e-8

#scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS-warmup_epochs+40, eta_min=LR_MIN)
#scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
#scheduler.step()

######### Loss ###########
#criterion = losses.CharbonnierLoss()
criterion = nn.L1Loss(reduction='mean')
#criterion = losses.EdgeLoss()
h_grad = losses.Get_gradient_sobel().cuda()
#criterion=  nn.MSELoss()

######### DataLoaders ###########
BATCH_SIZE = 2
train_noise_dir = '../data/dataset/noisy/'   #your train datanoise
train_gt_dir = '../data/dataset/ground truth/'

val_noise_dir = '../data/valset/noisy/'
val_gt_dir = '../data/valset/ground_truth/'

train_dataset = DngDataLoader(train_noise_dir,train_gt_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=10, drop_last=False, pin_memory=True)

val_dataset = ValDngDataLoader(val_noise_dir,val_gt_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=5, drop_last=False, pin_memory=True)

best_psnr = 0
best_epoch = 0
best_iter = 0
best_score = 0
best_ssim = 0

eval_now = len(train_loader)//3 - 1
print(f"\nEval after every {eval_now} Iterations !!!\n")

for epoch in range(NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    model.train()

    for batch, (input, target, _) in enumerate(tqdm(train_loader)):
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        restored = model(input)
        restored_grad = h_grad(restored)
        target_grad = h_grad(target)
        
        loss1 = criterion(restored,target)
        loss2 =  criterion(restored_grad,target_grad)
        loss = 0.8*loss1+0.2*loss2
        #loss = abs(restored - target).mean()
        loss.backward()
        optimizer.step()
        epoch_loss +=loss.item()
    print("-----------------------------val-------------------------------------")

    model.eval()
    psnr_val_rgb = []
    ssim_val_rgb = []
    
    for batch, (input, target, _) in enumerate(tqdm(val_loader)):
        
        with torch.no_grad():
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            restored = model(input)

        
        temp_psnr,temp_ssim = utils.cal_psnr_ssim(target,restored)
        psnr_val_rgb.append(temp_psnr)
        ssim_val_rgb.append(temp_ssim)

    psnr_val_rgb  = np.array(psnr_val_rgb).mean().item()
    ssim_val_rgb  = np.array(ssim_val_rgb).mean().item()
    [w, psnr_max, psnr_min, ssim_min] =  [0.8, 60, 30, 0.8]
    psnr = psnr_val_rgb
    ssim = ssim_val_rgb
    Score = (w * max(psnr - psnr_min, 0) / (psnr_max - psnr_min) + (1 - w) * max(ssim - ssim_min, 0) / (1 - ssim_min)) * 100

    if Score >best_score:
        best_psnr = psnr_val_rgb
        best_epoch = epoch
        best_iter = batch
        best_ssim = ssim
        best_score = Score
        torch.save(model, 'save_model/model.pth')
    
    print("[epoch %d  PSNR: %.4f SSIM: %.4f SCORE: %.4f --- best_epoch %d  Best_PSNR %.4f Bese_SSIM %.4F Score %.4f]" % (epoch, psnr, ssim, Score, best_epoch, best_psnr,best_ssim, best_score))
    
    #if (epoch+1) %100 ==0:
    #   new_lr = new_lr/2
    
     #  for param_group in optimizer.param_groups:
     #      param_group['lr'] = new_lr

    model.train()

    #scheduler.step()
    
    print("------------------------------------------------------------------")
    #print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.10f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\t LR: {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss,new_lr))
    print("------------------------------------------------------------------")

    torch.save(model, 'save_model/model_last.pth')

