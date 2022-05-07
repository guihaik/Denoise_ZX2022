import os
import glob
import numpy as np
import rawpy
import torch
import random
import torchvision.transforms.functional as TF


class DngDataLoader:
    def __init__(self,noise_path,gt_path,black_level=1024,white_level = 16383,TRAIN_PS = 256):
        self.image_noise = []
        self.image_gt = []
        self.noise_path = noise_path
        self.gt_path = gt_path
        self.path2name()
        self.black_level = black_level
        self.white_level = white_level
        self.ps = TRAIN_PS
        self.batch_size = 1
        self._set_dataset_length()
        

    
    def path2name(self):
        names_noise = sorted(glob.glob(os.path.join(self.noise_path, '*' )))
        name_gt = sorted(glob.glob(os.path.join(self.gt_path, '*' )))
        self.image_noise = names_noise
        self.image_gt = name_gt
    
    def _get_index(self, idx):

        if idx < self.random_border:
            return idx % len(self.image_gt)
        else:
            return np.random.randint(len(self.image_gt))
        
    
    def _set_dataset_length(self):
        
        self.dataset_length = 450 * self.batch_size
        repeat = self.dataset_length // len(self.image_gt)
        self.random_border = len(self.image_gt) * repeat
            


    
    def __len__(self):
        return self.dataset_length
    
    
    def __getitem__(self, idx):
        noise, gt, filename = self._load_file(idx)
        return noise, gt, filename
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        
        ps = self.ps
        f_noise = self.image_noise[idx]
        f_gt = self.image_gt[idx]

        filename, _ = os.path.splitext(os.path.basename(f_gt))
        filename1, _ = os.path.splitext(os.path.basename(f_noise))
        

        noise_raw_data_expand_c, noise_height, noise_width = self.read_image(f_noise)
        gt_raw_data_expand_c, gt_height, gt_width = self.read_image(f_gt)

        noise_raw_data_expand_c_normal = self.normalization(noise_raw_data_expand_c)
        gt_raw_data_expand_c_normal = self.normalization(gt_raw_data_expand_c)

        

        w,h,_ = gt_raw_data_expand_c_normal.shape
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        
        
        

        noise_raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        noise_raw_data_expand_c_normal.reshape(noise_height//2, noise_width//2, 4), (2,0,1))).float()

        gt_raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        gt_raw_data_expand_c_normal.reshape(noise_height//2, noise_width//2, 4), (2,0,1))).float()
        
        if padw!=0 or padh!=0:
            noise_raw_data_expand_c_normal = TF.pad(noise_raw_data_expand_c_normal, (0,0,padw,padh), padding_mode='reflect')
            gt_raw_data_expand_c_normal = TF.pad(gt_raw_data_expand_c_normal, (0,0,padw,padh), padding_mode='reflect')

        hh, ww = gt_raw_data_expand_c_normal.shape[1], gt_raw_data_expand_c_normal.shape[2]
    

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        noise_raw_data_expand_c_normal = noise_raw_data_expand_c_normal[:, rr:rr+ps, cc:cc+ps]
        gt_raw_data_expand_c_normal = gt_raw_data_expand_c_normal[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            noise_raw_data_expand_c_normal = noise_raw_data_expand_c_normal.flip(1)
            gt_raw_data_expand_c_normal = gt_raw_data_expand_c_normal.flip(1)
        elif aug==2:
            noise_raw_data_expand_c_normal = noise_raw_data_expand_c_normal.flip(2)
            gt_raw_data_expand_c_normal = gt_raw_data_expand_c_normal.flip(2)
        elif aug==3:
            noise_raw_data_expand_c_normal = torch.rot90(noise_raw_data_expand_c_normal,dims=(1,2))
            gt_raw_data_expand_c_normal = torch.rot90(gt_raw_data_expand_c_normal,dims=(1,2))
        elif aug==4:
            noise_raw_data_expand_c_normal = torch.rot90(noise_raw_data_expand_c_normal,dims=(1,2), k=2)
            gt_raw_data_expand_c_normal = torch.rot90(gt_raw_data_expand_c_normal,dims=(1,2), k=2)
        elif aug==5:
            noise_raw_data_expand_c_normal = torch.rot90(noise_raw_data_expand_c_normal,dims=(1,2), k=3)
            gt_raw_data_expand_c_normal = torch.rot90(gt_raw_data_expand_c_normal,dims=(1,2), k=3)
        elif aug==6:
            noise_raw_data_expand_c_normal = torch.rot90(noise_raw_data_expand_c_normal.flip(1),dims=(1,2))
            gt_raw_data_expand_c_normal = torch.rot90(gt_raw_data_expand_c_normal.flip(1),dims=(1,2))
        elif aug==7:
            noise_raw_data_expand_c_normal = torch.rot90(noise_raw_data_expand_c_normal.flip(2),dims=(1,2))
            gt_raw_data_expand_c_normal = torch.rot90(gt_raw_data_expand_c_normal.flip(2),dims=(1,2))

        return noise_raw_data_expand_c_normal, gt_raw_data_expand_c_normal, filename
    
    def read_image(self,input_path):
        raw = rawpy.imread(input_path)
        raw_data = raw.raw_image_visible
        height = raw_data.shape[0]
        width = raw_data.shape[1]

        raw_data_expand = np.expand_dims(raw_data, axis=2)
        raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                            raw_data_expand[0:height:2, 1:width:2, :],
                                            raw_data_expand[1:height:2, 0:width:2, :],
                                            raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
        return raw_data_expand_c, height, width

    def normalization(self,input_data):
        output_data = (input_data.astype(float) - self.black_level) / (self.white_level - self.black_level)
        return output_data

    def inv_normalization(self,input_data):
        output_data = np.clip(input_data, 0., 1.) * (self.white_level - self.black_level) + self.black_level
        output_data = output_data.astype(np.uint16)
        return output_data

class ValDngDataLoader:
    def __init__(self,noise_path,gt_path,black_level=1024,white_level = 16383,TRAIN_PS = 512):
        self.image_noise = []
        self.image_gt = []
        self.noise_path = noise_path
        self.gt_path = gt_path
        self.path2name()
        self.black_level = black_level
        self.white_level = white_level
        self.ps = TRAIN_PS

    def __len__(self):
        return len(self.image_noise)
    
    def path2name(self):
        names_noise = sorted(glob.glob(os.path.join(self.noise_path, '*' )))
        name_gt = sorted(glob.glob(os.path.join(self.gt_path, '*' )))
        self.image_noise = names_noise
        self.image_gt = name_gt
    
    def __getitem__(self, idx):
        noise, gt, filename = self._load_file(idx)
        return noise, gt, filename
    
    def _load_file(self, idx):
        
        ps = self.ps
        f_noise = self.image_noise[idx]
        f_gt = self.image_gt[idx]

        filename, _ = os.path.splitext(os.path.basename(f_gt))
        filename1, _ = os.path.splitext(os.path.basename(f_noise))

        noise_raw_data_expand_c, noise_height, noise_width = self.read_image(f_noise)
        gt_raw_data_expand_c, gt_height, gt_width = self.read_image(f_gt)

        noise_raw_data_expand_c_normal = self.normalization(noise_raw_data_expand_c)
        gt_raw_data_expand_c_normal = self.normalization(gt_raw_data_expand_c)

        




        noise_raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        noise_raw_data_expand_c_normal.reshape(noise_height//2, noise_width//2, 4), (2,0,1))).float()

        gt_raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        gt_raw_data_expand_c_normal.reshape(noise_height//2, noise_width//2, 4), (2,0,1))).float()

        if self.ps is not None:
            noise_raw_data_expand_c_normal = TF.center_crop(noise_raw_data_expand_c_normal, (ps,ps))
            gt_raw_data_expand_c_normal = TF.center_crop(gt_raw_data_expand_c_normal, (ps,ps))

        return noise_raw_data_expand_c_normal, gt_raw_data_expand_c_normal, filename
    
    def read_image(self,input_path):
        raw = rawpy.imread(input_path)
        raw_data = raw.raw_image_visible
        height = raw_data.shape[0]
        width = raw_data.shape[1]

        raw_data_expand = np.expand_dims(raw_data, axis=2)
        raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                            raw_data_expand[0:height:2, 1:width:2, :],
                                            raw_data_expand[1:height:2, 0:width:2, :],
                                            raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
        return raw_data_expand_c, height, width

    def normalization(self,input_data):
        output_data = (input_data.astype(float) - self.black_level) / (self.white_level - self.black_level)
        return output_data

    def inv_normalization(self,input_data):
        output_data = np.clip(input_data, 0., 1.) * (self.white_level - self.black_level) + self.black_level
        output_data = output_data.astype(np.uint16)
        return output_data



if __name__ =='__main__':
    from torch.utils.data import DataLoader
    trainset = DngDataLoader('../data/dataset/noisy/','../data/dataset/ground truth/')
    loader_train = DataLoader(
                trainset,
                batch_size=1,
                num_workers=0,
                shuffle=True,
            )
    for batch, (lr, hr, _) in enumerate(loader_train):
        print(lr.size())
        print(hr.size())
        break
