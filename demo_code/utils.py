import torch
import numpy as np
import cv2
import skimage.metrics

def inv_normalization(input_data, black_level=1024,white_level = 16383):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data

def write_image(input_data):
    _,height,width,_ = input_data.shape
    width = 2*width
    height = 2*height
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data

def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def numpyPSNR(tar_img, prd_img):
    imdff = np.float32(prd_img) - np.float32(tar_img)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(255/rmse)
    return ps

def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def cal_psnr_ssim(tar_img,prd_img):
    
    tar_img = tar_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
    tar_img = inv_normalization(tar_img)
    tar_img = write_image(tar_img)
    
    prd_img = prd_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
    prd_img = inv_normalization(prd_img)
    prd_img = write_image(prd_img)
    

    psnr = skimage.metrics.peak_signal_noise_ratio(
        tar_img.astype(np.float), prd_img.astype(np.float), data_range=16383)
    ssim = skimage.metrics.structural_similarity(
        tar_img.astype(np.float), prd_img.astype(np.float), multichannel=True, data_range=16383)
    
    return psnr,ssim
