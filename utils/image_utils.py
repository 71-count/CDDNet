import torch
import numpy as np
import pickle
import cv2

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_jpg_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

def load_img_train(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = cv2.resize(img,(512,512))
    img = img/255.
    return img


def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


#def myPSNR(tar_img, prd_img):
#    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
#    rmse = (imdff**2).mean().sqrt()
#    ps = 20*torch.log10(1/rmse)
#    return ps

def myPSNR(tar_img, prd_img):
    imdiff = torch.clamp(prd_img * 255, 0, 255) - torch.clamp(tar_img * 255 , 0 , 255)
    rmse = (imdiff**2).mean().sqrt()
    ps = 20*torch.log10(255/rmse)
    return ps

def batch_PSNR(img1, img2, data_range=None):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR)
