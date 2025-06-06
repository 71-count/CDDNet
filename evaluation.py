import os
import glob
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import torch
import lpips
import argparse

def calculate_psnr(img1, img2):
    """计算PSNR值"""
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """计算SSIM值"""
    # 如果是彩色图像，需要在每个通道上计算SSIM然后取平均值
    if len(img1.shape) == 3:
        ssim_value = 0
        for i in range(img1.shape[2]):
            ssim_value += ssim(img1[:,:,i], img2[:,:,i], data_range=255)
        return ssim_value / img1.shape[2]
    else:
        return ssim(img1, img2, data_range=255)

def calculate_rmse(img1, img2):
    """计算RMSE值"""
    return np.sqrt(mse(img1, img2))

def calculate_lpips(img1, img2, loss_fn):
    """计算LPIPS值"""
    # 转换为PyTorch张量
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    # 确保尺寸为[0,1]
    img1 = 2 * img1 - 1
    img2 = 2 * img2 - 1
    
    # 计算LPIPS
    with torch.no_grad():
        lpips_value = loss_fn(img1, img2).item()
    
    return lpips_value

def main():
    parser = argparse.ArgumentParser(description='计算图像质量评价指标')
    parser.add_argument('--gt_dir', type=str, default=r'D:\Desktop\hqy\dataset\test2\GT', help='GT图像目录')
    parser.add_argument('--di_dir', type=str, default=r'D:\Desktop\hqy\result\test2\img\CDDNet_doubletask', help='DI图像目录')
    parser.add_argument('--pattern', type=str, default='*.jpg', help='图像文件匹配模式，默认为*.png')
    args = parser.parse_args()
    
    # 初始化LPIPS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # 获取所有GT图像
    gt_images = sorted(glob.glob(os.path.join(args.gt_dir, args.pattern)))
    di_images = sorted(glob.glob(os.path.join(args.di_dir, args.pattern)))
    
    if len(gt_images) == 0 or len(di_images) == 0:
        print(f"未找到匹配的图像文件。GT: {len(gt_images)}张, DI: {len(di_images)}张")
        return
    
    if len(gt_images) != len(di_images):
        print(f"警告: GT和DI图像数量不匹配。GT: {len(gt_images)}张, DI: {len(di_images)}张")
    
    # 存储所有指标
    psnr_values = []
    ssim_values = []
    lpips_values = []
    rmse_values = []
    
    # 遍历所有图像对
    for i, (gt_path, di_path) in enumerate(zip(gt_images, di_images)):
        gt_name = os.path.basename(gt_path)
        di_name = os.path.basename(di_path)
        
        # 读取图像
        gt_img = cv2.imread(gt_path)
        di_img = cv2.imread(di_path)
        
        # 确保图像尺寸一致
        if gt_img.shape != di_img.shape:
            print(f"警告: 图像尺寸不匹配 - {gt_name}: {gt_img.shape}, {di_name}: {di_img.shape}")
            di_img = cv2.resize(di_img, (gt_img.shape[1], gt_img.shape[0]))
        
        # 计算指标
        psnr_value = calculate_psnr(gt_img, di_img)
        ssim_value = calculate_ssim(gt_img, di_img)
        lpips_value = calculate_lpips(gt_img, di_img, loss_fn)
        rmse_value = calculate_rmse(gt_img, di_img)
        
        # 存储指标
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        lpips_values.append(lpips_value)
        rmse_values.append(rmse_value)
        
        # 打印单个图像对的指标
        print(f"img {i+1}/{len(gt_images)} - {gt_name} vs {di_name}:")
        print(f"  PSNR: {psnr_value:.4f} dB")
        print(f"  SSIM: {ssim_value:.4f}")
        print(f"  LPIPS: {lpips_value:.4f}")
        print(f"  RMSE: {rmse_value:.4f}")
        print("-" * 50)
    
    # 计算平均值
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)
    avg_rmse = np.mean(rmse_values)
    
    # 打印平均指标
    print("\n" + "=" * 50)
    print(f"avg_metrics (total {len(psnr_values)} pair_images):")
    print(f"  avg PSNR: {avg_psnr:.4f} dB")
    print(f"  avg SSIM: {avg_ssim:.4f}")
    print(f"  avg LPIPS: {avg_lpips:.4f}")
    print(f"  avg RMSE: {avg_rmse:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()