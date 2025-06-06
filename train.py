import os
import sys
from config import Config 
opt = Config('./training.yml')

print("batch size: ", opt.OPTIM.BATCH_SIZE)

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
print("GPU:: ", gpus)

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import random
import time
import numpy as np

import utils
from dataloaders.data_rgb import get_training_data, get_validation_data

from networks.CDDNet import Denoise
from networks.DepthNet import DN as DepthNet
from networks.UNet import UNet
from utils.losses import ContrastLoss as crloss

from tqdm import tqdm 
from tensorboardX import SummaryWriter

# 导入预训练的深度估计模型

from depth.networks import hrnet18, DepthDecoder_MSF

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

start_epoch = 1
mode = opt.MODEL.MODE
session = opt.MODEL.SESSION

train_dir = opt.TRAINING.TRAIN_DIR_KCL
val_dir = opt.TRAINING.VAL_DIR_KCL

######### 加载预训练的深度估计模型 ###########
with torch.no_grad():
    model_path = os.path.join("./depth/models", 'RA-Depth')#RA=GT  OUR=IN L1LOSS=IN~DI
    assert os.path.isdir(model_path), \
        "Cannot find a folder at {}".format(model_path)
    print("-> Loading depth estimation weights from {}".format(model_path))

    encoder_path = os.path.join(model_path, "encoder.pth")
    decoder_path = os.path.join(model_path, "depth.pth")
    
    encoder_dict = torch.load(encoder_path)
    encoder = hrnet18(False)
    depth_decoder = DepthDecoder_MSF(encoder.num_ch_enc, [0], num_output_channels=1)
    
    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))
    
    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

######### 初始化模型 ###########
deep_estimate_net = UNet().cuda()
# 去噪网络
denoise_net = Denoise()
denoise_net.cuda()

# 深度估计网络（用于估计去噪后图像的深度）
depth_net = DepthNet()
depth_net.cuda()

# 创建模型保存目录
model_dir = os.path.join(opt.TRAINING.SAVE_DIR)
utils.mkdir(model_dir)

# 设置训练模式
denoise_net.train()
depth_net.train()

# 优化器设置
optimizer_denoise = optim.Adam(denoise_net.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)
optimizer_depth = optim.Adam(depth_net.parameters(), lr=opt.OPTIM.LR_INITIAL * 0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-8)

scheduler_denoise = optim.lr_scheduler.CosineAnnealingLR(optimizer_denoise, opt.OPTIM.NUM_EPOCHS, eta_min=1e-6)
scheduler_depth = optim.lr_scheduler.CosineAnnealingLR(optimizer_depth, opt.OPTIM.NUM_EPOCHS, eta_min=1e-6)

scaler = GradScaler()

######### 断点恢复训练 ###########
if opt.TRAINING.RESUME:
    print('===> 尝试从断点恢复训练')
    # 加载去噪网络的检查点
    denoise_path = os.path.join(model_dir, "denoise_best.pth")
    if os.path.exists(denoise_path):
        print(f"找到去噪网络检查点: {denoise_path}")
        checkpoint = torch.load(denoise_path)
        denoise_net.load_state_dict(checkpoint['state_dict'])
        optimizer_denoise.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        
        # 加载深度网络的检查点
        depth_path = os.path.join(model_dir, "depth_best.pth")
        if os.path.exists(depth_path):
            print(f"找到深度网络检查点: {depth_path}")
            depth_checkpoint = torch.load(depth_path)
            depth_net.load_state_dict(depth_checkpoint['state_dict'])
            optimizer_depth.load_state_dict(depth_checkpoint['optimizer'])
        
        # 更新学习率调度器
        for i in range(1, start_epoch):
            scheduler_denoise.step()
            scheduler_depth.step()
        
        print('------------------------------------------------------------------------------')
        print("==> 从第 {} 轮继续训练，当前学习率: {:.6f}(去噪网络), {:.6f}(深度网络)"
              .format(start_epoch, scheduler_denoise.get_last_lr()[0], scheduler_depth.get_last_lr()[0]))
        print('------------------------------------------------------------------------------')
    else:
        print("未找到检查点文件，将从头开始训练")

device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    denoise_net = nn.DataParallel(denoise_net, device_ids=device_ids)
    depth_net = nn.DataParallel(depth_net, device_ids=device_ids)

######### 损失函数 ###########
criterion_denoise = nn.L1Loss().cuda()
criterion_denoise_cr = crloss().cuda()
criterion_depth = nn.L1Loss().cuda()

######### 数据加载器 ###########
train_dataset = get_training_data(train_dir)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=16, drop_last=False)
val_dataset = get_validation_data(val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=False, num_workers=8, drop_last=False)

print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

# 创建TensorBoard日志
# 获取网络名称
network_name = 'CDDNet'  # 从日志目录名称提取
log_dir = os.path.join('runs', network_name)
writer = SummaryWriter(log_dir)

best_psnr = 0
best_epoch = 0
best_iter = 0

eval_now = len(train_loader)//4 - 1
print(f"\nEvaluation after every {eval_now} Iterations !!!\n")

def train_step(source_img, target_img):
    # 去噪网络前向传播
    denoise_output_img = denoise_net(source_img)
    denoise_output_img = torch.clamp(denoise_output_img, 0, 1)###DI
    
    # 使用预训练的深度估计模型获取真实图像的深度
    with torch.no_grad():
        real_img_2_depth_img = depth_decoder(encoder(target_img))###DMGT
        real_img_2_depth_img = real_img_2_depth_img[("disp", 0)]
        # 归一化深度图
        real_img_2_depth_img = (real_img_2_depth_img - real_img_2_depth_img.min()) / (real_img_2_depth_img.max() - real_img_2_depth_img.min() + 1e-7)
    
    # 使用深度网络估计去噪后图像的深度
    denoise_output_img_2_depth_img = depth_net(denoise_output_img)##DMDI
    # 归一化深度图
    denoise_output_img_2_depth_img = (denoise_output_img_2_depth_img - denoise_output_img_2_depth_img.min()) / (denoise_output_img_2_depth_img.max() - denoise_output_img_2_depth_img.min() + 1e-7)
    
    # 计算去噪图像和目标图像之间的差异，添加值范围限制
    diff_denoise = torch.sub(denoise_output_img, target_img)###DI-GT
    diff_denoise = torch.clamp(diff_denoise, -1, 1)  # 限制差异值范围
    
    B, C, H, W = diff_denoise.shape
    diff_denoise = diff_denoise.permute(0, 2, 3, 1)
    diff_denoise = diff_denoise.reshape(-1, C * H * W)
    
    # 添加数值稳定性
    epsilon = 1e-7
    diff_denoise_max = torch.max(diff_denoise, dim=1, keepdim=True)[0]
    diff_denoise_min = torch.min(diff_denoise, dim=1, keepdim=True)[0]
    diff_denoise_norm = (diff_denoise - diff_denoise_min) / (diff_denoise_max - diff_denoise_min + epsilon)
    
    diff_d_w = F.softmax(diff_denoise_norm, dim=-1)
    diff_d_w = torch.clamp(diff_d_w, epsilon, 1.0)  # 确保权重在有效范围内
    diff_d_w = diff_d_w.reshape(B, H, W, C).permute(0, 3, 1, 2)
    diff_denoise_w = torch.sum(diff_d_w, dim=1, keepdim=True)
    
    # 加权深度图
    weighted_depth_output_img = torch.mul(denoise_output_img_2_depth_img, diff_denoise_w)###DMDI
    weighted_real_img_2_depth_img = torch.mul(real_img_2_depth_img, diff_denoise_w)###DMGT
    
    # 深度一致性损失
    loss_depth_consis = criterion_depth(weighted_depth_output_img, weighted_real_img_2_depth_img)###加权深度图LOSS
    loss_depth_consis_w = criterion_depth(denoise_output_img_2_depth_img, real_img_2_depth_img)###深度图LOSS
    loss_total_depth = loss_depth_consis + loss_depth_consis_w###total=l1+l2
    
    # UNet特征提取
    t_d1, t_d2, t_d3 = deep_estimate_net(target_img)
    o_d1, o_d2, o_d3 = deep_estimate_net(source_img)
    
    # 去噪损失
    loss_denoise_consis = criterion_denoise(denoise_output_img, target_img)##DI GT  10
    loss_denoise_consis_w = criterion_denoise(denoise_output_img_2_depth_img, real_img_2_depth_img)###DMDI DMGT 10
    loss_denoise_cr = criterion_denoise_cr(denoise_output_img, target_img, source_img)##DI,GT,IN
    
    # 添加梯度裁剪以防止梯度爆炸
    loss_denoise_u1 = torch.clamp(criterion_denoise(t_d1, o_d1), 0, 10)
    loss_denoise_u2 = torch.clamp(criterion_denoise(t_d2, o_d2), 0, 10)
    loss_denoise_u3 = torch.clamp(criterion_denoise(t_d3, o_d3), 0, 10)
    
    # 总去噪损失，添加损失权重
    # 修改损失计算和NaN处理部分
    loss_denoise_total = loss_denoise_consis + 0.01 * loss_denoise_consis_w + 0.01 * loss_denoise_cr + \
                       0.1 * (loss_denoise_u1 + loss_denoise_u2 + loss_denoise_u3)
    
    # 检查并处理可能的NaN值
    if torch.isnan(loss_denoise_total) or torch.isnan(loss_total_depth):
        print("Warning: NaN detected in loss calculation")
        # 使用requires_grad=True创建张量
        if torch.isnan(loss_denoise_total):
            loss_denoise_total = torch.tensor(1.0, requires_grad=True, device='cuda')
        if torch.isnan(loss_total_depth):
            loss_total_depth = torch.tensor(1.0, requires_grad=True, device='cuda')
    
    return denoise_output_img, loss_denoise_total, loss_total_depth

# 在主训练循环中添加梯度裁剪
for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
    epoch_start_time = time.time()
    denoise_epoch_loss = 0
    depth_epoch_loss = 0
    batch_psnr = []  # 记录训练时的PSNR
    
    denoise_net.train()
    depth_net.train()
    
    for i, data in enumerate(tqdm(train_loader), 0):
        # 清空梯度
        optimizer_denoise.zero_grad()
        optimizer_depth.zero_grad()
        
        # data[0]->gt    data[1]->noise
        target = data[0].cuda()
        input_ = data[1].cuda()
        
        with autocast(enabled=True):
            denoise_output, loss_denoise, loss_depth = train_step(input_, target)
        
        # 反向传播
        scaler.scale(loss_denoise + loss_depth).backward()
        
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(denoise_net.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(depth_net.parameters(), max_norm=1.0)
        
        # 更新参数
        scaler.step(optimizer_denoise)
        scaler.step(optimizer_depth)
        scaler.update()
        
        denoise_epoch_loss += loss_denoise.item()
        depth_epoch_loss += loss_depth.item()
        
        # 计算训练时的PSNR
        with torch.no_grad():
            batch_psnr.append(utils.batch_PSNR(denoise_output, target, 1.))
        
        #### 评估 ####
        if i % eval_now == 0 and i > 0:
            denoise_net.eval()
            with torch.no_grad():
                psnr_val_rgb = []
                for ii, data_val in enumerate(val_loader, 0):
                    target = data_val[0].cuda()
                    input_ = data_val[1].cuda()
                    
                    restored = denoise_net(input_)
                    restored = torch.clamp(restored, 0, 1)
                    psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.))
                
                psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
                
                # 记录验证PSNR
                writer.add_scalar('Val/PSNR', psnr_val_rgb, epoch)
                writer.add_scalar('Val/Best_PSNR', best_psnr, epoch)
                
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch = epoch
                    best_iter = i
                    
                    # 保存去噪网络
                    torch.save({
                        'epoch': epoch,
                        'state_dict': denoise_net.state_dict(),
                        'optimizer': optimizer_denoise.state_dict()
                    }, os.path.join(model_dir, "denoise_best.pth"))
                    
                    # 保存深度网络
                    torch.save({
                        'epoch': epoch,
                        'state_dict': depth_net.state_dict(),
                        'optimizer': optimizer_depth.state_dict()
                    }, os.path.join(model_dir, "depth_best.pth"))
                
                print("[Ep %d it %d\t PSNR: %.4f\t] ----  [best_Ep %d best_it %d Best_PSNR %.4f] " % 
                      (epoch, i, psnr_val_rgb, best_epoch, best_iter, best_psnr))
            
            denoise_net.train()
            depth_net.train()
    
    # 更新学习率
    scheduler_denoise.step()
    scheduler_depth.step()
    
    # 计算平均损失和训练PSNR
    avg_denoise_loss = denoise_epoch_loss / len(train_loader)
    avg_depth_loss = depth_epoch_loss / len(train_loader)
    train_psnr = sum(batch_psnr) / len(batch_psnr)
    epoch_time = time.time() - epoch_start_time
    current_lr_denoise = scheduler_denoise.get_last_lr()[0]
    current_lr_depth = scheduler_depth.get_last_lr()[0]
    
    # 记录每个epoch的指标，与trainwithlog.py保持一致
    writer.add_scalar('Train/Loss', avg_denoise_loss, epoch)  # 使用去噪损失作为主要损失
    writer.add_scalar('Train/PSNR', train_psnr, epoch)
    writer.add_scalar('Train/Time', epoch_time, epoch)
    writer.add_scalar('Train/LearningRate', current_lr_denoise, epoch)  # 使用去噪网络的学习率
    
    # 额外记录深度网络相关指标
    writer.add_scalar('Train/Depth_Loss', avg_depth_loss, epoch)
    writer.add_scalar('Train/LR_Depth', current_lr_depth, epoch)
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tDepth Loss: {:.4f}\tTrain PSNR: {:.4f}\tVal PSNR: {:.4f}\tLearningRate {:.6f}".format(
        epoch, epoch_time, avg_denoise_loss, avg_depth_loss, train_psnr, psnr_val_rgb, current_lr_denoise))
    print("------------------------------------------------------------------")
    
    # 保存最新模型
    torch.save({
        'epoch': epoch,
        'state_dict': denoise_net.state_dict(),
        'optimizer': optimizer_denoise.state_dict()
    }, os.path.join(model_dir, "denoise_latest.pth"))
    
    torch.save({
        'epoch': epoch,
        'state_dict': depth_net.state_dict(),
        'optimizer': optimizer_depth.state_dict()
    }, os.path.join(model_dir, "depth_latest.pth"))

# 关闭TensorBoard写入器
writer.close()