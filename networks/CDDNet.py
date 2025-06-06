
import torch
import torch.nn as nn

import time

from torch import Tensor
from utils.antialias import Edge as edge
# from antialias import Edge as edge
from torch.nn import functional as F
    
class Denoise(nn.Module):   
    def __init__(self):
        super(Denoise, self).__init__()   
        self.in_ch = 3
        self.n_fea = 3

        self.cddnet = CDD_Net()

        self.conv1 = nn.Conv2d(in_channels=self.in_ch, out_channels=self.n_fea , kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.n_fea, out_channels=self.n_fea , kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.n_fea, out_channels=self.in_ch , kernel_size=1)\
        
        self.ae = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, img):

        intensity_noise = self.intensity_cal(img)

        noise_baseline = 0.9

        contrast_noise = self.contrast_cal(img, intensity_noise)

        net_input = torch.cat((img, contrast_noise), dim =1)
        estimation = self.cddnet(net_input, img)
        prediction = self.restore(img, noise_baseline, estimation)

        return  prediction 

    def restore(self, img, noise_baseline, estimation):

        denominator = 1 - estimation / noise_baseline
 
        denominator = torch.clamp(denominator, min=1e-6)

        prediction = (img - estimation) / denominator
        return prediction


    def intensity_cal(self,img):   
        intensity = torch.mean(img, dim=1, keepdim=True)
        return intensity

    def contrast_cal(self,img, intensity):

        min, _ = torch.min(img, dim = 1, keepdim=True)

        me = torch.finfo(torch.float32).eps
        contrast = 1.0 - min / (intensity + me)

        return contrast
    
class DRDB(nn.Module):
    def __init__(self, in_ch=1, growth_rate=32):
        super(DRDB, self).__init__()
        in_ch_ = in_ch
        self.Dcov1 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov3 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov4 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov5 = nn.Conv2d(in_ch_, growth_rate, 3, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)
        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.Dcov4(x3)
        x4 = F.relu(x4)
        x4 = torch.cat([x3, x4], dim=1)
        x5 = self.Dcov5(x4)
        x5 = F.relu(x5)
        x5 = torch.cat([x4, x5], dim=1)
        x6 = self.conv(x5)
        out = x + F.relu(x6)
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class LightDRDB(nn.Module):
    def __init__(self, in_ch, growth_rate=16):
        super(LightDRDB, self).__init__()
        in_ch_ = in_ch
        # 使用深度可分离卷积替代标准卷积，减少参数量
        self.Dcov1 = DepthwiseSeparableConv(in_ch_, growth_rate, padding=2, dilation=2)
        in_ch_ += growth_rate
        self.Dcov2 = DepthwiseSeparableConv(in_ch_, growth_rate, padding=2, dilation=2)
        in_ch_ += growth_rate
        # 减少DRDB中的层数，从5层减少到3层
        self.Dcov3 = DepthwiseSeparableConv(in_ch_, growth_rate, padding=2, dilation=2)
        in_ch_ += growth_rate
        
        # 1x1卷积用于特征融合
        self.conv = nn.Conv2d(in_ch_, in_ch, 1, padding=0)

    def forward(self, x):
        x1 = self.Dcov1(x)
        x1 = F.relu(x1)
        x1 = torch.cat([x, x1], dim=1)

        x2 = self.Dcov2(x1)
        x2 = F.relu(x2)
        x2 = torch.cat([x1, x2], dim=1)

        x3 = self.Dcov3(x2)
        x3 = F.relu(x3)
        x3 = torch.cat([x2, x3], dim=1)

        x4 = self.conv(x3)
        out = x + F.relu(x4)
        return out

class LightDN(nn.Module):
    def __init__(self):
        super(LightDN, self).__init__()
        # 减少基础通道数
        base_ch = 16  # 原来是24/48/96/128
        growth_rate = 16  # 原来是32
        
        # 编码器部分
        self.DRDB_layer1 = LightDRDB(in_ch=3, growth_rate=growth_rate)
        self.down1 = nn.Sequential(
            DepthwiseSeparableConv(3, base_ch, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.DRDB_layer2 = LightDRDB(in_ch=base_ch, growth_rate=growth_rate)
        self.down2 = nn.Sequential(
            DepthwiseSeparableConv(base_ch, base_ch*2, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.DRDB_layer3 = LightDRDB(in_ch=base_ch*2, growth_rate=growth_rate)
        self.down3 = nn.Sequential(
            DepthwiseSeparableConv(base_ch*2, base_ch*4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 瓶颈层
        self.DRDB_layer4 = LightDRDB(in_ch=base_ch*4, growth_rate=growth_rate)
        
        # 解码器部分 - 使用更轻量级的上采样结构
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(base_ch*4, base_ch*2, padding=1)
        )
        self.DRDB_layer5 = LightDRDB(in_ch=base_ch*2, growth_rate=growth_rate)
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(base_ch*2, base_ch, padding=1)
        )
        self.DRDB_layer6 = LightDRDB(in_ch=base_ch, growth_rate=growth_rate)
        
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparableConv(base_ch, 3, padding=1)
        )
        
        # 最终输出层
        self.final_conv = nn.Conv2d(3, 1, 1)

    def forward(self, x):
        # 编码器路径
        x1 = self.DRDB_layer1(x)
        x1 = self.down1(x1)
        
        x2 = self.DRDB_layer2(x1)
        x2 = self.down2(x2)
        
        x3 = self.DRDB_layer3(x2)
        x3 = self.down3(x3)
        
        # 瓶颈层
        x4 = self.DRDB_layer4(x3)
        
        # 解码器路径
        x5 = self.up1(x4)
        x5 = self.DRDB_layer5(x5)
        
        x6 = self.up2(x5)
        x6 = self.DRDB_layer6(x6)
        
        x7 = self.up3(x6)
        
        # 最终输出
        depth = self.final_conv(x7)
        
        return depth

class LEGM(nn.Module):
    """
    Local Feature-Embedded Global Feature Extraction Module (LEGM)
    """
    def __init__(self, channel):
        super(LEGM, self).__init__()
        
        # 初始特征提取
        self.conv_in = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.norm = nn.BatchNorm2d(channel)
        
        # 三条线性路径
        self.linear_path1 = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.linear_path2 = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        self.linear_path3 = nn.Conv2d(channel, channel, kernel_size=1, bias=True)
        
        # Softmax处理前的卷积
        self.softmax_conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        
        # 特征融合后的卷积
        self.fusion_conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        self.fusion_conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=True)
        
        # MLP处理
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 初始特征提取
        x_in = self.conv_in(x)
        x_norm = self.norm(x_in)
        
        # 三条线性路径
        path1 = self.linear_path1(x_norm)
        path2 = self.linear_path2(x_norm)
        path3 = self.linear_path3(x_norm)
        
        # 特征交互 - 第一个乘法操作
        interact1 = path1 * path2
        
        # Softmax处理
        softmax_in = self.softmax_conv(interact1)
        # 模拟Softmax操作（在特征图上）
        softmax_out = F.softmax(softmax_in.flatten(2), dim=-1).view_as(softmax_in)
        
        # 特征交互 - 第二个乘法操作
        interact2 = softmax_out * path3
        
        # 特征融合 - 第一个加法操作
        fusion1 = interact2 + path1
        fusion1 = self.fusion_conv1(fusion1)
        
        # 特征融合 - 第二个加法操作
        fusion2 = fusion1 + path3
        fusion2 = self.fusion_conv2(fusion2)
        
        # MLP处理
        output = self.mlp(fusion2)
        
        # 残差连接
        output = output + x
        
        return output
    
class FEPC(nn.Module):

    def __init__(self, dim, n_div = 4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3 - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.la_conv = edge(channels=self.dim_conv3 ,filt_size=3,stride=1 )

        self.param = nn.Parameter(torch.empty(1).uniform_(0, 1))


    def forward(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2, x3 = torch.split(x, [self.dim_conv3, self.dim_conv3, self.dim_untouched], dim=1)
        device = x.device
        
        x3 = x3.to(device)
        
        x1 = self.partial_conv3(x1)
        x2_la = self.la_conv(x2)
        x2 = x2 + x2_la * self.param
        x1 = x1.to(device)
        x2 = x2.to(device)
        out = torch.cat((x1, x2, x3), 1)

        return out
    

class SCAB(nn.Module):

    def __init__(self, channel, n_div = 4):
        super().__init__()

        self.prelu = nn.PReLU()

        self.conv1 = nn.Conv2d(channel // 2, channel // 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(channel // 2, channel // 2, 1, 1, 0, bias=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel// 2, channel // 8, 1, padding=0, bias=True),
                nn.PReLU(),
                nn.Conv2d(channel // 8, channel// 2, 1, padding=0, bias=True),
                nn.PReLU()
        )

        self.pa = nn.Sequential(
                nn.Conv2d(channel// 2, channel // 8, 1, padding=0, bias=True),
                nn.PReLU(),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.PReLU()
        )

    def forward(self, x):
        A , B = torch.chunk(x, 2 ,dim = 1)

        #CA
        A1 = self.prelu(self.conv1(A))
        B1_ave = self.avg_pool(B)
        B1_ave = self.ca(B1_ave)
        B1 = B1_ave
        A1 = A1 * B1

        #pixel attention
        B2 = self.prelu(self.conv2(B))
        A2 = self.pa(A)
        B2 = A2 * B2

        out = torch.cat((A1,B2), 1)

        return out


class CDD_Net(nn.Module):

    def __init__(self):
        super(CDD_Net, self).__init__()

        # mainNet Architecture
        self.prelu = nn.PReLU()
        self.ch = 32
        self.n_div = 4

        self.conv_layer1 = nn.Conv2d(4, self.ch, 3, 1, 1, bias=True)
        self.depth_net = LightDN()
        self.drdb_featx1 = DRDB (in_ch = self.ch)
        self.drdb_imgfeat = DRDB(in_ch=3)
        self.drdb_depth = DRDB(in_ch=1)
        # 多模态特征融合
        self.feature_fusion = nn.Conv2d(self.ch + 1 + 3, self.ch, 1, 1, 0, bias=True)
        # 使用LEGM模块替换原有的注意力机制
        self.legm = LEGM(self.ch)
        self.conv_layer6 = nn.Conv2d(self.ch , 3, 1, 1, 0, bias=True)
        self.SCAB = SCAB(self.ch)

        self.spatial_mixing1 = FEPC(
            self.ch ,
            self.n_div,
            
        )
        self.PointConv1 = nn.Conv2d(self.ch , self.ch , 1, 1, 0, bias=True)

        self.spatial_mixing2 = FEPC(
            self.ch ,
            self.n_div,            
        )
        self.PointConv2 = nn.Conv2d(self.ch , self.ch , 1, 1, 0, bias=True)
        self.gate2 = nn.Conv2d(self.ch * 3, self.ch  , 1, 1, 0, bias=True)

    def forward(self, img, original_img):
        x1 = self.conv_layer1(img)
        x1 = self.prelu(x1)
        x1 = self.drdb_featx1(x1)
        
        H, W =original_img.shape[2:]
        depth_map = self.depth_net(original_img)
        depth_map = depth_map[:, :, :H, :W]
        depth_map = self.drdb_depth(depth_map)
        
        d = self.drdb_imgfeat(original_img)
        fused_feaatures = torch.cat([x1, depth_map, d], dim=1)
        fused_feaatures = self.feature_fusion(fused_feaatures)
        
        attended_features = self.legm(fused_feaatures)

        x22 = self.spatial_mixing1(attended_features)
        x2 = self.PointConv1(x22)
        x2 = self.prelu(x2)

        x33 = self.spatial_mixing2(x2)
        x3 = self.PointConv2(x33) 
        x3 = self.prelu(x3)
        
        gates = self.gate2(torch.cat((x1, x2, x3), 1))
        x6 = self.prelu(gates)

        x7 = self.SCAB(x6)
        
        x11 = self.conv_layer6(x7)

        return x11
    
    def swish(self,x):
        return x * torch.sigmoid(x)

@torch.no_grad()
def measure_latency(images, model, GPU=True, chan_last=False, half=False, num_threads=None, iter=200):
    """
    :param images: b, c, h, w
    :param model: model
    :param GPU: whther use GPU
    :param chan_last: data_format
    :param half: half precision
    :param num_threads: for cpu
    :return:
    """

    if GPU:
        model.cuda()
        model.eval()
        torch.backends.cudnn.benchmark = True

        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()

        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        tic1 = time.time()
        for i in range(iter):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on gpu {throughput}")
        print(f"batch_size {batch_size} latency on gpu {latency} ms")

        return throughput, latency
    else:
        model.eval()
        if num_threads is not None:
            torch.set_num_threads(num_threads)

        batch_size = images.shape[0]

        if chan_last:
            images = images.to(memory_format=torch.channels_last)
            model = model.to(memory_format=torch.channels_last)
        if half:
            images = images.half()
            model = model.half()
        for i in range(10):
            model(images)
        tic1 = time.time()
        for i in range(iter):
            model(images)
        tic2 = time.time()
        throughput = iter * batch_size / (tic2 - tic1)
        latency = 1000 * (tic2 - tic1) / iter
        print(f"batch_size {batch_size} throughput on cpu {throughput}")
        print(f"batch_size {batch_size} latency on cpu {latency} ms")

        return throughput, latency

if __name__ == "__main__":
    from thop import profile
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.enabled = False
    input = torch.ones(1, 3, 512, 512, dtype=torch.float, requires_grad=False).cuda()

    model = Denoise().cuda()

    out = model(input)

    flops, params = profile(model, inputs=(input,))
#
    print('input shape:', input.shape)
    print('parameters:', params/1e6, 'M')
    print('flops', flops/1e9 , 'G')
    print('output shape', out.shape)

    throughput, latency = measure_latency(input, model, GPU=True)

			

			
	