import argparse

import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image



# network = dehazeformer_v5.DDHAZE_V5()
# model = DepthNet_v1.DN()
# dehaze_state_dict = torch.load('saved_models/indoor/DDHAZE-V5.pth')  # 将去雾网络load
# depth_state_dict = torch.load('saved_models/indoor/DN.pth')  # 将深度估计网络load
# network.load_state_dict(dehaze_state_dict['dehaze_net'])
# model.load_state_dict(depth_state_dict['depth_net'])

# Pre_Midas = DPTDepthModel(path='midas/dpt_swin2_tiny_256.pt', backbone="swin2t16_256", non_negative=True, )
# Pre_Midas.requires_grad_(False)
# model = Pre_Midas
# model.eval()

# download_model_if_doesnt_exist(args.model_name)
import networks

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description='Simple testing funtion for Monodepthv2 models.')
#
#     parser.add_argument('--image_path', type=str, default='hazy',
#                         help='path to a test image or folder of images')#原来有个required=True
#     parser.add_argument('--model_name', type=str,default='RA-Depth',
#                         help='name of a pretrained model to use',
#                         choices=[
#                             "RA-Depth"])
#     parser.add_argument('--ext', type=str,
#                         help='image extension to search for in folder', default="jpg")
#     parser.add_argument("--no_cuda",
#                         help='if set, disables CUDA',
#                         action='store_true')
#
#     return parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_path = os.path.join("models", 'RA-Depth')
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
# encoder = networks.ResnetEncoder(18, False)
encoder = networks.hrnet18(False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("   Loading pretrained decoder")
depth_decoder = networks.DepthDecoder_MSF(
    num_ch_enc=encoder.num_ch_enc, scales=range(1))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()


# 图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 读取并预处理图像
image = Image.open('test/0001_0.8_0.2.jpg').convert('RGB')
image = transform(image).unsqueeze(0).to(device)

# 使用模型进行推理
with torch.no_grad():
    features = encoder(image)
    output = depth_decoder(features)

# 后处理
# output = np.array(output)
# output = output.squeeze(0)
output = output[0].squeeze()

output = (output + 1) / 2  # 反归一化

# 保存结果
output_image = transforms.ToPILImage()(output)
output_image.save('output_image.jpg')
