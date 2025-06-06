import os
import shutil
import random
import glob

# 设置路径
gt_dir = "CT/GT"  # 清晰图像目录
in_dir = "CT/IN"  # 噪点图像目录

# 创建训练集、验证集和测试集目录
train_gt_dir = "CT/train/GT"
train_in_dir = "CT/train/IN"
val_gt_dir = "CT/val/GT"
val_in_dir = "CT/val/IN"
test_gt_dir = "CT/test/GT"
test_in_dir = "CT/test/IN"

# 确保目录存在
for directory in [train_gt_dir, train_in_dir, val_gt_dir, val_in_dir, test_gt_dir, test_in_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# 获取GT目录下的所有图像
gt_images = glob.glob(os.path.join(gt_dir, "*.jpg"))

# 随机打乱图像列表
random.seed(42)  # 设置随机种子以确保可重复性
random.shuffle(gt_images)

# 计算划分数量
total_images = len(gt_images)
train_count = int(total_images * 0.7)
val_count = int(total_images * 0.2)
# 测试集为剩余的图像

# 划分数据集
train_images = gt_images[:train_count]
val_images = gt_images[train_count:train_count + val_count]
test_images = gt_images[train_count + val_count:]

print(f"总图像数: {total_images}")
print(f"训练集: {len(train_images)} 图像")
print(f"验证集: {len(val_images)} 图像")
print(f"测试集: {len(test_images)} 图像")

# 复制图像到相应的目录
def copy_images(image_list, gt_source_dir, in_source_dir, gt_target_dir, in_target_dir):
    for gt_path in image_list:
        # 获取基本文件名
        base_name = os.path.basename(gt_path)
        
        # 构建对应的IN图像路径
        in_path = os.path.join(in_source_dir, base_name)
        
        # 检查IN图像是否存在
        if os.path.exists(in_path):
            # 复制GT图像
            shutil.copy2(gt_path, os.path.join(gt_target_dir, base_name))
            
            # 复制IN图像
            shutil.copy2(in_path, os.path.join(in_target_dir, base_name))
            
            print(f"已处理: {base_name}")
        else:
            print(f"警告: 未找到与 {base_name} 对应的IN图像")

# 复制训练集图像
print("\n处理训练集...")
copy_images(train_images, gt_dir, in_dir, train_gt_dir, train_in_dir)

# 复制验证集图像
print("\n处理验证集...")
copy_images(val_images, gt_dir, in_dir, val_gt_dir, val_in_dir)

# 复制测试集图像
print("\n处理测试集...")
copy_images(test_images, gt_dir, in_dir, test_gt_dir, test_in_dir)

print("\n数据集划分完成！")