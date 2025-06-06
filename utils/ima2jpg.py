import os
import pydicom
from PIL import Image
import numpy as np

# 设置输入和输出目录
input_dir = r'D:\Desktop\CTdataset\3mm D45\quarter_3mm_sharp\L506\quarter_3mm_sharp'  
output_dir = r'D:\Desktop\CTdataset\test\IN'  

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取输入目录下的所有.IMA文件
dicom_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.ima')]

# 遍历每个DICOM文件
for dicom_file in dicom_files:
    try:
        # 构建文件的完整路径
        dicom_path = os.path.join(input_dir, dicom_file)
        
        # 读取DICOM文件
        dicom_data = pydicom.dcmread(dicom_path)

        # 获取像素数据并转换为HU值
        pixel_array = dicom_data.pixel_array
        slope = dicom_data.get('RescaleSlope', 1)
        intercept = dicom_data.get('RescaleIntercept', 0)
        pixel_array = slope * pixel_array + intercept

        # 设置窗宽和窗位
        window_center = -200  # 窗位最开始-200
        window_width = 1000   # 窗宽1000

        # 应用窗宽窗位
        min_window = window_center - window_width // 2
        max_window = window_center + window_width // 2
        pixel_array = np.clip(pixel_array, min_window, max_window)
        pixel_array = (pixel_array - min_window) / (max_window - min_window) * 255
        pixel_array = pixel_array.astype(np.uint8)

        # 将处理后的数组转换为Image对象
        image = Image.fromarray(pixel_array)

        # 构建输出文件名
        if dicom_file.lower().endswith('.ima'):
            output_filename = dicom_file[:-4] + '.jpg'  # 去除'.IMA'扩展名并添加'.jpg'
        else:
            output_filename = dicom_file + '.jpg'  # 如果没有'.IMA'扩展名，直接添加'.jpg'

        output_path = os.path.join(output_dir, output_filename)

        # 保存为JPEG格式
        image.save(output_path, 'JPEG')
        print(f"已转换并保存: {output_path}")
    except Exception as e:
        print(f"处理文件 {dicom_file} 时出错: {e}")
