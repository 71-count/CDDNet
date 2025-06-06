import os

# 设置低质量图片文件夹的路径
low_quality_directory = r'D:\Desktop\CTdataset\test\IN'

# 获取文件夹中所有的低质量图片文件
low_quality_files = sorted(os.listdir(low_quality_directory))

# 遍历低质量图片文件并重命名
for index, file in enumerate(low_quality_files, start=1):
    # 获取文件扩展名（假设所有文件都是 jpg 格式）
    file_extension = os.path.splitext(file)[1]

    # 生成新的文件名
    new_file_name = f"{index}{file_extension}"

    # 构造文件的完整路径
    old_file_path = os.path.join(low_quality_directory, file)
    new_file_path = os.path.join(low_quality_directory, new_file_name)

    try:
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"成功将 {file} 重命名为 {new_file_name}")
    except Exception as e:
        print(f"重命名 {file} 失败: {e}")
