import os
import shutil

# 源文件夹路径
source_dir = './ultrasound/train'
# 目标文件夹路径
target_dir = './coco_mix_ulso/train'

# 图片后缀列表（如果有其他格式也可以加入）
img_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.gif']

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

for filename in os.listdir(source_dir):
    # 检查文件是否为图片
    if os.path.splitext(filename)[1] in img_exts:
        # 创建新的文件名
        new_filename = 'ultrasound_' + filename
        # 复制文件
        shutil.copyfile(os.path.join(source_dir, filename), os.path.join(target_dir, new_filename))
        
print('图片复制完毕!')

