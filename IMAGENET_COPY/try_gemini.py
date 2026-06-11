import os
import shutil
from tqdm import tqdm

# ================= 配置路径 =================
# 您的平铺结构的 val 文件夹路径
val_dir = '/root/autodl-tmp/imagenet/val' 
# 官方的 ground truth 文件路径
gt_file = '/root/autodl-tmp/devkitILSVRC2012_devkit_t12/ILSVRC2012_validation_ground_truth.txt'
# 您项目中的类别索引文件
class_idx_file = '/root/autodl-tmp/imagenet/class_indices.txt'

def reorganize_val():
    # 1. 读取 1000 个类名的 synsets
    with open(class_idx_file, 'r') as f:
        synsets = [line.strip() for line in f.readlines()]

    # 2. 读取 50,000 个图片的类别标签 (1-1000)
    with open(gt_file, 'r') as f:
        gt_labels = [int(line.strip()) for line in f.readlines()]

    # 3. 获取 val 目录下所有的 JPEG 文件并排序
    # 确保 ILSVRC2012_val_00000001.JPEG 排在第一位
    images = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
    
    if len(images) != 50000:
        print(f"警告：检测到图片数量为 {len(images)}，标准 ImageNet 验证集应为 50000 张。")

    print("开始整理文件夹...")
    # 4. 遍历并移动
    for i, img_name in enumerate(tqdm(images)):
        # 获取该图片对应的类别索引 (gt_labels 里的值通常是 1-1000)
        label_idx = gt_labels[i] - 1 
        # 获取文件夹名
        target_synset = synsets[label_idx]
        
        # 创建目标文件夹
        target_folder = os.path.join(val_dir, target_synset)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # 移动文件
        src_path = os.path.join(val_dir, img_name)
        dst_path = os.path.join(target_folder, img_name)
        shutil.move(src_path, dst_path)

    print("整理完成！")

if __name__ == "__main__":
    reorganize_val()