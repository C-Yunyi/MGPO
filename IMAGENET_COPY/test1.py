import os
import shutil
from scipy.io import loadmat
from tqdm import tqdm

# ================= 配置路径 (请根据你的实际路径修改) =================
val_dir = '/root/autodl-tmp/imagenet/val'               # 平铺的验证集路径
gt_file = '/root/autodl-tmp/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt'
meta_file = '/root/autodl-tmp/devkit/ILSVRC2012_devkit_t12/data/meta.mat'
# =================================================================

def fix_val_dataset():
    # 1. 加载 meta.mat 获取 ID 到 WNID 的映射
    print("正在加载类别映射信息...")
    meta = loadmat(meta_file)
    # synsets(i).WNID 对应 ILSVRC2012_ID == i
    synsets = meta['synsets']
    # 建立 ID -> WNID 的映射表 (ID从1开始)
    id_to_wnid = {int(s[0][0][0][0]): s[0][1][0] for s in synsets}

    # 2. 读取 50,000 个图片的地面真值 (数字ID)
    with open(gt_file, 'r') as f:
        gt_ids = [int(line.strip()) for line in f.readlines()]

    # 3. 获取并排序所有验证集图片
    images = sorted([f for f in os.listdir(val_dir) if f.endswith('.JPEG')])
    
    if len(images) != 50000:
        print(f"⚠️ 警告: 发现 {len(images)} 张图片，标准应为 50,000 张。")

    print("开始按照类别整理文件夹...")
    # 4. 遍历图片并移动到 WNID 文件夹
    for i, img_name in enumerate(tqdm(images)):
        # 获取该图片的 ID 和对应的 WNID (文件夹名)
        current_id = gt_ids[i]
        target_wnid = id_to_wnid[current_id]
        
        # 创建目标路径: val/nXXXXXXXX/
        target_path = os.path.join(val_dir, target_wnid)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
            
        # 移动文件
        src = os.path.join(val_dir, img_name)
        dst = os.path.join(target_path, img_name)
        shutil.move(src, dst)

    print(f"\n✅ 整理完成！验证集现在位于: {val_dir}")
    print("结构示例: val/n01440764/ILSVRC2012_val_00000293.JPEG")

if __name__ == "__main__":
    fix_val_dataset()