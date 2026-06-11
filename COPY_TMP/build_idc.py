import os
import shutil
from tqdm import tqdm

# ===============================
# CONFIG
# ===============================

IMAGENET_ROOT = "/root/autodl-tmp/imagenet"   # 原始 ImageNet-1K 根目录
OUT_ROOT = "/root/autodl-tmp/imagenetIDC"     # 输出 ImageNetIDC

SPLITS = ["train", "val"]

IDC_CLASSES = [
    "n01749939",
    "n01773797",
    "n02091831",
    "n02107142",
    "n02488291",
    "n02869837",
    "n03062245",
    "n04517823",
    "n04589890",
    "n13037406",
]

COPY_MODE = "symlink"  # "copy" 或 "symlink"


# ===============================
# UTILS
# ===============================

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def copy_or_link(src, dst):
    if COPY_MODE == "copy":
        shutil.copy2(src, dst)
    elif COPY_MODE == "symlink":
        if not os.path.exists(dst):
            os.symlink(src, dst)
    else:
        raise ValueError("COPY_MODE must be 'copy' or 'symlink'")


# ===============================
# MAIN
# ===============================

def main():
    print("Building ImageNetIDC dataset...")
    print(f"Source ImageNet: {IMAGENET_ROOT}")
    print(f"Target path:    {OUT_ROOT}")
    print(f"Copy mode:      {COPY_MODE}")

    for split in SPLITS:
        print(f"\nProcessing split: {split}")

        src_split = os.path.join(IMAGENET_ROOT, split)
        dst_split = os.path.join(OUT_ROOT, split)
        safe_mkdir(dst_split)

        for cls in IDC_CLASSES:
            src_cls_dir = os.path.join(src_split, cls)
            dst_cls_dir = os.path.join(dst_split, cls)

            if not os.path.isdir(src_cls_dir):
                raise FileNotFoundError(f"Missing class dir: {src_cls_dir}")

            safe_mkdir(dst_cls_dir)

            images = sorted(os.listdir(src_cls_dir))
            print(f"  {cls}: {len(images)} images")

            for img in tqdm(images, leave=False):
                src_img = os.path.join(src_cls_dir, img)
                dst_img = os.path.join(dst_cls_dir, img)
                copy_or_link(src_img, dst_img)

    print("\n✅ ImageNetIDC dataset construction finished!")
    print(f"Dataset location: {OUT_ROOT}")


if __name__ == "__main__":
    main()