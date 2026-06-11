import shutil
from pathlib import Path


def organize_val_images(val_dir: str, wnid_file: str, gt_file: str, dry_run: bool = False):
    val_dir = Path(val_dir)
    wnid_file = Path(wnid_file)
    gt_file = Path(gt_file)

    assert val_dir.exists(), f"val_dir not found: {val_dir}"
    assert wnid_file.exists(), f"wnid_file not found: {wnid_file}"
    assert gt_file.exists(), f"gt_file not found: {gt_file}"

    # 1000 wnids, in order (class_id 1..1000 maps to wnids[0..999])
    wnids = [line.strip() for line in wnid_file.read_text().splitlines() if line.strip()]
    if len(wnids) != 1000:
        raise ValueError(f"Expected 1000 wnids, got {len(wnids)}")

    # 50000 labels (class_id in 1..1000)
    gt = [int(line.strip()) for line in gt_file.read_text().splitlines() if line.strip()]

    # only take images in val_dir root (not inside existing subfolders)
    images = sorted([p for p in val_dir.iterdir() if p.is_file() and p.suffix.lower() in [".jpeg", ".jpg", ".png"]])

    if len(images) != len(gt):
        print(f"[WARN] Found {len(images)} images in {val_dir}, but {len(gt)} labels in gt file.")
        print("       Will match by sorted filename order. Make sure filenames are ILSVRC2012_val_000xxxxx.JPEG")

    n = min(len(images), len(gt))
    moved = 0

    for i in range(n):
        img_path = images[i]
        class_id = gt[i]  # 1..1000
        wnid = wnids[class_id - 1]

        target_dir = val_dir / wnid
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / img_path.name

        if dry_run:
            print(f"[DRY] {img_path.name} -> {wnid}/")
        else:
            shutil.move(str(img_path), str(target_path))
        moved += 1

    print(f"Done. Organized {moved} images into wnid folders under: {val_dir}")


if __name__ == "__main__":
    organize_val_images(
        val_dir="../autodl-tmp/imagenet/val",
        wnid_file="../autodl-tmp/imagenet/class_indices.txt",
        gt_file="../autodl-tmp/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
        dry_run=False,  # 先改 True 预演也行
    )
