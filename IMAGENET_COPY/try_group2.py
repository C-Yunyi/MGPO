import re
import shutil
from pathlib import Path


VAL_RE = re.compile(r"ILSVRC2012_val_(\d{8})\.JPEG$", re.IGNORECASE)


def organize_val_images_by_id(val_dir: str, wnid_file: str, gt_file: str, dry_run: bool = False):
    val_dir = Path(val_dir)
    wnid_file = Path(wnid_file)
    gt_file = Path(gt_file)

    assert val_dir.exists(), f"val_dir not found: {val_dir}"
    assert wnid_file.exists(), f"wnid_file not found: {wnid_file}"
    assert gt_file.exists(), f"gt_file not found: {gt_file}"

    wnids = [line.strip() for line in wnid_file.read_text().splitlines() if line.strip()]
    if len(wnids) != 1000:
        raise ValueError(f"Expected 1000 wnids, got {len(wnids)}")

    gt = [int(line.strip()) for line in gt_file.read_text().splitlines() if line.strip()]
    # gt[i] is class_id for val image index (i+1)
    if len(gt) < 50000:
        print(f"[WARN] Ground truth file has {len(gt)} lines, expected 50000.")

    moved = 0
    skipped = 0

    for img_path in val_dir.iterdir():
        if not img_path.is_file():
            continue

        m = VAL_RE.match(img_path.name)
        if not m:
            # ignore weird files, e.g. txt, tar, or renamed images
            skipped += 1
            continue

        img_id = int(m.group(1))  # 1..50000
        if img_id < 1 or img_id > len(gt):
            print(f"[WARN] Image id out of range: {img_path.name}")
            skipped += 1
            continue

        class_id = gt[img_id - 1]  # 1..1000
        wnid = wnids[class_id - 1]

        target_dir = val_dir / wnid
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / img_path.name

        if dry_run:
            print(f"[DRY] {img_path.name} (id={img_id}) -> {wnid}/")
        else:
            shutil.move(str(img_path), str(target_path))
        moved += 1

    print(f"Done. Moved {moved} images. Skipped {skipped} non-matching files.")


if __name__ == "__main__":
    organize_val_images_by_id(
        val_dir="/root/autodl-tmp/imagenet/val",
        wnid_file="/root/autodl-tmp/imagenet/class_indices.txt",
        gt_file="/root/autodl-tmp/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
        dry_run=False,  # 先 True 预演
    )
