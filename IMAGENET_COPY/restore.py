import shutil
from pathlib import Path


IMG_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}


def restore_val_to_flat(val_dir: str, remove_empty_dirs: bool = True, dry_run: bool = False):
    """
    Move all images from val/*/ back to val/ (flatten).
    Optionally remove empty class folders afterwards.
    """
    val_dir = Path(val_dir)
    assert val_dir.exists(), f"val_dir not found: {val_dir}"

    moved = 0
    skipped = 0

    # Iterate subdirectories under val/
    for sub in val_dir.iterdir():
        if not sub.is_dir():
            continue

        # Skip hidden/system dirs if any
        if sub.name.startswith("."):
            continue

        # Move files in this subdir back to val root
        for f in sub.iterdir():
            if not f.is_file():
                continue
            if f.suffix.lower() not in IMG_EXTS:
                skipped += 1
                continue

            target = val_dir / f.name

            # If a file with same name already exists, avoid overwrite
            if target.exists():
                # rename safely
                target = val_dir / f"RESTORED__{sub.name}__{f.name}"

            if dry_run:
                print(f"[DRY] {f} -> {target}")
            else:
                shutil.move(str(f), str(target))
            moved += 1

        # Remove empty directory
        if remove_empty_dirs and not dry_run:
            try:
                sub.rmdir()  # only removes if empty
            except OSError:
                # not empty or cannot remove, ignore
                pass

    print(f"Done. Moved {moved} images back to {val_dir}. Skipped {skipped} non-image files.")


if __name__ == "__main__":
    restore_val_to_flat(
        val_dir="/root/autodl-tmp/imagenet/val",
        remove_empty_dirs=True,
        dry_run=False,   # ✅ 你可以先改 True 预演
    )
