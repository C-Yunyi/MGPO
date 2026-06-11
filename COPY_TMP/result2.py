import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

img1_root = '/root/autodl-tmp/results/idc_minimax'
img2_root = '/root/autodl-tmp/results/idc_mgpo'
output_path = '/root/autodl-tmp/results/idc_compare.png'

# =========================
# config
# =========================
n_classes = 10
n_imgs_per_class = 5
img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

thumb_size = 96
col_gap = 8
group_gap = 28
row_gap = 10
left_label_width = 0
top_title_height = 60
margin = 20
border_color = (180, 180, 180)
bg_color = (255, 255, 255)

# =========================
# helpers
# =========================
def list_subdirs(root):
    return sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

def list_images(folder):
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(img_exts)
    ]
    # 让 0.png,1.png,...,9.png 按数字顺序排
    def sort_key(x):
        stem = os.path.splitext(x)[0]
        return int(stem) if stem.isdigit() else stem
    return sorted(files, key=sort_key)

def prettify_label(name):
    return name.replace('_', ' ').replace('-', ' ')

def resize_and_pad(img, size):
    img = img.convert('RGB')
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new('RGB', (size, size), (255, 255, 255))
    x = (size - img.width) // 2
    y = (size - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas

def load_font(size):
    # 尝试 serif 字体；找不到就退回默认字体
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

# =========================
# class discovery
# =========================
classes1 = list_subdirs(img1_root)
classes2 = list_subdirs(img2_root)
common_classes = sorted(set(classes1) & set(classes2))

if not common_classes:
    raise RuntimeError("No common class folders found.")

common_classes = common_classes[:n_classes]

print("Using classes:")
for c in common_classes:
    print("  ", c)

# =========================
# validate and preview file lists
# =========================
for cls in common_classes:
    folder1 = os.path.join(img1_root, cls)
    folder2 = os.path.join(img2_root, cls)

    imgs1 = list_images(folder1)
    imgs2 = list_images(folder2)

    print(f"\nClass: {cls}")
    print("  Minimax:", imgs1[:n_imgs_per_class])
    print("  MGPO   :", imgs2[:n_imgs_per_class])

    if len(imgs1) < n_imgs_per_class:
        raise RuntimeError(f"{folder1} has only {len(imgs1)} images.")
    if len(imgs2) < n_imgs_per_class:
        raise RuntimeError(f"{folder2} has only {len(imgs2)} images.")

# =========================
# canvas size
# =========================
panel_width = n_imgs_per_class * thumb_size + (n_imgs_per_class - 1) * col_gap
row_height = thumb_size
canvas_width = (
    margin
    + left_label_width
    + panel_width
    + group_gap
    + panel_width
    + margin
)
canvas_height = (
    margin
    + top_title_height
    + len(common_classes) * row_height
    + (len(common_classes) - 1) * row_gap
    + margin
)

canvas = Image.new("RGB", (canvas_width, canvas_height), bg_color)
draw = ImageDraw.Draw(canvas)

title_font = load_font(26)
label_font = load_font(18)

x_minimax_start = margin + left_label_width
x_mgpo_start = x_minimax_start + panel_width + group_gap
sep_x = x_minimax_start + panel_width + group_gap // 2

# =========================
# titles
# =========================
def draw_centered_text(draw, x_center, y_center, text, font, fill=(0, 0, 0)):
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.text((x_center - w / 2, y_center - h / 2), text, font=font, fill=fill)

draw_centered_text(
    draw,
    x_minimax_start + panel_width / 2,
    margin + top_title_height / 2,
    "Minimax",
    title_font
)

draw_centered_text(
    draw,
    x_mgpo_start + panel_width / 2,
    margin + top_title_height / 2,
    "MGPO",
    title_font
)

# dashed separator
dash_len = 10
gap_len = 8
y0 = margin
y1 = canvas_height - margin
y = y0
while y < y1:
    draw.line((sep_x, y, sep_x, min(y + dash_len, y1)), fill=(120, 120, 120), width=3)
    y += dash_len + gap_len

# =========================
# paste images row by row
# =========================
for row_idx, cls in enumerate(common_classes):
    y_top = margin + top_title_height + row_idx * (thumb_size + row_gap)
    y_center = y_top + thumb_size / 2

    # # class label
    # label = prettify_label(cls)
    # bbox = draw.textbbox((0, 0), label, font=label_font)
    # tw = bbox[2] - bbox[0]
    # th = bbox[3] - bbox[1]
    # draw.text(
    #     (margin + left_label_width - 10 - tw, y_center - th / 2),
    #     label,
    #     font=label_font,
    #     fill=(0, 0, 0)
    # )

    folder1 = os.path.join(img1_root, cls)
    folder2 = os.path.join(img2_root, cls)

    imgs1 = list_images(folder1)[:n_imgs_per_class]
    imgs2 = list_images(folder2)[:n_imgs_per_class]

    # left: Minimax
    for col_idx, fname in enumerate(imgs1):
        img_path = os.path.join(folder1, fname)
        img = Image.open(img_path)
        img = resize_and_pad(img, thumb_size)

        x = x_minimax_start + col_idx * (thumb_size + col_gap)
        canvas.paste(img, (x, y_top))
        draw.rectangle(
            [x, y_top, x + thumb_size, y_top + thumb_size],
            outline=border_color,
            width=1
        )

    # right: MGPO
    for col_idx, fname in enumerate(imgs2):
        img_path = os.path.join(folder2, fname)
        img = Image.open(img_path)
        img = resize_and_pad(img, thumb_size)

        x = x_mgpo_start + col_idx * (thumb_size + col_gap)
        canvas.paste(img, (x, y_top))
        draw.rectangle(
            [x, y_top, x + thumb_size, y_top + thumb_size],
            outline=border_color,
            width=1
        )

# =========================
# save directly
# =========================
canvas.save(output_path)
pdf_path = output_path.replace(".png", ".pdf")
canvas.save(pdf_path, "PDF", resolution=300)

print(f"Saved PNG: {output_path}")
print(f"Saved PDF: {pdf_path}")