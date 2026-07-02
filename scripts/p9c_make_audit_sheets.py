import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


MANIFEST = Path("data/gqa/manifests/gqa_evidence_scaled_250.json")
IMAGE_DIR = Path("data/gqa/scaled_images")
OUT_DIR = Path("results/p9c/audit_sheets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

samples = json.loads(MANIFEST.read_text(encoding="utf-8"))
font = ImageFont.load_default(size=18)
small = ImageFont.load_default(size=16)
tile_w, tile_h = 600, 500


def fit_image(image, max_w, max_h):
    scale = min(max_w / image.width, max_h / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS), scale


for page_start in range(0, len(samples), 10):
    page_samples = samples[page_start : page_start + 10]
    sheet = Image.new("RGB", (tile_w * 5, tile_h * 2), "white")
    for offset, sample in enumerate(page_samples):
        index = page_start + offset
        image_id = str(sample["image_id"])
        image = Image.open(IMAGE_DIR / f"{image_id}.jpg").convert("RGB")
        obj = sample["critical_objects"][0]
        draw = ImageDraw.Draw(image)
        width = max(3, round(min(image.size) / 100))
        draw.rectangle(obj["bbox_xyxy"], outline=(255, 0, 0), width=width)
        shown, scale = fit_image(image, 570, 335)
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        x = (tile_w - shown.width) // 2
        tile.paste(shown, (x, 5))
        td = ImageDraw.Draw(tile)
        y = 345
        lines = [
            f"INDEX {index:03d} | QID {sample['question_id']} | {sample['category']}",
            f"Q: {sample['question']}",
            f"A: {sample['answer']} | object: {obj['name']}",
            f"bbox: {[round(v, 1) for v in obj['bbox_xyxy']]}",
        ]
        for line_number, line in enumerate(lines):
            wrapped = textwrap.wrap(line, width=61) or [""]
            for part in wrapped:
                td.text((10, y), part, fill="black", font=font if line_number == 0 else small)
                y += 21
        col, row = offset % 5, offset // 5
        sheet.paste(tile, (col * tile_w, row * tile_h))
    out = OUT_DIR / f"audit_{page_start:03d}_{page_start + len(page_samples) - 1:03d}.jpg"
    sheet.save(out, quality=92)
    print(out)

print(f"Created {(len(samples) + 9) // 10} audit sheets in {OUT_DIR}")
