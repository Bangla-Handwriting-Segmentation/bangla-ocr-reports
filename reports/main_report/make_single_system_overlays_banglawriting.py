"""Per-system-only overlays on BanglaWriting (GT + one system at a time), so each
system's raw inference quality can be judged without the other system's boxes
cluttering the page."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw

from bench_bn_drishti_syn_hw import predict_lines, predict_words
from bench_swin_bn_drishti_syn_hw import PROJECT_ROOT, load_swin_detectors, swin_predict
from bench_swin_bn_drishti_banglawriting import load_banglawriting_pages
from pseudolabel.adapters.bn_drishti import load_line_model, load_word_model
from pseudolabel.adapters.sources import decoded_rgb

GT_COLOR = (33, 150, 243)
BND_COLOR = (244, 67, 54)
SWIN_COLOR = (76, 175, 80)

STEMS = ["0_21_0", "10_20_0", "105_24_1", "111_18_0", "113_16_0"]
OUT_DIR = PROJECT_ROOT / "report" / "assets" / "overlays_banglawriting_single"


def draw_rects(image, gt_boxes, pred_boxes, pred_color):
    picture = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(picture)
    for box in gt_boxes:
        draw.rectangle(box, outline=GT_COLOR, width=3)
    for box, _ in pred_boxes:
        draw.rectangle(box, outline=pred_color, width=2)
    return picture


def draw_polygons(image, gt_boxes, instances):
    picture = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(picture)
    for box in gt_boxes:
        draw.rectangle(box, outline=GT_COLOR, width=3)
    for inst in instances:
        points = [(float(x), float(y)) for x, y in inst["polygon"]]
        draw.line(points + [points[0]], fill=SWIN_COLOR, width=2)
    return picture


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = PROJECT_ROOT / "BanglaWriting" / "converted" / "converted"
    pages = {p["stem"]: p for p in load_banglawriting_pages(root)}

    bnd_line_model = load_line_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "line_model_best.pt", str(device))
    bnd_word_model = load_word_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "word_model_best.pt", str(device))
    swin_detectors = load_swin_detectors(
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_lines" / "best.pth",
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_words" / "best.pth",
        device, score=0.5, mask_threshold=0.5, batch=16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in STEMS:
        page = pages.get(stem)
        if page is None:
            print(f"[skip] {stem} not found")
            continue
        image = decoded_rgb(page["image_path"])
        pil_page = Image.fromarray(image)
        gt_words = page["boxes"]

        bnd_line_preds = predict_lines(bnd_line_model, image)
        bnd_word_preds = predict_words(bnd_word_model, image, [box for box, _ in bnd_line_preds])
        _, swin_words = swin_predict(swin_detectors, pil_page, 1.0)

        draw_rects(image, gt_words, bnd_word_preds, BND_COLOR).save(
            OUT_DIR / f"bnd_only_{stem}.jpg", quality=90)
        draw_polygons(image, gt_words, swin_words).save(
            OUT_DIR / f"swin_only_{stem}.jpg", quality=90)
        print(f"[done] {stem}: gt={len(gt_words)} bnd={len(bnd_word_preds)} swin={len(swin_words)}")


if __name__ == "__main__":
    main()
