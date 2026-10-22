"""Full-corpus per-system inference on BanglaWriting (all 260 pages), each system's
predictions saved to its own folder (GT overlaid in blue for reference), so every page
can be visually judged system-by-system without the other system's boxes cluttering
the view.

Run:
    venv/bin/python report/make_all_banglawriting_inference.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from bench_bn_drishti_syn_hw import predict_lines, predict_words
from bench_swin_bn_drishti_syn_hw import PROJECT_ROOT, load_swin_detectors, swin_predict
from bench_swin_bn_drishti_banglawriting import load_banglawriting_pages
from pseudolabel.adapters.bn_drishti import load_line_model, load_word_model
from pseudolabel.adapters.sources import decoded_rgb

GT_COLOR = (33, 150, 243)
BND_COLOR = (244, 67, 54)
SWIN_COLOR = (76, 175, 80)

OUT_ROOT = PROJECT_ROOT / "bench_results" / "banglawriting_full_inference"
BND_DIR = OUT_ROOT / "bn_drishti"
SWIN_DIR = OUT_ROOT / "swin_ours"


def draw_rects(image, gt_boxes, pred_boxes, pred_color):
    picture = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(picture)
    for box in gt_boxes:
        draw.rectangle(box, outline=GT_COLOR, width=3)
    for box, _ in pred_boxes:
        draw.rectangle(box, outline=pred_color, width=2)
    return picture


def draw_polygons(image, gt_boxes, instances):
    """Draws the actual 50-point ink-tracing polygon (not a bbox reduction) with
    vertex markers, thick enough to visibly distinguish it from a rectangle even at
    full-page zoom-out scale."""
    picture = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(picture)
    for box in gt_boxes:
        draw.rectangle(box, outline=GT_COLOR, width=3)
    for inst in instances:
        points = [(float(x), float(y)) for x, y in inst["polygon"]]
        draw.line(points + [points[0]], fill=SWIN_COLOR, width=3)
        for (px, py) in points:
            draw.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(0, 90, 200))
    return picture


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = PROJECT_ROOT / "BanglaWriting" / "converted" / "converted"
    pages = load_banglawriting_pages(root)
    print(f"[data] {len(pages)} BanglaWriting pages")

    bnd_line_model = load_line_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "line_model_best.pt", str(device))
    bnd_word_model = load_word_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "word_model_best.pt", str(device))
    swin_detectors = load_swin_detectors(
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_lines" / "best.pth",
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_words" / "best.pth",
        device, score=0.5, mask_threshold=0.5, batch=16)

    BND_DIR.mkdir(parents=True, exist_ok=True)
    SWIN_DIR.mkdir(parents=True, exist_ok=True)

    started = time.time()
    bar = tqdm(pages, desc="pages", dynamic_ncols=True)
    for page in bar:
        image = decoded_rgb(page["image_path"])
        pil_page = Image.fromarray(image)
        gt_words = page["boxes"]

        bnd_line_preds = predict_lines(bnd_line_model, image)
        bnd_word_preds = predict_words(bnd_word_model, image, [box for box, _ in bnd_line_preds])
        _, swin_words = swin_predict(swin_detectors, pil_page, 1.0)

        draw_rects(image, gt_words, bnd_word_preds, BND_COLOR).save(
            BND_DIR / f"{page['stem']}.jpg", quality=88)
        draw_polygons(image, gt_words, swin_words).save(
            SWIN_DIR / f"{page['stem']}.jpg", quality=88)

        bar.set_postfix({"bnd": len(bnd_word_preds), "swin": len(swin_words)})

    elapsed = time.time() - started
    print(f"\n[done] {len(pages)} pages in {elapsed:.0f}s ({elapsed / max(len(pages), 1):.2f}s/page)")
    print(f"[out] {BND_DIR}")
    print(f"[out] {SWIN_DIR}")


if __name__ == "__main__":
    main()
