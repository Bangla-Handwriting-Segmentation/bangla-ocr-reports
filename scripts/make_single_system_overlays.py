"""Full-page overlays showing each system's predictions in isolation (GT + one
system only), for the report's expanded qualitative section."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image, ImageDraw

from bench_bn_drishti_syn_hw import (
    load_line_ground_truth, load_word_ground_truth, predict_lines, predict_words, val_split_images,
)
from bench_swin_bn_drishti_syn_hw import PROJECT_ROOT, load_swin_detectors, swin_predict
from pseudolabel.adapters.bn_drishti import load_line_model, load_word_model
from pseudolabel.adapters.sources import decoded_rgb

GT_COLOR = (33, 150, 243)
BND_COLOR = (244, 67, 54)
SWIN_COLOR = (76, 175, 80)

PAGES = ["group_000/shard_0000/images/page_00014.jpg", "group_000/shard_0000/images/page_00099.jpg"]
OUT_DIR = PROJECT_ROOT / "report" / "assets" / "overlays_single"


def draw_rects(image, gt_boxes, pred_boxes, pred_color):
    picture = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(picture)
    for box in gt_boxes:
        draw.rectangle(box, outline=GT_COLOR, width=4)
    for box, _ in pred_boxes:
        draw.rectangle(box, outline=pred_color, width=3)
    return picture


def draw_polygons(image, gt_boxes, instances):
    picture = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(picture)
    for box in gt_boxes:
        draw.rectangle(box, outline=GT_COLOR, width=4)
    for inst in instances:
        points = [(float(x), float(y)) for x, y in inst["polygon"]]
        draw.line(points + [points[0]], fill=SWIN_COLOR, width=3)
    return picture


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_images = val_split_images(PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_lines" / "best.pth")
    wanted = set(PAGES) | set(val_images[:200])  # ensure our two target pages resolve even if not literally in val_ids naming

    line_gt = load_line_ground_truth(PROJECT_ROOT / "syn-hw" / "file_segment.json", set(PAGES))
    word_gt = load_word_ground_truth(PROJECT_ROOT / "syn-hw" / "word_segment.json",
                                     PROJECT_ROOT / "syn-hw" / "word_segment.json.pageindex.json", set(PAGES))

    bnd_line_model = load_line_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "line_model_best.pt", str(device))
    bnd_word_model = load_word_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "word_model_best.pt", str(device))
    swin_detectors = load_swin_detectors(
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_lines" / "best.pth",
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_words" / "best.pth",
        device, score=0.5, mask_threshold=0.5, batch=16)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for page_rel in PAGES:
        if page_rel not in line_gt or page_rel not in word_gt:
            print(f"[skip] no GT for {page_rel}")
            continue
        stem = Path(page_rel).stem
        image = decoded_rgb(PROJECT_ROOT / "syn-hw" / page_rel)
        pil_page = Image.fromarray(image)

        bnd_line_preds = predict_lines(bnd_line_model, image)
        bnd_word_preds = predict_words(bnd_word_model, image, [box for box, _ in bnd_line_preds])
        swin_lines, swin_words = swin_predict(swin_detectors, pil_page, pad_scale=1.0)

        draw_rects(image, line_gt[page_rel]["boxes"], bnd_line_preds, BND_COLOR).save(
            OUT_DIR / f"line_bnd_only_{stem}.jpg", quality=88)
        draw_polygons(image, line_gt[page_rel]["boxes"], swin_lines).save(
            OUT_DIR / f"line_swin_only_{stem}.jpg", quality=88)
        draw_rects(image, word_gt[page_rel]["boxes"], bnd_word_preds, BND_COLOR).save(
            OUT_DIR / f"word_bnd_only_{stem}.jpg", quality=88)
        draw_polygons(image, word_gt[page_rel]["boxes"], swin_words).save(
            OUT_DIR / f"word_swin_only_{stem}.jpg", quality=88)
        print(f"[done] {stem}: {len(bnd_line_preds)} bnd lines, {len(swin_lines)} swin lines, "
              f"{len(bnd_word_preds)} bnd words, {len(swin_words)} swin words")


if __name__ == "__main__":
    main()
