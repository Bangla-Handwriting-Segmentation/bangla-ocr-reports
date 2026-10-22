"""Direct, model-independent proof of why BN-DRISHTI's rectangles score higher mAP@75
than Swin's polygons against a rectangle ground truth.

For every ground-truth box matched to a prediction from either system, this script:
  1. Crops the *source page image* (not either model's output) at the GT box and
     Otsu-thresholds it to separate true ink pixels from background pixels. This is
     an independent measurement of "how much of the GT rectangle is actually text"
     vs. "how much is empty margin" -- it does not depend on any model's prediction.
  2. Rasterizes the matched prediction (BN-DRISHTI's rectangle, or Swin's polygon)
     into the same crop frame and measures:
       - ink_recall: fraction of true ink pixels the prediction covers
       - background_coverage: fraction of true *background* (non-ink) pixels within
         the GT box that the prediction *also* claims as its own area
  If BN-DRISHTI's rectangle and Swin's polygon achieve similar ink_recall (comparable
  detection quality) but very different background_coverage (BN-DRISHTI much higher),
  that is direct, per-pixel proof that BN-DRISHTI's mAP@75 edge comes from also
  claiming empty margin the ground truth rectangle happens to include -- not from
  more accurate text localization.

Run:
    venv/bin/python report/empty_space_analysis.py --limit 60
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image

from bench_bn_drishti_syn_hw import (
    load_line_ground_truth, load_word_ground_truth, predict_lines, predict_words, val_split_images,
)
from bench_swin_bn_drishti_syn_hw import PROJECT_ROOT, load_swin_detectors, swin_predict
from pseudolabel.adapters.bn_drishti import load_line_model, load_word_model
from pseudolabel.adapters.sources import decoded_rgb
from pseudolabel.domain.geometry import clip_rect_xyxy, rect_iou

MIN_PIXELS = 40  # skip degenerate crops (too small to threshold meaningfully)


def best_match_index(pred_box, gt_boxes: list[tuple[float, float, float, float]], iou_min: float = 0.3):
    best_j, best_iou = -1, 0.0
    for j, gt in enumerate(gt_boxes):
        iou = rect_iou(pred_box, gt)
        if iou > best_iou:
            best_iou, best_j = iou, j
    return (best_j, best_iou) if best_iou >= iou_min else (-1, 0.0)


def shape_bbox(shape) -> tuple[float, float, float, float]:
    array = np.asarray(shape, dtype=np.float64)
    if array.shape == (4,):
        return tuple(array)
    xs, ys = array[:, 0], array[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def rasterize_in_crop(shape, crop_x1: int, crop_y1: int, crop_w: int, crop_h: int) -> np.ndarray:
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    array = np.asarray(shape, dtype=np.float64)
    if array.shape == (4,):
        x1, y1, x2, y2 = array
        sx1, sy1 = int(round(x1 - crop_x1)), int(round(y1 - crop_y1))
        sx2, sy2 = int(round(x2 - crop_x1)), int(round(y2 - crop_y1))
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(crop_w, sx2), min(crop_h, sy2)
        if sx2 > sx1 and sy2 > sy1:
            mask[sy1:sy2, sx1:sx2] = 1
        return mask
    points = np.round(array - np.array([crop_x1, crop_y1])).astype(np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask


def ink_mask_for_crop(gray_crop: np.ndarray) -> np.ndarray | None:
    if gray_crop.size < MIN_PIXELS:
        return None
    if gray_crop.max() == gray_crop.min():
        return None  # uniform crop, thresholding undefined
    _, binary = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return (binary > 0)


def analyze_task(gray: np.ndarray, width: int, height: int, gt_boxes: list[tuple[float, float, float, float]],
                 preds: list, is_polygon: bool, examples: list | None, example_tag: str) -> list[dict]:
    records = []
    for gt_box in gt_boxes:
        clipped = clip_rect_xyxy(gt_box, width, height)
        if clipped is None:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in clipped)
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 6 or crop_h < 6:
            continue
        gray_crop = gray[y1:y2, x1:x2]
        ink_mask = ink_mask_for_crop(gray_crop)
        if ink_mask is None:
            continue
        ink_count = int(ink_mask.sum())
        bg_count = ink_mask.size - ink_count
        if ink_count < MIN_PIXELS or bg_count < MIN_PIXELS:
            continue

        best_j, best_iou = -1, 0.0
        for idx, pred in enumerate(preds):
            shape = pred[0] if not is_polygon else pred["polygon"]
            box = shape_bbox(shape)
            iou = rect_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_j = iou, idx
        if best_j < 0:
            continue
        shape = preds[best_j][0] if not is_polygon else preds[best_j]["polygon"]

        pred_mask = rasterize_in_crop(shape, x1, y1, crop_w, crop_h)
        ink_recall = float((pred_mask & ink_mask).sum()) / ink_count
        bg_coverage = float((pred_mask & ~ink_mask).sum()) / bg_count

        records.append({
            "gt_area_px": crop_w * crop_h,
            "ink_fraction": ink_count / (crop_w * crop_h),
            "background_fraction": bg_count / (crop_w * crop_h),
            "match_iou": best_iou,
            "ink_recall": ink_recall,
            "background_coverage": bg_coverage,
        })

        if examples is not None and sum(1 for e in examples if e["tag"] == example_tag) < 2:
            examples.append({
                "tag": example_tag,
                "crop_rgb": None,  # filled by caller (needs color image)
                "gray_crop": gray_crop,
                "ink_mask": ink_mask,
                "pred_mask": pred_mask,
                "gt_box_local": (0, 0, crop_w, crop_h),
                "is_polygon": is_polygon,
                "shape_local": (np.asarray(shape, dtype=np.float64) - np.array([x1, y1])
                                if is_polygon else
                                np.array(shape, dtype=np.float64) - np.array([x1, y1, x1, y1])),
                "crop_origin": (x1, y1),
                "crop_size": (crop_w, crop_h),
            })
    return records


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    return {
        "n": len(records),
        "mean_ink_fraction": round(stats.mean(r["ink_fraction"] for r in records), 4),
        "mean_background_fraction": round(stats.mean(r["background_fraction"] for r in records), 4),
        "mean_ink_recall": round(stats.mean(r["ink_recall"] for r in records), 4),
        "mean_background_coverage": round(stats.mean(r["background_coverage"] for r in records), 4),
        "median_background_coverage": round(stats.median(r["background_coverage"] for r in records), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "bench_results" / "empty_space_analysis.json")
    parser.add_argument("--examples-out", type=Path, default=PROJECT_ROOT / "report" / "assets" / "examples")
    args = parser.parse_args()
    device = torch.device(args.device)

    val_images = val_split_images(PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_lines" / "best.pth")[: args.limit]
    wanted = set(val_images)
    line_gt = load_line_ground_truth(PROJECT_ROOT / "syn-hw" / "file_segment.json", wanted)
    word_gt = load_word_ground_truth(PROJECT_ROOT / "syn-hw" / "word_segment.json",
                                     PROJECT_ROOT / "syn-hw" / "word_segment.json.pageindex.json", wanted)
    usable = [img for img in val_images if img in line_gt and img in word_gt]
    print(f"[data] {len(usable)} usable pages")

    bnd_line_model = load_line_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "line_model_best.pt", str(device))
    bnd_word_model = load_word_model(PROJECT_ROOT / "BN-DRISHTI" / "yolov5",
                                     PROJECT_ROOT / "BN-DRISHTI" / "models" / "word_model_best.pt", str(device))
    swin_detectors = load_swin_detectors(
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_lines" / "best.pth",
        PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_words" / "best.pth",
        device, score=0.5, mask_threshold=0.5, batch=16)

    all_records = {"line": {"bnd": [], "swin": []}, "word": {"bnd": [], "swin": []}}
    examples: list = []

    from tqdm import tqdm
    for image_rel in tqdm(usable, desc="pages", dynamic_ncols=True):
        image = decoded_rgb(PROJECT_ROOT / "syn-hw" / image_rel)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        height, width = image.shape[:2]
        pil_page = Image.fromarray(image)

        bnd_line_preds = predict_lines(bnd_line_model, image)
        bnd_word_preds = predict_words(bnd_word_model, image, [box for box, _ in bnd_line_preds])
        swin_lines, swin_words = swin_predict(swin_detectors, pil_page, pad_scale=1.0)

        all_records["line"]["bnd"] += analyze_task(gray, width, height, line_gt[image_rel]["boxes"],
                                                    bnd_line_preds, False, examples, "line_bnd")
        all_records["line"]["swin"] += analyze_task(gray, width, height, line_gt[image_rel]["boxes"],
                                                     swin_lines, True, examples, "line_swin")
        all_records["word"]["bnd"] += analyze_task(gray, width, height, word_gt[image_rel]["boxes"],
                                                    bnd_word_preds, False, examples, "word_bnd")
        all_records["word"]["swin"] += analyze_task(gray, width, height, word_gt[image_rel]["boxes"],
                                                     swin_words, True, examples, "word_swin")

        for ex in examples:
            if ex.get("crop_rgb") is None:
                x1, y1 = ex["crop_origin"]
                w, h = ex["crop_size"]
                ex["crop_rgb"] = image[y1:y1 + h, x1:x1 + w].copy()

    summary = {
        "pages": len(usable),
        "line": {"bnd": summarize(all_records["line"]["bnd"]), "swin": summarize(all_records["line"]["swin"])},
        "word": {"bnd": summarize(all_records["word"]["bnd"]), "swin": summarize(all_records["word"]["swin"])},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))
    print(f"[out] {args.out}")

    args.examples_out.mkdir(parents=True, exist_ok=True)
    np.save(args.examples_out / "examples.npy", np.array(examples, dtype=object), allow_pickle=True)
    print(f"[examples] {len(examples)} saved to {args.examples_out / 'examples.npy'}")


if __name__ == "__main__":
    main()
