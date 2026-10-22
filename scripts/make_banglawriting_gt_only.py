"""Ground-truth-only overlays on BanglaWriting: just the human-annotated word boxes,
no model predictions, so the raw annotation looseness/tightness can be judged on its
own terms. No model inference needed -- this is pure data inspection.

Run:
    venv/bin/python report/make_banglawriting_gt_only.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw
from tqdm import tqdm

from bench_swin_bn_drishti_syn_hw import PROJECT_ROOT
from bench_swin_bn_drishti_banglawriting import load_banglawriting_pages
from pseudolabel.adapters.sources import decoded_rgb

GT_COLOR = (33, 150, 243)
OUT_DIR = PROJECT_ROOT / "bench_results" / "banglawriting_ground_truth_only"


def main() -> None:
    root = PROJECT_ROOT / "BanglaWriting" / "converted" / "converted"
    pages = load_banglawriting_pages(root)
    print(f"[data] {len(pages)} BanglaWriting pages, {sum(len(p['boxes']) for p in pages)} GT word boxes")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for page in tqdm(pages, desc="pages", dynamic_ncols=True):
        image = decoded_rgb(page["image_path"])
        picture = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(picture)
        for box in page["boxes"]:
            draw.rectangle(box, outline=GT_COLOR, width=3)
        picture.save(OUT_DIR / f"{page['stem']}.jpg", quality=88)

    print(f"[out] {OUT_DIR}")


if __name__ == "__main__":
    main()
