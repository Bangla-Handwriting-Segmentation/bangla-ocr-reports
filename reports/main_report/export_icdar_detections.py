"""Export raw per-instance WORD-level detections (BN-DRISHTI, Swin) and ground truth,
for all three benchmark corpora (syn-hw, BN-HTRd, BanglaWriting), to ICDAR-style
per-image txt files -- the input format the official CLEval (github.com/clovaai/CLEval)
and TIoU-metric (github.com/Yuliang-Liu/TIoU-metric) tools expect.

No raw per-instance prediction cache existed anywhere in bench_results/ before this --
only aggregate metric JSONs. This script re-runs inference once per dataset/system and
persists every instance, in a format directly consumable by the official tools, so a
CLEval and a TIoU evaluation pass can be run without re-deriving this format.

Format (confirmed by reading the official repos' source, not just their README):
  - Points are listed clockwise, arbitrary even count >= 8 (quad or polygon), then an
    optional confidence, then an optional transcription -- all comma-separated. Order is
    always: x1,y1,...,xn,yn[,confidence][,transcription]. Both tools infer point count
    from the comma count, so predictions (no transcription) and ground truth (with
    transcription, when known) can use different point counts per line freely.
  - CLEval reads gt/pred as *.zip of per-image .txt files; the archive-internal filename
    (after stripping a leading "gt_"/"res_" and the extension) is the join key between
    gt.zip and result.zip -- so gt files here are named ``gt_<id>.txt`` and prediction
    files ``res_<id>.txt`` for the same ``<id>``.
  - TIoU-metric (official, Yuliang-Liu/TIoU-metric, Python 2 code -- will need porting,
    see caveats below) additionally requires the numeric id in the filename to be
    *purely digits* (its ``GT_SAMPLE_NAME_2_ID``/``DET_SAMPLE_NAME_2_ID`` regexes are
    ``([0-9]+).txt`` / ``img([0-9]+).txt`` depending on preset) -- hence the zero-padded
    sequential ids used here instead of each dataset's native (non-numeric) page stems.
    A ``stems.json`` sidecar in each dataset folder maps id -> original page stem/path.
  - CLEval's own don't-care convention is the literal transcription string ``"###"``.
    We do not use it here: none of our GT boxes are meant to be excluded from scoring.

Ground-truth transcription availability (checked directly in each dataset's raw source,
not assumed):
  - BanglaWriting: real per-word transcription in each LabelMe JSON's ``shapes[i].label``.
  - syn-hw: real per-word transcription in ``word_segment.json``'s ``text`` field (this
    export re-reads word_segment.json directly for ``text`` and the 50-point ink-tracing
    ``polygon_norm``, both of which ``bench_bn_drishti_syn_hw.load_word_ground_truth``
    discards -- so this is a strictly richer/tighter GT shape than the axis-aligned box
    used elsewhere in the project, which matters for TIoU's tightness scoring).
  - BN-HTRd: NO transcription anywhere in its word-level source (Pascal-VOC XML objects
    are all generically named "word", not the word's text) and no polygon, only an
    axis-aligned box. Exported GT transcription is left as an empty string for BN-HTRd
    with this fact documented here; CLEval's character-count scoring needs a non-empty
    string per box to be meaningful (its own code sums ``len(gt_box.transcription)`` as
    the recall denominator even in detection-only mode), so a literal CLEval run on
    BN-HTRd as exported will degenerate to recall=0.0 dataset-wide. Whether to run CLEval
    on BN-HTRd at all -- and if so, with what placeholder transcription-length heuristic
    -- is a scoring-methodology decision left to whoever runs CLEval, not baked into this
    raw export.

TIoU-metric caveats (does not block export, but blocks running the tool as-is):
  - The cloned repo is Python 2 only (``from StringIO import StringIO``, bare
    ``except Exception, e:``, ``xrange`` -- confirmed via ``ast.parse`` failing under
    Python 3.12). A community Python 3 port exists (PkuDavidGuan/TIoU-metric-python3,
    linked from the official README) and is likely far less work than patching the
    original by hand.
  - ``shapely`` (imported by ``rrc_evaluation_funcs.py``) is not installed in this
    project's venv.
  - ``curved-tiou/script.py``'s ``default_evaluation_params()`` is hardcoded to only
    recognize ``-g=ctw1500-gt.zip`` or ``-g=total-text-gt.zip`` by literal basename
    string match (not a generic path) -- any other gt zip name silently falls through
    with no return value. Our exported gt.zip must be passed under one of those two
    exact basenames (e.g. a symlink/copy), or the function needs a new preset branch.
  - The curved (Total-Text-style) preset also has a real Python-2/3 numeric bug at
    ``polygon_from_points``: ``resBoxes[0, inp/2]`` uses ``/`` where NumPy indexing in
    Python 3 needs ``//`` (float index otherwise raises).

Run:
    venv/bin/python report/export_icdar_detections.py                      # all 3 datasets
    venv/bin/python report/export_icdar_detections.py --datasets banglawriting
    venv/bin/python report/export_icdar_detections.py --limit 20           # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from tqdm import tqdm

from bench_bn_drishti_syn_hw import predict_lines, predict_words, val_split_images
from bench_swin_bn_drishti_syn_hw import PROJECT_ROOT, load_swin_detectors, swin_predict, to_scored_boxes
from bench_swin_bn_drishti_banglawriting import load_banglawriting_pages
from bench_swin_bn_drishti_bnhtr_real import BN_HTR_ROOT, LINES_DIR, load_pages as load_bnhtr_pages, word_boxes_for_page
from pseudolabel.adapters.bn_drishti import load_line_model, load_word_model
from pseudolabel.adapters.sources import decoded_rgb

OUT_ROOT = PROJECT_ROOT / "bench_results" / "icdar_format"


# ========================== quad / polygon helpers ==========================


def box_to_quad(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def ensure_clockwise(points: list[float]) -> list[float]:
    """CLEval/TIoU both validate clockwise point order (image coords, y-down) via the
    same edge-sum test: sum((x_{i+1}-x_i)*(y_{i+1}+y_i)) must be <= 0. Reverse if not."""
    pts = list(zip(points[0::2], points[1::2]))
    edge_sum = sum((pts[(i + 1) % len(pts)][0] - pts[i][0]) * (pts[(i + 1) % len(pts)][1] + pts[i][1])
                   for i in range(len(pts)))
    if edge_sum > 0:
        pts = pts[::-1]
    out: list[float] = []
    for x, y in pts:
        out.extend((x, y))
    return out


def fmt_points(points: list[float]) -> str:
    return ",".join(f"{v:.2f}" for v in points)


def quote_if_needed(text: str) -> str:
    if "," in text or '"' in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


# ========================== per-dataset page iterators ==========================
# Each yields (page_id: str, image: np.ndarray, gt_items: list[(points, transcription)])
# gt points are already pixel-space, clockwise-ordered.


def iter_syn_hw(syn_hw_root: Path, swin_line_ckpt: Path, limit: int | None):
    val_images = val_split_images(swin_line_ckpt)
    if limit:
        val_images = val_images[:limit]
    wanted = set(val_images)

    pageindex = json.loads((syn_hw_root / "word_segment.json.pageindex.json").read_text(encoding="utf-8"))
    entries = [e for e in pageindex if e["image"] in wanted]
    seg_path = syn_hw_root / "word_segment.json"
    with seg_path.open("rb") as handle:
        for entry in entries:
            handle.seek(entry["offset"])
            page = json.loads(handle.read(entry["length"]))
            width, height = float(page["width"]), float(page["height"])
            items = []
            for word in page["lines"]:
                text = word.get("text", "")
                poly_norm = word.get("polygon_norm")
                if poly_norm and len(poly_norm) >= 4:
                    flat = []
                    for px, py in poly_norm:
                        flat.extend((px * width, py * height))
                else:
                    x1n, y1n, x2n, y2n = word["bbox_norm"]
                    flat = box_to_quad((x1n * width, y1n * height, x2n * width, y2n * height))
                items.append((ensure_clockwise(flat), text))
            image_path = syn_hw_root / entry["image"]
            yield entry["image"].replace("/", "__"), image_path, items


# ========================== writer ==========================


class IcdarWriter:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.gt_dir = dataset_dir / "gt"
        self.pred_dirs: dict[str, Path] = {}
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        self.stems: dict[str, str] = {}
        self.counts = {"pages": 0, "gt_instances": 0}
        self.pred_counts: dict[str, int] = {}

    def pred_dir(self, system: str) -> Path:
        if system not in self.pred_dirs:
            d = self.dataset_dir / f"pred_{system}"
            d.mkdir(parents=True, exist_ok=True)
            self.pred_dirs[system] = d
            self.pred_counts[system] = 0
        return self.pred_dirs[system]

    def write_page(self, seq_id: str, stem: str, gt_items: list[tuple[list[float], str]],
                   preds: dict[str, list[tuple[list[float], float]]]):
        self.stems[seq_id] = stem
        self.counts["pages"] += 1
        gt_lines = []
        for points, text in gt_items:
            gt_lines.append(f"{fmt_points(points)},{quote_if_needed(text)}")
        (self.gt_dir / f"gt_{seq_id}.txt").write_text("\n".join(gt_lines), encoding="utf-8")
        self.counts["gt_instances"] += len(gt_items)

        for system, instances in preds.items():
            pred_dir = self.pred_dir(system)
            lines = [f"{fmt_points(points)},{score:.4f}" for points, score in instances]
            (pred_dir / f"res_{seq_id}.txt").write_text("\n".join(lines), encoding="utf-8")
            self.pred_counts[system] = self.pred_counts.get(system, 0) + len(instances)

    def finalize(self):
        (self.dataset_dir / "stems.json").write_text(json.dumps(self.stems, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
        _zip_dir(self.gt_dir, self.dataset_dir / "gt.zip")
        for system, d in self.pred_dirs.items():
            _zip_dir(d, self.dataset_dir / f"pred_{system}.zip")


def _zip_dir(src_dir: Path, out_zip: Path):
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(src_dir.glob("*.txt")):
            zf.write(f, arcname=f.name)


# ========================== main ==========================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", nargs="+", choices=["syn_hw", "bn_htrd", "banglawriting"],
                        default=["syn_hw", "bn_htrd", "banglawriting"])
    parser.add_argument("--syn-hw-root", type=Path, default=PROJECT_ROOT / "syn-hw")
    parser.add_argument("--banglawriting-root", type=Path,
                        default=PROJECT_ROOT / "BanglaWriting" / "converted" / "converted")
    parser.add_argument("--bnhtrd-manifest", type=Path, default=BN_HTR_ROOT / "split_manifest.json")
    parser.add_argument("--bnhtrd-split", default="test")
    parser.add_argument("--swin-line-ckpt", type=Path,
                        default=PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_lines" / "best.pth")
    parser.add_argument("--swin-word-ckpt", type=Path,
                        default=PROJECT_ROOT / "SWIN_lines_words_grapheme" / "swin_mask_rcnn_words" / "best.pth")
    parser.add_argument("--swin-score", type=float, default=0.5)
    parser.add_argument("--swin-mask-threshold", type=float, default=0.5)
    parser.add_argument("--swin-batch", type=int, default=16)
    parser.add_argument("--pad-scale", type=float, default=1.0)
    parser.add_argument("--bn-drishti-repo", type=Path, default=PROJECT_ROOT / "BN-DRISHTI" / "yolov5")
    parser.add_argument("--bn-line-weights", type=Path, default=PROJECT_ROOT / "BN-DRISHTI" / "models" / "line_model_best.pt")
    parser.add_argument("--bn-word-weights", type=Path, default=PROJECT_ROOT / "BN-DRISHTI" / "models" / "word_model_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="cap pages per dataset (smoke test)")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    device = torch.device(args.device)
    bnd_line_model = load_line_model(args.bn_drishti_repo, args.bn_line_weights, str(device))
    bnd_word_model = load_word_model(args.bn_drishti_repo, args.bn_word_weights, str(device))
    print(f"[model] bn-drishti line={bnd_line_model.weights_hash[:10]} word={bnd_word_model.weights_hash[:10]}")
    swin_detectors = load_swin_detectors(args.swin_line_ckpt, args.swin_word_ckpt, device,
                                         args.swin_score, args.swin_mask_threshold, args.swin_batch)
    print(f"[model] swin loaded, device={device}")

    def run_inference(image):
        pil_page = Image.fromarray(image)
        bnd_line_preds = predict_lines(bnd_line_model, image)
        bnd_word_preds = predict_words(bnd_word_model, image, [box for box, _ in bnd_line_preds])
        _, swin_words = swin_predict(swin_detectors, pil_page, args.pad_scale)
        bnd_out = [(ensure_clockwise(box_to_quad(box)), score) for box, score in bnd_word_preds]
        swin_out = [(ensure_clockwise([c for xy in inst["polygon"] for c in (float(xy[0]), float(xy[1]))]), inst["score"])
                   for inst in swin_words]
        return {"bn_drishti": bnd_out, "swin": swin_out}

    summary = {}

    if "banglawriting" in args.datasets:
        print("\n=== BanglaWriting (word) ===")
        writer = IcdarWriter(args.out_root / "banglawriting")
        started = time.time()
        pages = list(load_banglawriting_pages(args.banglawriting_root))
        if args.limit:
            pages = pages[: args.limit]
        for i, page in enumerate(tqdm(pages, desc="banglawriting", dynamic_ncols=True)):
            image = decoded_rgb(page["image_path"])
            data = json.loads(page["image_path"].with_suffix(".json").read_text(encoding="utf-8"))
            items = []
            for shape in data["shapes"]:
                (x1, y1), (x2, y2) = shape["points"]
                x1, x2 = sorted((float(x1), float(x2)))
                y1, y2 = sorted((float(y1), float(y2)))
                if x2 > x1 and y2 > y1:
                    items.append((ensure_clockwise(box_to_quad((x1, y1, x2, y2))), shape.get("label", "")))
            preds = run_inference(image)
            writer.write_page(f"{i:04d}", page["stem"], items, preds)
        writer.finalize()
        summary["banglawriting"] = {"pages": writer.counts["pages"], "gt_instances": writer.counts["gt_instances"],
                                    "pred_instances": writer.pred_counts, "elapsed_s": round(time.time() - started, 1)}

    if "syn_hw" in args.datasets:
        print("\n=== syn-hw (word) ===")
        writer = IcdarWriter(args.out_root / "syn_hw")
        started = time.time()
        it = list(iter_syn_hw(args.syn_hw_root, args.swin_line_ckpt, args.limit))
        for i, (seq_stem, image_path, items) in enumerate(tqdm(it, desc="syn_hw", dynamic_ncols=True)):
            image = decoded_rgb(image_path)
            preds = run_inference(image)
            writer.write_page(f"{i:05d}", seq_stem, items, preds)
        writer.finalize()
        summary["syn_hw"] = {"pages": writer.counts["pages"], "gt_instances": writer.counts["gt_instances"],
                             "pred_instances": writer.pred_counts, "elapsed_s": round(time.time() - started, 1)}

    if "bn_htrd" in args.datasets:
        print("\n=== BN-HTRd (word) ===")
        writer = IcdarWriter(args.out_root / "bn_htrd")
        started = time.time()
        pages = load_bnhtr_pages(args.bnhtrd_manifest, args.bnhtrd_split)
        if args.limit:
            pages = pages[: args.limit]
        for i, page in enumerate(tqdm(pages, desc="bn_htrd", dynamic_ncols=True)):
            writer_id, page_id = page["writer_id"], page["page_id"]
            line_boxes = [tuple(box) for box in page["boxes"]]
            words = word_boxes_for_page(writer_id, page_id, line_boxes)
            image = decoded_rgb(LINES_DIR / page["image"])
            items = [(ensure_clockwise(box_to_quad(box)), "") for box in words]
            preds = run_inference(image)
            writer.write_page(f"{i:04d}", f"{writer_id}__{page_id}", items, preds)
        writer.finalize()
        summary["bn_htrd"] = {"pages": writer.counts["pages"], "gt_instances": writer.counts["gt_instances"],
                              "pred_instances": writer.pred_counts, "elapsed_s": round(time.time() - started, 1)}

    (args.out_root / "export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n[done]")
    print(json.dumps(summary, indent=2))
    print(f"[out] {args.out_root}")


if __name__ == "__main__":
    main()
