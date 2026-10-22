# Bangla Handwriting OCR Reports

Comprehensive collection of technical reports, analysis scripts, and visualization assets for Bengali/Bangla handwriting recognition and synthesis research.

## Directory Structure

```
bangla-ocr-reports/
├── README.md                    # This file
├── .gitignore                   # Git ignore rules
│
├── scripts/                     # Python scripts for analysis & visualization
│   ├── make_figures_*.py        # Report figure generation
│   ├── make_single_system_overlays*.py
│   ├── export_icdar_detections.py
│   ├── empty_space_analysis.py
│   └── [16 Python scripts total]
│
├── docs/                        # Documentation & reports
│   ├── *.md                     # Technical notes and documentation
│   ├── *.tex                    # LaTeX report sources
│   ├── 01_goal_method_data.md
│   ├── 03_evaluation.md
│   ├── 07_layout_v1.md
│   ├── 08_layout_v2.md
│   ├── 09_layout_spectral.md
│   └── [25 markdown + 9 LaTeX files]
│
├── assets/                      # Visualization & analysis results
│   ├── *.png                    # High-resolution figures
│   ├── *.jpg                    # Sample dataset images
│   ├── overlays/                # Detection/segmentation visualizations
│   ├── dataset_samples/         # Document samples
│   └── [98 image files]
│
└── reports/                     # Complete original report directories
    ├── main_report/             # Primary benchmark & segmentation reports
    │   ├── benchmark_report.pdf
    │   ├── benchmark_report.tex
    │   ├── benchmark_report_summary.pdf
    │   ├── assets/              # Figures and visualizations
    │   ├── exec_summary/        # Executive summary documents
    │   └── [make_figures*.py]   # Report generation scripts
    │
    ├── wordgen_reports/         # Word generation model reports
    │   └── final_v2/
    │       ├── report.tex       # Complete technical report
    │       ├── report.pdf
    │       ├── figures/         # Generated figures
    │       ├── notes/           # Detailed technical notes (*.md)
    │       └── [training logs & metrics]
    │
    └── layout_v2_report/        # Layout model v2 analysis
        ├── report/
        └── [figures & visualizations]
```

## Key Files

### Main Reports
- **benchmark_report.pdf**: Comprehensive benchmark results for Bangla HTR
- **wordgen_reports/final_v2/report.pdf**: Advanced word generation model documentation
- **reports/main_report/exec_summary/**: Executive summary with key metrics

### Scripts
- `scripts/make_figures_*.py`: Generate report visualizations
- `scripts/empty_space_analysis.py`: Analyze inter-word spacing
- `scripts/export_icdar_detections.py`: Export ICDAR-format predictions

### Documentation
- `docs/01_goal_method_data.md`: Research goals and dataset overview
- `docs/03_evaluation.md`: Evaluation metrics and methodology
- `docs/07_layout_v1.md`: Layout model v1 architecture
- `docs/08_layout_v2.md`: Layout model v2 improvements
- `docs/09_layout_spectral.md`: Spectral analysis of layout models

### Visualizations
- Line, word, grapheme detection overlays
- Dataset sample pages
- Benchmark result figures
- Model comparison charts

## Research Areas

1. **Bangla Handwriting Recognition (HTR)**
   - Polygon-based detection (lines, words, graphemes)
   - Multi-dataset evaluation (BN-DRISHTI, BanglaWriting)
   - Domain gap analysis

2. **Bangla Handwriting Synthesis**
   - Word generation with realistic layouts
   - Font and style learning
   - Text encoder models (CANINE, RoBERTa)

3. **Layout & Structure Analysis**
   - Line and word boundary detection
   - Grapheme-level segmentation
   - Spatial layout modeling

4. **Evaluation & Benchmarking**
   - Precision, Recall, F1 metrics
   - IoU-based polygon matching
   - Cross-dataset generalization

## File Statistics

| Type | Count |
|------|-------|
| PNG images | 85 |
| PDF reports | 52 |
| JPG images | 30 |
| Markdown docs | 25 |
| Python scripts | 16 |
| LaTeX files | 9 |
| Other (logs, aux) | 22 |
| **Total** | **239** |

## Data Organization

### Reports Subfolder (`reports/`)

#### main_report/
- Primary benchmark and segmentation results
- Executive summary with key findings
- High-resolution comparison figures
- Dataset visualizations

#### wordgen_reports/
- Advanced word generation model (final_v2)
- Technical documentation in Markdown
- Generated figures and training logs
- Model evaluation metrics

#### layout_v2_report/
- Layout model v2 analysis and results
- Figure generation scripts
- Comparison with v1

## Usage

### Generating Figures
```bash
cd scripts/
python make_figures_banglawriting.py
python make_single_system_overlays.py
```

### Building LaTeX Reports
```bash
cd docs/
pdflatex benchmark_report.tex
pdflatex reports/wordgen_reports/final_v2/report.tex
```

### Accessing Results
- View PDFs in `reports/*/` for full reports
- Check `assets/` for individual figures
- Read `docs/` for technical documentation

## Attribution

This repository contains comprehensive technical reports and analysis materials for Bangla handwriting recognition and synthesis research conducted by the team at CAIL (Center for Advanced Image and Language).

**Researchers**: Shyama Nayan  
**Email**: shyan@cail.du.ac.bd  
**Organization**: CAIL, University of Dhaka

## License

These technical reports and supporting materials are provided for research and educational purposes. Please refer to individual report files for specific licensing information.

## Related Repositories

- [Bangla-Handwriting-Segmentation](https://github.com/Bangla-Handwriting-Segmentation)
- Parent project: OCR research for Bengali handwriting documents

---

**Last Updated**: October 22, 2026  
**Repository Maintained By**: Claude Haiku (AI Agent)
