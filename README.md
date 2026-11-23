# ChromaGrade

**Explainable deep learning for three-level anemia severity grading from peripheral blood smear images**

ChromaGrade grades red-blood-cell (RBC) hypochromia as **Normochromic**, **Mild Hypochromic** or **Severely Hypochromic**. It does not stop at a binary "hypochromic / not hypochromic" call. The framework has two complementary branches:

| Branch | Level | Model | What it does |
|---|---|---|---|
| **Image-level** | Whole smear field (bag of 12 patches) | ConvNeXt-Tiny + gated-attention MIL + CBC fusion | Predicts one severity grade per image and shows which patches drove the decision |
| **Cell-level** | Individual RBCs | YOLO26s detector | Detects every RBC and assigns it a pallor-severity class |

Both branches come with an **explainability study** (attention maps, Grad-CAM++, SHAP, EigenCAM, D-RISE, GradientSHAP), which includes a quantitative comparison of the XAI methods.

> ⚠️ **Research code only — not a medical device.** The outputs must not be used for clinical diagnosis.

---

## Repository structure

```
ChromaGrade/
├── notebooks/
│   ├── 01_convnext_attention_mil.ipynb        # Image-level MIL classifier (+ CBC fusion, k-fold CV, XAI)
│   ├── 02_yolo26_rbc_severity_detector.ipynb  # Per-cell YOLO26 detector on pseudo-labelled RBCs
│   └── 03_yolo26_explainability.ipynb         # XAI comparison study for the YOLO26 branch
├── figures/                                   # Key figures for this README (optional)
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## Pipeline

```
Blood smear images + CBC table
            │
            ▼
 ┌──────────────────────────────┐
 │ 01  ConvNeXt + Attention-MIL │──► patch_pallor_labels.csv, bag-level grade
 └──────────────────────────────┘
            │ (pallor labels)
            ▼
 ┌──────────────────────────────┐
 │ 02  YOLO26 per-cell detector │──► best.pt (fine-tuned RBC detector)
 └──────────────────────────────┘
            │ (weights)
            ▼
 ┌──────────────────────────────┐
 │ 03  Explainability study     │──► XAI scores, saliency figures
 └──────────────────────────────┘
```

Run the notebooks **in order**. Notebook 02 needs the `patch_pallor_labels.csv` that notebook 01 produces, and notebook 03 needs the YOLO26 weights from notebook 02.

---

## Notebooks

### 01 — ConvNeXt + Gated-Attention MIL (`01_convnext_attention_mil.ipynb`)
*Platform: Google Colab (GPU)*

- Each image is split into **12 patches**. The patches form one **MIL bag**.
- A **central-pallor ratio** is measured for each patch and turned into patch pallor labels. Each bag takes the **median-severity** label of its patches, so a single outlier patch cannot set the label.
- A **ConvNeXt-Tiny** (ImageNet-pretrained) encoder feeds **gated attention pooling** (Ilse et al., 2018). The pooled representation is fused with **CBC tabular features**.
- Training uses a class-balanced loss, warm-up followed by a cosine LR schedule, early stopping and gradient accumulation.
- Evaluation covers a held-out test set and **stratified 5-fold cross-validation** with pooled out-of-fold (OOF) metrics. The notebook reports balanced accuracy, macro-F1, ROC and PR curves, and bootstrap CIs.
- Explainability: per-patch **attention weights**, **Grad-CAM++** and an optional **SHAP** comparison.

### 02 — YOLO26 RBC Severity Detector (`02_yolo26_rbc_severity_detector.ipynb`)
*Platform: Kaggle (GPU T4)*

- Individual RBCs are segmented with Otsu thresholding, a distance transform and watershed splitting, then filtered by area and circularity.
- The pallor rule assigns each cell a **pseudo-label**, and the labels are exported as a YOLO-format dataset built from every field image.
- **YOLO26s** (Ultralytics) is fine-tuned for 50 epochs.
- Validation produces a confusion matrix and PR/F1 curves, plus **EigenCAM** and **D-RISE** maps with a localisation hit-rate check.

### 03 — YOLO26 Explainability Study (`03_yolo26_explainability.ipynb`)
*Platform: Kaggle (GPU)*

- Methods compared: **EigenCAM** (stride-4, stride-8, multi-layer), **Pseudo-EigenCAM** (gated to each cell), **D-RISE** and **GradientSHAP**.
- Quantitative ranking by **deletion AUC**, **pointing game** and **energy-in-box**, scored against a smoothed random baseline.
- **Detection-improvement test**: the notebook checks whether EigenCAM rescoring or EigenCAM-guided contrast enhancement improves YOLO26 mAP, with bootstrap CIs.

---

## Results (summary)

**Image-level ConvNeXt-MIL**

| Evaluation | Accuracy |
|---|---|
| Pooled 5-fold OOF | ~82.4 % |
| Single held-out test split (n = 75 bags) | ~70.7 % |

**Cell-level YOLO26s** (validation set, 1,200 fields, ~107k cells)

| Class | AP@0.5 |
|---|---|
| Normochromic | 0.943 |
| Mild Hypochromic | 0.883 |
| Severely Hypochromic | 0.929 |
| **All (mAP@0.5 / mAP@0.5:0.95)** | **0.918 / 0.862** |

**XAI comparison (YOLO26 branch, 36 detections)**: **GradientSHAP** and **D-RISE** ranked best (deletion AUC 0.120 / 0.101, pointing game 0.97 / 0.89). EigenCAM variants localised individual cells poorly. Neither EigenCAM rescoring nor EigenCAM-guided enhancement improved detection mAP.

> **Caveats.** The YOLO26 metrics measure agreement with **rule-based pseudo-labels** from the central-pallor ratio, not independent expert annotation. The Severely Hypochromic class is small (14 bags in total), so its single-split metrics have high variance. Report the k-fold numbers.

---

## Dataset

The experiments use 500 peripheral blood smear images with matching CBC and morphology tables. Each image comes in three versions (`Original_Images/`, `RGB_Segmented/`, `Binary_Segmented/`) and is split into 12 patches named `IIII_PP_a.png`.

The data are taken from the **AneRBC** dataset. Download it from its original source and cite the dataset paper. **The dataset is not redistributed in this repository.**

Expected layout once unzipped:

```
anemia_500/
├── Original_Images/
├── RGB_Segmented/
├── Binary_Segmented/
├── cbc.csv
└── morphology.csv
```

---

## Getting started

**Colab / Kaggle (recommended).** Open a notebook, enable a GPU runtime, then set the data path or Drive link in the configuration cell near the top. Each notebook installs its own extra packages in its first cell.

**Local**

```bash
git clone https://github.com/<2946-cpu!>/ChromaGrade.git
cd ChromaGrade
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Paths in the notebooks point to `/content/...` (Colab) or `/kaggle/...` (Kaggle). Change them to your local folders.

---

## Citation

If you use this code, please cite it (see `CITATION.cff`). The paper is in preparation.

```bibtex
@software{afzal_chromagrade_2026,
  author = {Afzal, Ayesha},
  title  = {ChromaGrade: Explainable Deep Learning for Three-Level Anemia Severity Grading},
  year   = {2026},
  url    = {https://github.com/<your-username>/ChromaGrade}
}
```

## Acknowledgements

This work was carried out at **NUST School of Interdisciplinary Engineering & Sciences (SINES)**, Islamabad, Pakistan, under the supervision of **Dr. Muhammad Shazad**.

Key references:
- Ilse, Tomczak & Welling (2018). *Attention-based Deep Multiple Instance Learning.* ICML.
- Liu et al. (2022). *A ConvNet for the 2020s* (ConvNeXt). CVPR.
- Petsiuk et al. (2021). *Black-box Explanation of Object Detectors via Saliency Maps* (D-RISE). CVPR.
- Ultralytics YOLO26 — https://github.com/ultralytics/ultralytics

## License

Released under the MIT License (see `LICENSE`). The dataset is subject to its own license.
