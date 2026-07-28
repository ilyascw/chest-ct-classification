# Model artifacts

Inference expects two files in this directory:

| File | Purpose | Distribution |
|---|---|---|
| `CT_LiPro_v2.pt` | Published CT-CLIP/CT-LiPro checkpoint | Download from the [CT-RATE model collection](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/tree/main/models/CT-CLIP-Related) |
| `catboost_pathology_classifier.cbm` | Project-specific binary classifier over 512D embeddings | Export from the training workflow in `notebooks/final.ipynb` |

The binaries are excluded from Git because of their size and third-party
distribution terms. The API intentionally fails during startup when either
artifact is absent, rather than serving uninitialized predictions.
