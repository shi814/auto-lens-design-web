# Auto Lens Design Web

Public Streamlit inference application for generating and evaluating scan-lens
designs from target F-number and half field of view (HFOV).

This repository intentionally contains only the code and assets required for
inference, optical metric calculation, visualization, and ZMX/JSON export.
Training scripts, optimizers, training datasets, experiment logs, and research
utilities are maintained separately and are not included.

## Structure

```text
auto-lens-design-web/
├── app.py                         # Streamlit entry point
├── Test_Model.py                  # Inference pipeline
├── USL_Loss.py                    # Optical ray tracing and metrics
├── dataset_norm.py                # Inference normalization
├── utils.py                       # Inference configuration/model loading
├── models/
│   └── inference_model.py         # Network architecture used for inference
├── weights/
│   ├── stage1.pth
│   ├── AirGapUnsupervised_final.pth
│   └── parameters_airgap_unsupervised.txt
├── data/
│   └── normalization_reference.csv
├── glass/                         # OTS catalog and glass-name data
├── exports/                       # ZMX/JSON export implementation
├── glass_matching/                # Glass-name matching
├── lens_visualization/            # 2D, spot, and distortion plots
├── assets/
└── requirements.txt
```

## Run locally

Use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud

Deploy `app.py` and select Python 3.12. The CPU-only PyTorch index is configured
in `requirements.txt`.

## Scope

The included model architecture and optical evaluator are required to load the
published weights and calculate the metrics shown by the web application. No
training loop or training dataset is included.
