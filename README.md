# AI Cancer Diagnosis Demo

## Setup
1. Create a virtualenv (recommended) and install requirements:
python -m venv venv source venv/bin/activate # or venv\Scripts\activate on Windows pip install -r requirements.txt

2. Place your dataset in `data/` with folders `train/benign`, `train/malignant`, `val/...`.

3. Train (example):
python train.py --data_root data --epochs 10
4. Run app:
python app.py
Open http://127.0.0.1:5000

**Warning:** This is a research/demo tool. Not for clinical use.