# TelecomTS Benchmark

Run strong time-series baselines on TelecomTS — a large, high-resolution, multi-modal 5G observability dataset — for anomaly detection, root-cause classification, anomaly duration prediction, and forecasting.

## 🔧 Setup

> Requires **Python 3.11**

```bash
# 1) Clone
git clone <repo>
cd TelecomTS_Benchmark

# 2) Create & activate a virtual environment
python3.11 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 3) Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## 🚀 Run

```bash
# Activate your env first
source .venv/bin/activate

# Launch a benchmark run (uses configs/config.yaml)
python3 src/run.py
```

* **Change encoder**: set `encoder_type` in `configs/config.yaml`.
  Available:

  * TimesNet
  * Autoformer
  * Nonstationary_Transformer
  * FEDformer
  * Informer

* **Change task**: set `task_type` to one of:

  * `anomaly detection`
  * `root-cause analysis` (multi-class CE)
  * `anomaly duration`
  * `forecasting`
