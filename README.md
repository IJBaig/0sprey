# ML-based Cyber Attack Detection in Simulated Networks

## Project Overview
This project demonstrates a machine learning-based approach to cyber-attack detection in a simulated network environment. It covers:
- Lab setup and traffic generation
- Traffic capture and dataset creation
- Feature extraction and preprocessing
- Machine learning model training and evaluation
- Real-time/replay deployment for attack detection

## Directory Structure
- `src/` — Core Python modules (feature extraction, ML pipeline, deployment)
- `scripts/` — Utility scripts for running tasks
- `data/` — Place to store raw pcap files and processed datasets
- `models/` — Saved ML models
- `notebooks/` — Jupyter notebooks for exploration and analysis
- `.github/` — Copilot instructions and project automation

## Setup Instructions
1. **Install Python 3.8+**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Place your pcap files in the `data/` directory.**
4. **Run feature extraction:**
   ```bash
   python scripts/extract_features.py --input data/your_capture.pcap --output data/flows.csv
   ```
5. **Train the ML model:**
   ```bash
   python scripts/train_model.py --input data/flows.csv --output models/attack_detector.joblib
   ```
6. **Deploy and classify new traffic:**
   ```bash
   python scripts/classify_live.py --input data/new_capture.pcap --model models/attack_detector.joblib
   ```

## Requirements
- GNS3/EVE-NG, VirtualBox/VMware, Kali Linux, Wireshark (for lab setup)
- Python libraries: scikit-learn, pandas, numpy, scapy, pyshark, joblib

## References
- See `notebooks/` for example analysis and feature engineering.
- See `src/` for reusable modules.

---

**Replace sample data and models with your own for real experiments.**
