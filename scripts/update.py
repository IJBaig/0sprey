import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import argparse
from ml_pipeline import update_model

"""
Usage:
    python update.py --model model.onnx --input new_attacks.csv
    python update.py --model model.onnx --input new_data.csv --algo sgd
"""

def main():
    parser = argparse.ArgumentParser(description="Update model with new data")
    parser.add_argument("--model", required=True, help="Existing .onnx model")
    parser.add_argument("--input", required=True, help="New labeled CSV data")
    parser.add_argument("--algo", default=None, choices=["rf", "sgd"], help="Override algorithm (default: keep original)")
    args = parser.parse_args()

    update_model(args.model, args.input, args.algo)


if __name__ == "__main__":
    main()

