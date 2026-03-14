import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import argparse
from ml_pipeline import train_model

"""
Usage:
    python train.py --input data/Combination.csv --output model.onnx --algo rf
    python train.py --input data/Combination.csv --output model.onnx --algo sgd
"""

def main():
    parser = argparse.ArgumentParser(description='Train ML model for attack detection')
    parser.add_argument('--input', required=True, help='Input dataset CSV file')
    parser.add_argument('--output', required=True, help='Output .onnx model file')
    parser.add_argument('--algo', default='rf', choices=['rf', 'sgd'], help='Algorithm: rf (Random Forest) or sgd (SGD)')
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    args = parser.parse_args()
    train_model(args.input, args.output, args.algo, args.test_size)

if __name__ == '__main__':
    main()



    




