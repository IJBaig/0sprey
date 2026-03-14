import os
import pandas as pd
import argparse
from ml_pipeline import predict
from feature_extraction import extract_features, keep_cic_features


def classify_pcap(pcap_path, model_path):
    ext = os.path.splitext(pcap_path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(pcap_path)
        df = keep_cic_features(df)
    else:
        df = keep_cic_features(extract_features(pcap_path))
    col = 'label'
    if col in df.columns:
        df = df.drop(col, axis=1)

    result_df = predict(model_path, df)
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Classify pcap traffic using trained model')
    parser.add_argument('--input', required=True, help='Input pcap or csv file')
    parser.add_argument('--model', required=True, help='Trained ONNX model file')
    parser.add_argument('--output', default='predictions.csv', help='Output CSV')
    args = parser.parse_args()

    result_df = classify_pcap(args.input, args.model)
    result_df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")

if __name__ == '__main__':
    main()




