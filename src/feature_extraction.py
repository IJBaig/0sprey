import os
import subprocess
import glob
import pandas as pd
import argparse
import tempfile
import shutil

def extract_features(pcap_file):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    NATIVE_LIB = os.path.join(base_dir, "src/jnetpcap/linux/jnetpcap-1.3.0")
    CICFLOW_JAR = os.path.join(base_dir, "src/CICFlowMeter-all-4.0.jar")
    
    JAVA_BIN = "/usr/lib/jvm/temurin-8-jdk-amd64/bin/java"
    # replace with your own java 8 path
    
    if not os.path.isfile(JAVA_BIN):
        raise FileNotFoundError(
            f"JAVA 8 file not found: {JAVA_BIN}\n"
            "Download Java 8 And replace The path Of JAVA_BIN Accordigly\n"
            "See CICFlowMeter_instalation_process.md For Download Process"
        )
    if not os.path.isdir(NATIVE_LIB):
        raise FileNotFoundError(
            f"directory not found: {NATIVE_LIB}\n"
            "get 'jnetpcap/linux/jnetpcap-1.3.0' for detail\n"
            "See CICFlowMeter_instalation_process.md For Download Process"
        )
    if not os.path.isfile(pcap_file):
        raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
    if not os.path.isfile(CICFLOW_JAR):
        raise FileNotFoundError(
            f"CICFlowMeter JAR not found: {CICFLOW_JAR}\n"
            "See CICFlowMeter_instalation_process.md For Download Process"
        )
    temp_dir = tempfile.mkdtemp()
    try:
        cmd = [
            JAVA_BIN,
            "-Dlog4j.rootLogger=OFF",
            f"-Djava.library.path={NATIVE_LIB}",
            "-jar",
            CICFLOW_JAR,
            os.path.abspath(pcap_file),
            temp_dir
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"CICFlowMeter failed (exit code {result.returncode}):\n{result.stderr}"
            )
        csv_files = glob.glob(os.path.join(temp_dir, "*.csv"))
        if not csv_files:
            print(f"[!] No CSV output generated for: {pcap_file}")
            return pd.DataFrame()
        rows = [pd.read_csv(f, low_memory=False) for f in csv_files]
        df = pd.concat(rows, ignore_index=True)
        df.columns = df.columns.str.strip()
        # Drop rows where all values are NaN (null)
        df.dropna(how="all", inplace=True)
        # Replace infinity values
        df.replace([float("inf"), float("-inf")], pd.NA, inplace=True)
        return df
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"CICFlowMeter timed out processing: {pcap_file}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
def keep_cic_features(df):

    main_features = [
        "Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets", "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Fwd Packet Length Max", " Fwd Packet Length Min", " Fwd Packet Length Mean", " Fwd Packet Length Std", "Bwd Packet Length Max", " Bwd Packet Length Min", " Bwd Packet Length Mean", " Bwd Packet Length Std", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean", " Flow IAT Std", " Flow IAT Max", " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std", " Fwd IAT Max", " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std", " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags", " Bwd URG Flags", " Fwd Header Length", " Bwd Header Length", "Fwd Packets/s", " Bwd Packets/s", " Min Packet Length", " Max Packet Length", " Packet Length Mean", " Packet Length Std", " Packet Length Variance", "FIN Flag Count", " SYN Flag Count", " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count", " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio", " Average Packet Size", " Avg Fwd Segment Size", " Avg Bwd Segment Size", " Fwd Header Length.1", "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate", " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets", " Subflow Fwd Bytes", " Subflow Bwd Packets", " Subflow Bwd Bytes", "Init_Win_bytes_forward", " Init_Win_bytes_backward", " act_data_pkt_fwd", " min_seg_size_forward", "Active Mean", " Active Std", " Active Max", " Active Min", "Idle Mean", " Idle Std", " Idle Max", " Idle Min", " Label"
    ]
    # Keep only the columns that exist in df
    features_to_keep = [c for c in main_features if c in df.columns]
    df = df[features_to_keep].copy()
    return df

def main():
    parser = argparse.ArgumentParser(description='Extract flow features from pcap')
    parser.add_argument('--input', required=True, help='Input pcap file')
    parser.add_argument('--output', required=True, help='Output CSV file')
    args = parser.parse_args()
    df = keep_cic_features(extract_features(args.input))
    df.to_csv(args.output, index=False)
    print(f"Features extracted to {args.output}")

if __name__ == '__main__':
    main()
