import pandas as pd
import glob
import os

def merge_csv(output_file):
    folder_path = '../data'
    ext = os.path.splitext(output_file)[1].lower()
    if ext != '.csv':
        output_file = output_file + ".csv"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []

    for file in csv_files:
        print("Loading:", file)
        df = pd.read_csv(file, low_memory=False)
        df_list.append(df)
    merged_df = pd.concat(df_list, ignore_index=True)
    print("Total rows:", len(merged_df))
    output_path = os.path.join(folder_path, output_file)
    merged_df.to_csv(output_path, index=False)
    print("Merged dataset saved:", output_file)
