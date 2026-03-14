import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import argparse
from csv_merge import merge_csv

def main():
    parser = argparse.ArgumentParser(description='Merge All CSV from Folder data into one')
    parser.add_argument('--output', required=True, help='Output File name')
    args = parser.parse_args()
    merge_csv(args.output)

if __name__ == '__main__':
    main()
