import argparse
import pandas as pd
import os
from glob import glob


def merge_csv_files(input_folder, output_folder, output_file_name="merged.csv"):
    csv_files = glob(os.path.join(input_folder, "*.csv"))
    merged_df = pd.DataFrame()

    for file in csv_files:
        df = pd.read_csv(file)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    merged_df.reset_index(drop=True, inplace=True)
    merged_df['ID'] = merged_df.index

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, output_file_name)
    merged_df.to_csv(output_path, index=False)

    print(f"Merge completed. The merged CSV file is saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge CSV files.')
    parser.add_argument('--input_folder', type=str, default='./temp_blocks_csv', help='Input folder containing CSV files.')
    parser.add_argument('--output_folder', type=str, default='./big_map_csv' ,help='Output folder for the merged CSV file.')
    parser.add_argument('--file_name', type=str, help='Optional: Specify the output file name.')

    args = parser.parse_args()

    # Use the provided file name, or fall back to the default if none is provided.
    file_name = args.file_name + '_global_located.csv' if args.file_name else "merged.csv"

    merge_csv_files(args.input_folder, args.output_folder, file_name)
