#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import json

def main():
    parser = argparse.ArgumentParser()
    # Accept the string expansion from TIRA's --input flag
    parser.add_argument('--input', type=str, required=True, help="Path pattern for input files")
    parser.add_argument('--t', '--task-description', type=str, required=True, help="Path to config.json")
    parser.add_argument('--output', type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    # Map to standard SISAP docker volume targets
    input_file = args.input
    output_file = f"{args.output}/plain_local_search.h5"

    print(f"--> Input Path:  {input_file}")
    print(f"--> Output Path: {output_file}")

    # 4. Check if the input file actually exists before starting C processing
    if not os.path.exists(input_file):
        print(f"Fatal Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.t) as file:
        data = json.load(file)
        dataset = data["dataset_name"]

    # 5. Execute your compiled C binary with the paths it expects
    cmd = ["/app/KNN", input_file, output_file, dataset]

    # 5. Execute with error capturing enabled
    try:
        my_env = os.environ.copy()
        my_env["OMP_NUM_THREADS"] = "8"
        print("--> Launching C Binary...")
        result = subprocess.run(
            cmd,env=my_env,
            capture_output=True,  # Captures stdout and stderr from the binary
            text=True             # Returns strings instead of raw bytes
        )
        
        print(f"--> Binary exited with code: {result.returncode}")
        print(f"--> Binary stdout: {result.stdout}")
        print(f"--> Binary stderr: {result.stderr}")
        
        sys.exit(result.returncode)

    except Exception as e:
        print(f"--> Python exception running binary: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
