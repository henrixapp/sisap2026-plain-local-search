#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # Accept the string expansion from TIRA's --input flag
    parser.add_argument('--input', type=str, required=True, help="Path pattern for input files")
    parser.add_argument('--t', '--task-description', type=str, required=True, help="Path to config.json")
    parser.add_argument('--output', type=str, required=True, help="Path to output directory")
    args, extra_args = parser.parse_known_args()

    # Map to standard SISAP docker volume targets
    input_file = args.input
    output_file = os.path.join(args.output, "plain_local_search.h5")

    print(f"--> Input Path:  {input_file}")
    print(f"--> Output Path: {output_file}")

    if not os.path.exists(input_file):
        print(f"Fatal Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.t) as file:
        data = json.load(file)
        dataset = data["dataset_name"]

    cmd = ["/app/KNN", input_file, "-o", output_file, "-d", dataset, *extra_args]

    try:
        print(f"--> Launching C Binary: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
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
