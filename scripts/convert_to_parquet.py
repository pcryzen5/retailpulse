"""Utility script to convert Online Retail II Excel workbook to Parquet format.

Converting online_retail_II.xlsx to Parquet reduces file read times from ~90-120 seconds
down to ~1-2 seconds with exact data types preserved.

Usage:
    uv run python scripts/convert_to_parquet.py
    uv run python scripts/convert_to_parquet.py --input online_retail_II.xlsx \
        --output data/online_retail_II.parquet
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter


def convert_workbook_to_parquet(
    input_path: Path,
    output_path: Path,
) -> None:
    """Read Online Retail II XLSX and export canonical sales frame to Parquet."""
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading Excel workbook: {input_path} (this may take 1-2 minutes)...")
    start_time = time.perf_counter()

    df = OnlineRetailIIAdapter.read_xlsx(input_path)
    read_elapsed = time.perf_counter() - start_time
    print(f"Read {len(df):,} rows from all sheets in {read_elapsed:.2f} seconds.")

    print(f"Writing Parquet file: {output_path}...")
    write_start = time.perf_counter()
    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")
    write_elapsed = time.perf_counter() - write_start

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    total_elapsed = time.perf_counter() - start_time

    print(f"Done! Created {output_path} ({file_size_mb:.2f} MB) in {write_elapsed:.2f} seconds.")
    print(f"Total conversion time: {total_elapsed:.2f} seconds.")
    print(
        f"You can now run: uv run retailpulse etl run --source online_retail_ii "
        f"--path {output_path}"
    )


def main() -> None:
    """CLI entrypoint for Parquet conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Online Retail II XLSX workbook to fast Parquet format.",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("online_retail_II.xlsx"),
        help="Path to input online_retail_II.xlsx file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/online_retail_II.parquet"),
        help="Path to output Parquet file.",
    )

    args = parser.parse_args()
    convert_workbook_to_parquet(args.input, args.output)


if __name__ == "__main__":
    main()
