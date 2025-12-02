"""
Compare solver outputs between SparseLU and PETSc GMRES runs.

Usage:
  python compare_linear_solvers.py --case-dir <path/to/case> \
    [--sparse output_sparseLU.csv] [--gmres output_gmres.csv] \
    [--atol 1e-8] [--rtol 1e-6]

The script aligns rows on (name, time), compares all remaining numeric columns,
and reports max/mean absolute and relative differences. It exits with a
non‑zero status if any column exceeds the tolerances.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_output(path: Path) -> pd.DataFrame:
    """Load an output CSV and index by (name, time) for alignment."""
    df = pd.read_csv(path)
    if "name" not in df.columns or "time" not in df.columns:
        raise ValueError(f"{path} is missing required columns 'name' and 'time'")
    df = df.set_index(["name", "time"]).sort_index()
    return df


def compute_differences(
    df_a: pd.DataFrame, df_b: pd.DataFrame, atol: float, rtol: float
) -> pd.DataFrame:
    """Compute absolute/relative differences for aligned numeric columns and flag rows that differ."""
    # Align on the index; non-overlapping rows will show up as NaNs
    a, b = df_a.align(df_b, join="outer")
    numeric_cols = [c for c in a.columns if np.issubdtype(a[c].dtype, np.number)]
    if not numeric_cols:
        raise ValueError("No numeric columns found to compare.")

    records = []
    for col in numeric_cols:
        a_col = a[col]
        b_col = b[col]
        abs_diff = (a_col - b_col).abs()
        rel_diff = abs_diff / np.maximum(np.maximum(a_col.abs(), b_col.abs()), 1e-15)

        # Identify the worst row for diagnostics.
        max_idx = abs_diff.idxmax()
        max_row = {"name": None, "time": None}
        if isinstance(max_idx, tuple) and len(max_idx) == 2:
            max_row["name"], max_row["time"] = max_idx

        record = {
            "column": col,
            "rows_compared": int(abs_diff.count()),
            "rows_nonzero_diff": int((abs_diff > 0).sum()),
            "rows_exceeding_tol": int(
                ((abs_diff > atol) & (rel_diff > rtol)).sum()
            ),
            "max_abs_diff": float(abs_diff.max()),
            "mean_abs_diff": float(abs_diff.mean()),
            "median_abs_diff": float(abs_diff.median()),
            "max_rel_diff": float(rel_diff.max()),
            "mean_rel_diff": float(rel_diff.mean()),
            "median_rel_diff": float(rel_diff.median()),
            "max_diff_at_name": max_row["name"],
            "max_diff_at_time": max_row["time"],
        }
        records.append(record)

    return pd.DataFrame.from_records(records).set_index("column")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare output CSVs from SparseLU and PETSc GMRES runs."
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the output CSVs (default: current directory).",
    )
    parser.add_argument(
        "--sparse",
        type=str,
        default="output_sparseLU.csv",
        help="Filename of the SparseLU output CSV (default: output_sparseLU.csv).",
    )
    parser.add_argument(
        "--gmres",
        type=str,
        default="output_gmres.csv",
        help="Filename of the PETSc GMRES output CSV (default: output_gmres.csv).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for declaring mismatch (default: 1e-8).",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-6,
        help="Relative tolerance for declaring mismatch (default: 1e-6).",
    )
    args = parser.parse_args(argv)

    sparse_path = args.case_dir / args.sparse
    gmres_path = args.case_dir / args.gmres

    if not sparse_path.exists():
        print(f"[error] SparseLU output not found: {sparse_path}", file=sys.stderr)
        return 1
    if not gmres_path.exists():
        print(f"[error] GMRES output not found: {gmres_path}", file=sys.stderr)
        return 1

    try:
        df_sparse = load_output(sparse_path)
        df_gmres = load_output(gmres_path)
    except Exception as exc:
        print(f"[error] Failed to load outputs: {exc}", file=sys.stderr)
        return 1

    try:
        summary = compute_differences(df_sparse, df_gmres, args.atol, args.rtol)
    except Exception as exc:
        print(f"[error] Failed to compute differences: {exc}", file=sys.stderr)
        return 1

    print("Comparison summary (SparseLU vs PETSc GMRES):")
    print(summary.to_string(float_format=lambda x: f"{x:.3e}"))

    failures = summary[summary["rows_exceeding_tol"] > 0].index.tolist()

    if failures:
        details = summary.loc[failures, ["rows_exceeding_tol", "max_abs_diff", "max_rel_diff"]]
        print(
            f"[fail] Columns exceeding tolerances (atol={args.atol}, rtol={args.rtol}):\n"
            f"{details.to_string(float_format=lambda x: f'{x:.3e}')}",
            file=sys.stderr,
        )
        return 2

    print(f"[pass] All compared columns within atol={args.atol}, rtol={args.rtol}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
