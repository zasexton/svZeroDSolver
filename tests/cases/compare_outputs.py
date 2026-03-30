#!/usr/bin/env python3
"""
Compare two svZeroDSolver output CSV files.

This script compares pressure and flow solutions across all vessel segments
(vessel-based CSV) or generic variables (variable-based CSV) between a
reference and a test run.

Usage
-----
    python compare_outputs.py ref.csv test.csv \
        [--abs-tol 1e-8] [--rel-tol 1e-6]

The script reports:
  - Keys present only in one file (e.g., missing vessels or times),
  - Mean and max absolute difference,
  - Mean and max relative difference (where the reference is non-zero),
  - Whether all differences satisfy the combined absolute/relative tolerances.
"""

import argparse
import csv
from dataclasses import dataclass
from math import fabs, isfinite
from typing import Dict, Iterable, List, Tuple


@dataclass
class Stats:
    count: int = 0
    sum_abs: float = 0.0
    max_abs: float = 0.0
    sum_rel: float = 0.0
    max_rel: float = 0.0

    def update(self, ref: float, diff: float, rel_eps: float = 1e-30) -> None:
        self.count += 1
        adiff = fabs(diff)
        self.sum_abs += adiff
        if adiff > self.max_abs:
            self.max_abs = adiff

        denom = max(fabs(ref), rel_eps)
        rel = adiff / denom if denom > 0.0 else 0.0
        if isfinite(rel):
            self.sum_rel += rel
            if rel > self.max_rel:
                self.max_rel = rel

    @property
    def mean_abs(self) -> float:
        return self.sum_abs / self.count if self.count > 0 else 0.0

    @property
    def mean_rel(self) -> float:
        return self.sum_rel / self.count if self.count > 0 else 0.0


def read_csv_vessel(path: str) -> Dict[Tuple[str, str], Tuple[float, float, float, float]]:
    """
    Read a vessel-based output CSV:
      name,time,flow_in,flow_out,pressure_in,pressure_out[, ...]

    Returns mapping (name, time_str) -> (flow_in, flow_out, pressure_in, pressure_out).
    """
    data: Dict[Tuple[str, str], Tuple[float, float, float, float]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"{path}: empty CSV")

        def idx(col: str) -> int:
            try:
                return header.index(col)
            except ValueError:
                raise RuntimeError(f"{path}: expected column '{col}' in header {header}")

        name_idx = idx("name")
        time_idx = idx("time")
        fi_idx = idx("flow_in")
        fo_idx = idx("flow_out")
        pi_idx = idx("pressure_in")
        po_idx = idx("pressure_out")

        for row in reader:
            if not row:
                continue
            key = (row[name_idx], row[time_idx])
            try:
                fi = float(row[fi_idx])
                fo = float(row[fo_idx])
                pi = float(row[pi_idx])
                po = float(row[po_idx])
            except ValueError as e:
                raise RuntimeError(f"{path}: non-numeric value in row {row}") from e
            data[key] = (fi, fo, pi, po)
    return data


def read_csv_variable(path: str) -> Dict[Tuple[str, str], float]:
    """
    Read a variable-based output CSV:
      name,time,y[,ydot]

    Returns mapping (name, time_str) -> y.
    """
    data: Dict[Tuple[str, str], float] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"{path}: empty CSV")

        def idx(col: str) -> int:
            try:
                return header.index(col)
            except ValueError:
                raise RuntimeError(f"{path}: expected column '{col}' in header {header}")

        name_idx = idx("name")
        time_idx = idx("time")
        y_idx = idx("y")

        for row in reader:
            if not row:
                continue
            key = (row[name_idx], row[time_idx])
            try:
                y = float(row[y_idx])
            except ValueError as e:
                raise RuntimeError(f"{path}: non-numeric value in row {row}") from e
            data[key] = y
    return data


def detect_mode(path: str) -> str:
    """Detect CSV mode: 'vessel' or 'variable' based on header columns."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"{path}: empty CSV")
    if "flow_in" in header and "pressure_in" in header:
        return "vessel"
    if "y" in header:
        return "variable"
    raise RuntimeError(f"{path}: could not detect CSV type from header {header}")


def compare_vessel(
    ref_path: str,
    test_path: str,
    abs_tol: float,
    rel_tol: float,
) -> int:
    ref = read_csv_vessel(ref_path)
    test = read_csv_vessel(test_path)

    ref_keys = set(ref.keys())
    test_keys = set(test.keys())

    only_ref = sorted(ref_keys - test_keys)
    only_test = sorted(test_keys - ref_keys)

    if only_ref:
        print(f"Entries only in reference ({len(only_ref)}):")
        for k in only_ref[:10]:
            print("  ", k)
        if len(only_ref) > 10:
            print("  ...")
    if only_test:
        print(f"Entries only in test ({len(only_test)}):")
        for k in only_test[:10]:
            print("  ", k)
        if len(only_test) > 10:
            print("  ...")

    common = sorted(ref_keys & test_keys)
    if not common:
        print("No common (name,time) entries to compare.")
        return 1

    stats_flow_in = Stats()
    stats_flow_out = Stats()
    stats_pres_in = Stats()
    stats_pres_out = Stats()

    failures: List[Tuple[Tuple[str, str], str, float, float]] = []

    for key in common:
        r_fi, r_fo, r_pi, r_po = ref[key]
        t_fi, t_fo, t_pi, t_po = test[key]

        for comp_name, r_val, t_val, stats in [
            ("flow_in", r_fi, t_fi, stats_flow_in),
            ("flow_out", r_fo, t_fo, stats_flow_out),
            ("pressure_in", r_pi, t_pi, stats_pres_in),
            ("pressure_out", r_po, t_po, stats_pres_out),
        ]:
            diff = t_val - r_val
            stats.update(r_val, diff)
            adiff = fabs(diff)
            denom = max(fabs(r_val), 1e-30)
            rel = adiff / denom if denom > 0.0 else 0.0
            if adiff > abs_tol and rel > rel_tol:
                failures.append((key, comp_name, adiff, rel))

    def print_stats(name: str, s: Stats) -> None:
        print(
            f"{name}: count={s.count}, "
            f"mean_abs={s.mean_abs:.6e}, max_abs={s.max_abs:.6e}, "
            f"mean_rel={s.mean_rel:.6e}, max_rel={s.max_rel:.6e}"
        )

    print(f"Compared {len(common)} common (name,time) entries.")
    print(f"Tolerances: abs_tol={abs_tol:.3e}, rel_tol={rel_tol:.3e}")
    print_stats("flow_in", stats_flow_in)
    print_stats("flow_out", stats_flow_out)
    print_stats("pressure_in", stats_pres_in)
    print_stats("pressure_out", stats_pres_out)

    if failures:
        print(f"\nNumber of entries exceeding tolerances: {len(failures)}")
        for key, comp, adiff, rel in failures[:20]:
            name, time_str = key
            print(
                f"  name={name}, time={time_str}, comp={comp}, "
                f"abs_diff={adiff:.6e}, rel_diff={rel:.6e}"
            )
        if len(failures) > 20:
            print("  ...")
        return 1

    print("\nAll vessel flow/pressure differences are within tolerances.")
    return 0


def compare_variable(
    ref_path: str,
    test_path: str,
    abs_tol: float,
    rel_tol: float,
) -> int:
    ref = read_csv_variable(ref_path)
    test = read_csv_variable(test_path)

    ref_keys = set(ref.keys())
    test_keys = set(test.keys())

    only_ref = sorted(ref_keys - test_keys)
    only_test = sorted(test_keys - ref_keys)

    if only_ref:
        print(f"Entries only in reference ({len(only_ref)}):")
        for k in only_ref[:10]:
            print("  ", k)
        if len(only_ref) > 10:
            print("  ...")
    if only_test:
        print(f"Entries only in test ({len(only_test)}):")
        for k in only_test[:10]:
            print("  ", k)
        if len(only_test) > 10:
            print("  ...")

    common = sorted(ref_keys & test_keys)
    if not common:
        print("No common (name,time) entries to compare.")
        return 1

    stats = Stats()
    failures: List[Tuple[Tuple[str, str], float, float]] = []

    for key in common:
        r_val = ref[key]
        t_val = test[key]
        diff = t_val - r_val
        stats.update(r_val, diff)
        adiff = fabs(diff)
        denom = max(fabs(r_val), 1e-30)
        rel = adiff / denom if denom > 0.0 else 0.0
        if adiff > abs_tol and rel > rel_tol:
            failures.append((key, adiff, rel))

    print(f"Compared {len(common)} common (name,time) entries.")
    print(f"Tolerances: abs_tol={abs_tol:.3e}, rel_tol={rel_tol:.3e}")
    print(
        f"y: count={stats.count}, "
        f"mean_abs={stats.mean_abs:.6e}, max_abs={stats.max_abs:.6e}, "
        f"mean_rel={stats.mean_rel:.6e}, max_rel={stats.max_rel:.6e}"
    )

    if failures:
        print(f"\nNumber of entries exceeding tolerances: {len(failures)}")
        for key, adiff, rel in failures[:20]:
            name, time_str = key
            print(
                f"  name={name}, time={time_str}, "
                f"abs_diff={adiff:.6e}, rel_diff={rel:.6e}"
            )
        if len(failures) > 20:
            print("  ...")
        return 1

    print("\nAll variable values are within tolerances.")
    return 0


def main(argv: Iterable[str] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two svZeroDSolver output CSV files."
    )
    parser.add_argument("ref", help="Reference CSV file")
    parser.add_argument("test", help="Test CSV file to compare against reference")
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for differences (default: 1e-8)",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=1e-6,
        help="Relative tolerance for differences (default: 1e-6)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    mode_ref = detect_mode(args.ref)
    mode_test = detect_mode(args.test)
    if mode_ref != mode_test:
        raise RuntimeError(
            f"CSV type mismatch: {args.ref} is {mode_ref}, {args.test} is {mode_test}"
        )

    if mode_ref == "vessel":
        return compare_vessel(args.ref, args.test, args.abs_tol, args.rel_tol)
    else:
        return compare_variable(args.ref, args.test, args.abs_tol, args.rel_tol)


if __name__ == "__main__":
    raise SystemExit(main())

