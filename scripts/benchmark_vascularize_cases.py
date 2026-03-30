#!/usr/bin/env python3

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time


DEFAULT_SVV_ROOT = pathlib.Path.home() / "Downloads" / "svVascularize"
DEFAULT_TEMPLATE = DEFAULT_SVV_ROOT / "test.py"
DEFAULT_GENERATED_DIR = DEFAULT_SVV_ROOT / "0d_tmp"
DEFAULT_SOLVER = pathlib.Path(__file__).resolve().parents[1] / "build" / "svzerodsolver"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate svZeroDSolver input cases from svVascularize/test.py for "
            "different terminal-segment counts, archive the generated inputs, "
            "and optionally run the solver with timing capture."
        )
    )
    parser.add_argument(
        "--segments",
        type=int,
        nargs="+",
        required=True,
        help="One or more terminal-segment counts to assign to N in svVascularize/test.py.",
    )
    parser.add_argument(
        "--svv-root",
        type=pathlib.Path,
        default=DEFAULT_SVV_ROOT,
        help=f"Path to the svVascularize repository. Default: {DEFAULT_SVV_ROOT}",
    )
    parser.add_argument(
        "--template",
        type=pathlib.Path,
        default=DEFAULT_TEMPLATE,
        help=f"Template script to patch temporarily. Default: {DEFAULT_TEMPLATE}",
    )
    parser.add_argument(
        "--generated-dir",
        type=pathlib.Path,
        default=DEFAULT_GENERATED_DIR,
        help=f"Expected output directory written by the generator. Default: {DEFAULT_GENERATED_DIR}",
    )
    parser.add_argument(
        "--conda-env",
        default="svv2",
        help="Conda environment used to run the svVascularize generator. Default: svv2",
    )
    parser.add_argument(
        "--solver-binary",
        type=pathlib.Path,
        default=DEFAULT_SOLVER,
        help=f"svZeroDSolver binary to run. Default: {DEFAULT_SOLVER}",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "large_case_benchmarks",
        help="Directory where archived inputs, outputs, and summaries are written.",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate and archive cases, but do not run the solver.",
    )
    parser.add_argument(
        "--cube-length",
        type=float,
        default=None,
        help=(
            "If provided, replace the first pv.Cube(...) call in the temporary "
            "generator script with a cube of this edge length."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing per-case archive directory.",
    )
    return parser.parse_args()


def replace_terminal_segments(template: str, segments: int) -> str:
    updated, count = re.subn(
        r"(?m)^N\s*=\s*\d+\s*$",
        f"N = {segments}",
        template,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not find a top-level 'N = <int>' assignment in the template script.")
    return updated


def patch_domain_mesh(template: str) -> tuple[str, bool]:
    has_m_assignment = re.search(r"(?m)^\s*m\s*=", template) is not None
    if "Domain(m)" not in template or has_m_assignment:
        return template, False

    updated, count = re.subn(r"Domain\(m\)", "Domain(pv.Cube())", template, count=1)
    if count != 1:
        raise RuntimeError("Failed to replace 'Domain(m)' with 'Domain(pv.Cube())'.")
    return updated, True


def patch_cube_length(template: str, cube_length: float | None) -> tuple[str, bool]:
    if cube_length is None:
        return template, False

    cube_expr = (
        f"pv.Cube(x_length={cube_length}, y_length={cube_length}, z_length={cube_length})"
    )
    updated, count = re.subn(r"pv\.Cube\([^)]*\)", cube_expr, template, count=1)
    if count == 0:
        return template, False
    return updated, True


def patch_output_target(template: str, generated_dir: pathlib.Path) -> tuple[str, bool]:
    output_expr = (
        "sim.write_0d_fluid_simulation("
        f"outdir={str(generated_dir.parent)!r}, folder={generated_dir.name!r})"
    )
    updated, count = re.subn(
        r"sim\.write_0d_fluid_simulation\([^)]*\)",
        output_expr,
        template,
        count=1,
    )
    if count == 0:
        return template, False
    return updated, True


def prepare_generator_script(
    template_path: pathlib.Path,
    segments: int,
    cube_length: float | None,
    generated_dir: pathlib.Path,
) -> tuple[str, bool, bool, bool]:
    template = template_path.read_text()
    template = replace_terminal_segments(template, segments)
    template, patched_domain = patch_domain_mesh(template)
    template, patched_cube = patch_cube_length(template, cube_length)
    template, patched_output = patch_output_target(template, generated_dir)
    return template, patched_domain, patched_cube, patched_output


def run_command(
    cmd: list[str],
    cwd: pathlib.Path,
    env: dict | None = None,
    stdout_path: pathlib.Path | None = None,
    stderr_path: pathlib.Path | None = None,
) -> dict:
    started = time.monotonic()
    stdout_handle = stdout_path.open("w") if stdout_path is not None else subprocess.PIPE
    stderr_handle = stderr_path.open("w") if stderr_path is not None else subprocess.PIPE
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            check=False,
        )
    finally:
        if stdout_path is not None:
            stdout_handle.close()
        if stderr_path is not None:
            stderr_handle.close()
    elapsed = time.monotonic() - started
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout": None if stdout_path is not None else completed.stdout,
        "stderr": None if stderr_path is not None else completed.stderr,
    }


def copy_if_exists(src: pathlib.Path, dst: pathlib.Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def repair_inflow_boundary_condition(case_dir: pathlib.Path) -> bool:
    solver_input = case_dir / "solver_0d.in"
    inflow_file = case_dir / "inflow.flow"
    if not solver_input.exists() or not inflow_file.exists():
        return False

    data = json.loads(solver_input.read_text())
    boundary_conditions = data.get("boundary_conditions", [])
    inflow_bc = next(
        (
            bc
            for bc in boundary_conditions
            if bc.get("bc_name") == "INFLOW" and bc.get("bc_type") == "FLOW"
        ),
        None,
    )
    if inflow_bc is None:
        return False

    bc_values = inflow_bc.get("bc_values", {})
    if bc_values.get("Q") and bc_values.get("t"):
        return False

    times = []
    flows = []
    for line in inflow_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        times.append(float(parts[0]))
        flows.append(float(parts[1]))

    if not times or not flows or len(times) != len(flows):
        raise RuntimeError(f"Could not reconstruct INFLOW bc_values from {inflow_file}")

    inflow_bc["bc_values"] = {"t": times, "Q": flows}
    solver_input.write_text(json.dumps(data, indent=4))
    return True


def parse_solver_metrics(stderr_text: str) -> dict:
    metrics = {}
    patterns = {
        "wall_clock_hms": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s+(.+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s+(\d+)",
        "user_time_seconds": r"User time \(seconds\):\s+([0-9.]+)",
        "system_time_seconds": r"System time \(seconds\):\s+([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stderr_text)
        if match:
            value = match.group(1).strip()
            if key.endswith("_kb"):
                metrics[key] = int(value)
            elif key.endswith("_seconds"):
                metrics[key] = float(value)
            else:
                metrics[key] = value
    return metrics


def analyze_solver_input(solver_input: pathlib.Path) -> dict:
    data = json.loads(solver_input.read_text())
    return {
        "num_vessels": len(data.get("vessels", [])),
        "num_junctions": len(data.get("junctions", [])),
        "num_boundary_conditions": len(data.get("boundary_conditions", [])),
        "simulation_parameters": data.get("simulation_parameters", {}),
    }


def generate_case(
    svv_root: pathlib.Path,
    template_path: pathlib.Path,
    generated_dir: pathlib.Path,
    conda_env: str,
    case_dir: pathlib.Path,
    segments: int,
    cube_length: float | None,
) -> dict:
    script_text, patched_domain, patched_cube, patched_output = prepare_generator_script(
        template_path, segments, cube_length, generated_dir
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_n_{segments}.py",
        prefix="svv_generate_",
        delete=False,
    ) as handle:
        handle.write(script_text)
        temp_script = pathlib.Path(handle.name)

    try:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(svv_root)
            if not existing_pythonpath
            else f"{svv_root}{os.pathsep}{existing_pythonpath}"
        )
        result = run_command(
            ["conda", "run", "-n", conda_env, "python", str(temp_script)],
            cwd=svv_root,
            env=env,
            stdout_path=case_dir / "generator.stdout.log",
            stderr_path=case_dir / "generator.stderr.log",
        )
    finally:
        temp_script.unlink(missing_ok=True)

    solver_input = generated_dir / "solver_0d.in"
    if result["returncode"] != 0:
        raise RuntimeError(
            f"svVascularize generation failed for N={segments}. "
            f"See {case_dir / 'generator.stderr.log'}."
        )
    if not solver_input.exists():
        raise RuntimeError(
            f"svVascularize did not produce {solver_input} for N={segments}."
        )

    copy_if_exists(generated_dir / "solver_0d.in", case_dir / "solver_0d.in")
    copy_if_exists(generated_dir / "geom.csv", case_dir / "geom.csv")
    copy_if_exists(generated_dir / "inflow.flow", case_dir / "inflow.flow")
    repaired_inflow_bc = repair_inflow_boundary_condition(case_dir)

    return {
        "generator_elapsed_seconds": result["elapsed_seconds"],
        "generator_returncode": result["returncode"],
        "patched_domain_mesh": patched_domain,
        "patched_cube_length": cube_length if patched_cube else None,
        "patched_output_target": str(generated_dir) if patched_output else None,
        "repaired_inflow_bc": repaired_inflow_bc,
    }


def run_solver(
    solver_binary: pathlib.Path,
    solver_input: pathlib.Path,
    case_dir: pathlib.Path,
) -> dict:
    if not solver_binary.exists():
        raise RuntimeError(f"Solver binary does not exist: {solver_binary}")

    output_csv = case_dir / "output.csv"
    result = run_command(
        ["/usr/bin/time", "-v", str(solver_binary), str(solver_input), str(output_csv)],
        cwd=case_dir,
    )
    stdout_path = case_dir / "solver.stdout.log"
    stderr_path = case_dir / "solver.stderr.log"
    stdout_path.write_text(result["stdout"] or "")
    stderr_path.write_text(result["stderr"] or "")

    if result["returncode"] != 0:
        raise RuntimeError(
            f"svZeroDSolver failed for {solver_input}. See {stderr_path}."
        )

    metrics = parse_solver_metrics(result["stderr"] or "")
    metrics["solver_elapsed_seconds"] = result["elapsed_seconds"]
    metrics["output_csv"] = str(output_csv)
    return metrics


def main() -> int:
    args = parse_args()

    svv_root = args.svv_root.resolve()
    template_path = args.template.resolve()
    generated_dir = args.generated_dir.resolve()
    solver_binary = args.solver_binary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not svv_root.exists():
        raise RuntimeError(f"svVascularize root does not exist: {svv_root}")
    if not template_path.exists():
        raise RuntimeError(f"Template script does not exist: {template_path}")

    summary = []

    for segments in args.segments:
        case_dir = output_dir / f"N_{segments:08d}"
        if case_dir.exists():
            if not args.overwrite:
                raise RuntimeError(
                    f"Case directory already exists: {case_dir}. Use --overwrite to replace it."
                )
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        case_summary = {
            "segments": segments,
            "svv_root": str(svv_root),
            "template": str(template_path),
            "generated_dir": str(generated_dir),
        }

        generation_metrics = generate_case(
            svv_root=svv_root,
            template_path=template_path,
            generated_dir=generated_dir,
            conda_env=args.conda_env,
            case_dir=case_dir,
            segments=segments,
            cube_length=args.cube_length,
        )
        case_summary.update(generation_metrics)

        solver_input = case_dir / "solver_0d.in"
        case_summary.update(analyze_solver_input(solver_input))

        if not args.generate_only:
            case_summary["solver_binary"] = str(solver_binary)
            case_summary.update(run_solver(solver_binary, solver_input, case_dir))

        (case_dir / "summary.json").write_text(json.dumps(case_summary, indent=2, sort_keys=True))
        summary.append(case_summary)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    for item in summary:
        solver_time = item.get("solver_elapsed_seconds")
        solver_time_text = f"{solver_time:.3f}s" if isinstance(solver_time, float) else "not run"
        print(
            f"N={item['segments']} "
            f"vessels={item['num_vessels']} "
            f"junctions={item['num_junctions']} "
            f"generate={item['generator_elapsed_seconds']:.3f}s "
            f"solve={solver_time_text}"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
