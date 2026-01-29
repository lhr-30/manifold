import argparse
import csv
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP_SCRIPT = os.path.join(HERE, "sweep_analysis.py")
GPU_SPY_CPP = os.path.join(HERE, "gpu_spy.cpp")
BUILD_SCRIPT = os.path.join(HERE, "build_gpu_spy.sh")
DEFAULT_CSV = os.path.join(HERE, "token_sweep_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sweep_analysis and tune MIN_BYTES_TO_TRACK based on overhead threshold."
    )
    parser.add_argument("--hidden", type=int, required=True, help="Model hidden size")
    parser.add_argument(
        "--tokens",
        type=str,
        default="",
        help="Comma-separated token counts, e.g. 1,8,32,128",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Overhead threshold in percent (default: 3.0)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help="CSV output path (default: token_sweep_results.csv)",
    )
    parser.add_argument(
        "--rerun",
        dest="rerun",
        action="store_true",
        help="Rerun sweep after rebuild",
    )
    parser.add_argument(
        "--no-rerun",
        dest="rerun",
        action="store_false",
        help="Do not rerun sweep after rebuild",
    )
    parser.set_defaults(rerun=True)
    return parser.parse_args()


def parse_tokens_arg(tokens_arg):
    if not tokens_arg:
        return []
    tokens = []
    for item in tokens_arg.split(","):
        item = item.strip()
        if not item:
            continue
        tokens.append(int(item))
    return tokens


def run_sweep(hidden, tokens, csv_path):
    cmd = [sys.executable, SWEEP_SCRIPT, "--hidden", str(hidden), "--csv", csv_path]
    if tokens:
        cmd += ["--tokens", ",".join(str(t) for t in tokens)]
    subprocess.run(cmd, cwd=HERE, check=True)


def read_results(csv_path, hidden):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tokens = int(row["Tokens"])
                overhead = float(row["Overhead_Pct"])
            except (KeyError, ValueError):
                continue
            size_bytes = tokens * hidden * 2
            rows.append({"tokens": tokens, "overhead": overhead, "bytes": size_bytes})
    return rows


def pick_min_bytes(rows, threshold):
    eligible = [r for r in rows if r["overhead"] < threshold]
    if not eligible:
        return None
    return min(eligible, key=lambda r: r["bytes"])


def update_min_bytes(min_bytes):
    with open(GPU_SPY_CPP, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = r"^#define\s+MIN_BYTES_TO_TRACK\s+\d+\s*$"
    replacement = f"#define MIN_BYTES_TO_TRACK {min_bytes}"
    new_content, count = re.subn(pattern, replacement, content, flags=re.MULTILINE)

    if count == 0:
        raise RuntimeError("MIN_BYTES_TO_TRACK not found in gpu_spy.cpp")

    if new_content != content:
        with open(GPU_SPY_CPP, "w", encoding="utf-8") as f:
            f.write(new_content)


def rebuild():
    subprocess.run(["bash", BUILD_SCRIPT], cwd=HERE, check=True)


def main():
    args = parse_args()
    tokens = parse_tokens_arg(args.tokens)

    print("[AutoTune] Running sweep_analysis...")
    run_sweep(args.hidden, tokens, args.csv)

    rows = read_results(args.csv, args.hidden)
    if not rows:
        print("[AutoTune] No valid rows found in sweep results.")
        return 1

    picked = pick_min_bytes(rows, args.threshold)
    if not picked:
        print(f"[AutoTune] No token count with overhead < {args.threshold}% found.")
        return 1

    print(
        "[AutoTune] Selected: tokens={tokens}, bytes={bytes}, overhead={overhead:.2f}%".format(
            **picked
        )
    )

    update_min_bytes(picked["bytes"])
    print(f"[AutoTune] Updated MIN_BYTES_TO_TRACK to {picked['bytes']} in gpu_spy.cpp")

    print("[AutoTune] Rebuilding libgpu_spy.so...")
    rebuild()

    if args.rerun:
        print("[AutoTune] Rerunning sweep_analysis after rebuild...")
        run_sweep(args.hidden, tokens, args.csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
