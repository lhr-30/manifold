import subprocess
import os
import sys
import csv
import argparse

# --- 配置 ---
LIB_PATH = "./libgpu_spy.so"
WORKLOAD_SCRIPT = "bench_token.py"
DEFAULT_MODEL_HIDDEN_SIZE = 5120  # Qwen2.5-32B 的 hidden_size

# Sweep 配置：Token 数量列表
# 覆盖：单解码 -> 小Batch -> 中Batch -> 大Prompt Prefill
DEFAULT_TOKEN_COUNTS = [1,2,4,8,16,32,64,128,512,1024,2048]

def run_iteration(tokens, hidden, iters, use_hook):
    env = os.environ.copy()
    
    cmd = [
        "python", WORKLOAD_SCRIPT, 
        "--tokens", str(tokens), 
        "--hidden", str(hidden),
        "--iters", str(iters)
    ]
    
    if use_hook:
        if not os.path.exists(LIB_PATH):
            print(f"Error: {LIB_PATH} not found.")
            sys.exit(1)
        env["LD_PRELOAD"] = os.path.abspath(LIB_PATH)
        # env["GPU_SPY_SILENT"] = "1" # 如果你的 C++ 有 debug print，建议开启静默

    # 运行 5 次取中位数，消除系统抖动
    results = []
    for _ in range(5):
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            return None
        try:
            results.append(float(res.stdout.strip()))
        except:
            return None
            
    results.sort()
    return results[2] # 取中位数

def parse_args():
    parser = argparse.ArgumentParser(description="Token overhead sweep for GPU spy.")
    parser.add_argument("--hidden", type=int, default=DEFAULT_MODEL_HIDDEN_SIZE,
                        help="Model hidden size (default: %(default)s)")
    parser.add_argument("--tokens", type=str, default="",
                        help="Comma-separated token counts, e.g. 1,8,32,128")
    parser.add_argument("--csv", type=str, default="token_sweep_results.csv",
                        help="CSV output path (default: %(default)s)")
    return parser.parse_args()

def parse_tokens_arg(tokens_arg):
    if not tokens_arg:
        return DEFAULT_TOKEN_COUNTS
    tokens = []
    for item in tokens_arg.split(","):
        item = item.strip()
        if not item:
            continue
        tokens.append(int(item))
    return tokens

def main():
    args = parse_args()
    model_hidden_size = args.hidden
    token_counts = parse_tokens_arg(args.tokens)

    print(f"=== SGLang Token Overhead Sweep (H={model_hidden_size}) ===")
    print(f"Library: {LIB_PATH}")
    print("-" * 95)
    print(f"{'Tokens':<8} | {'Size (KB)':<10} | {'Baseline (us)':<15} | {'Hooked (us)':<15} | {'Overhead':<10}")
    print("-" * 95)

    csv_rows = []

    for tokens in token_counts:
        # 计算该 Token 数下的传输大小 (KB)
        size_bytes = tokens * model_hidden_size * 2 # FP16 = 2 bytes
        size_kb = size_bytes / 1024.0
        
        # 动态调整迭代次数：小包跑多点，大包跑少点
        iters = 500 if size_kb < 1000 else 100
        if size_kb > 100000: iters = 25 

        base_us = run_iteration(tokens, model_hidden_size, iters, False)
        hook_us = run_iteration(tokens, model_hidden_size, iters, True)

        if base_us is None or hook_us is None:
            print(f"{tokens:<8} | {size_kb:<10.2f} | {'FAILED':<15} | {'FAILED':<15} | -")
            continue

        diff = hook_us - base_us
        overhead_pct = (diff / base_us) * 100
        
        # 颜色标记
        color = "\033[92m"
        if overhead_pct > 10.0: color = "\033[93m"
        if overhead_pct > 50.0: color = "\033[91m"
        reset = "\033[0m"

        print(f"{tokens:<8} | {size_kb:<10.2f} | {base_us:<15.2f} | {hook_us:<15.2f} | {color}{overhead_pct:>8.2f}%{reset}")
        
        csv_rows.append([tokens, size_kb, base_us, hook_us, overhead_pct])

    # 保存结果
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tokens", "Size_KB", "Baseline_us", "Hooked_us", "Overhead_Pct"])
        writer.writerows(csv_rows)
    
    print("-" * 95)
    print(f"Results saved to {args.csv}")

if __name__ == "__main__":
    main()
