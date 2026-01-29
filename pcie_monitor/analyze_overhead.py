import subprocess
import os
import sys

# --- 配置 ---
LIB_PATH = "./libgpu_spy.so"  # 你的 so 文件路径
WORKLOAD_SCRIPT = "bench_workload.py"

def run_test(scenario_name, size_kb, iters, use_hook=False):
    env = os.environ.copy()
    
    cmd = ["python", WORKLOAD_SCRIPT, "--size_kb", str(size_kb), "--iters", str(iters)]
    
    if use_hook:
        if not os.path.exists(LIB_PATH):
            print(f"错误: 找不到 {LIB_PATH}，请先编译 C++ 代码！")
            sys.exit(1)
        env["LD_PRELOAD"] = os.path.abspath(LIB_PATH)
        # 如果需要，可以屏蔽 hook 内部的 debug 打印，以免影响测速
        # env["GPU_SPY_SILENT"] = "1" 

    # 运行 5 次取平均值，减少波动
    times = []
    for _ in range(5):
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Benchmark Failed: {result.stderr}")
            return None
        try:
            times.append(float(result.stdout.strip()))
        except ValueError:
            print(f"解析输出失败: {result.stdout}")
            return None

    # 去掉最大最小值，取平均 (Trimmed Mean)
    times.sort()
    avg_time = sum(times[1:-1]) / 3 if len(times) >= 3 else sum(times) / len(times)
    return avg_time

def print_row(name, base, monitor):
    if base is None or monitor is None:
        return
    
    diff = monitor - base
    overhead_pct = (diff / base) * 100
    
    # 颜色高亮
    color = "\033[92m" # Green
    if overhead_pct > 5.0: color = "\033[93m" # Yellow
    if overhead_pct > 15.0: color = "\033[91m" # Red
    reset = "\033[0m"

    print(f"{name:<25} | {base:>10.2f} ms | {monitor:>10.2f} ms | {color}{overhead_pct:>8.2f}%{reset}")

def main():
    print(f"=== GPU Monitor Overhead Analysis ===")
    print(f"Library: {LIB_PATH}")
    print("-" * 75)
    print(f"{'Scenario':<25} | {'Baseline':>10} | {'With Hook':>10} | {'Overhead':>9}")
    print("-" * 75)

    # --- 场景 1: 小包高频 (Worst Case for CPU Overhead) ---
    # 8KB 数据 (刚好超过 MIN_BYTES_TO_TRACK 的 4KB 阈值)
    # 循环 2000 次
    base_t = run_test("Small Packet (80KB)", size_kb=80, iters=2000, use_hook=False)
    mon_t = run_test("Small Packet (80KB)", size_kb=80, iters=2000, use_hook=True)
    print_row("Small Packet (80KB)", base_t, mon_t)

    # --- 场景 2: 中包 (Common Case) ---
    # 1MB 数据
    # 循环 500 次
    base_t = run_test("Medium Packet (10MB)", size_kb=10240, iters=500, use_hook=False)
    mon_t = run_test("Medium Packet (10MB)", size_kb=10240, iters=500, use_hook=True)
    print_row("Medium Packet (10MB)", base_t, mon_t)

    # --- 场景 3: 大包 (Bandwidth Bound) ---
    # 100MB 数据
    # 循环 50 次
    base_t = run_test("Large Packet (100MB)", size_kb=102400, iters=50, use_hook=False)
    mon_t = run_test("Large Packet (100MB)", size_kb=102400, iters=50, use_hook=True)
    print_row("Large Packet (100MB)", base_t, mon_t)

    print("-" * 75)
    print("注：Overhead < 3% 通常被认为是无感知的 (Noise floor)。")

if __name__ == "__main__":
    main()