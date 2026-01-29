import torch
import time
import argparse

def run_token_benchmark(num_tokens, hidden_size, iters, device_id=0):
    device = torch.device(f"cuda:{device_id}")
    
    # 计算数据大小
    # Shape: [num_tokens, hidden_size]
    # Type: float16 (模拟半精度传输)
    try:
        data_cpu = torch.randn(num_tokens, hidden_size, dtype=torch.float16)
    except RuntimeError as e:
        print(f"OOM or Error creating tensor: {e}")
        return -1.0

    # 预热
    for _ in range(10):
        _ = data_cpu.to(device, non_blocking=True)
    torch.cuda.synchronize()

    # 计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    for _ in range(iters):
        # 模拟 Scheduler -> Engine (H2D)
        gpu_tensor = data_cpu.to(device, non_blocking=True)
        
        # 模拟 Engine -> Scheduler (D2H) (如采样结果回传，或者 Pipeline 并行)
        # 这里假设双向都有开销
        _ = gpu_tensor.to("cpu", non_blocking=True)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    
    # 返回每次迭代的平均耗时 (微秒 us)
    return (elapsed_ms * 1000) / iters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=5120) # Qwen 32B default
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    avg_us = run_token_benchmark(args.tokens, args.hidden, args.iters)
    print(f"{avg_us:.4f}")