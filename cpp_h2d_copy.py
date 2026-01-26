import torch
import gc
import torch.utils.cpp_extension
import time
import os

# ==========================================
# 1. C++ Extension: 暴露 cudaHostRegister
# ==========================================
# read the C++ source code
cpp_source = open('custom_pinner.cpp', 'r').read()

# 为了防止之前的错误构建缓存干扰，我们换个名字或者你也可以手动清理 ~/.cache/torch_extensions
module_name = 'custom_pinner_fixed'

print("正在编译 C++ 扩展，请稍候...")
try:
    custom_ops = torch.utils.cpp_extension.load_inline(
        name=module_name,
        cpp_sources=cpp_source,
        # 关键修改：删除了 functions=['...', '...'] 参数
        # 因为我们已经在 cpp_source 里自己写了 PYBIND11_MODULE
        verbose=True,
        with_cuda=True
    )
except Exception as e:
    print("\n编译依然失败。建议尝试执行: rm -rf ~/.cache/torch_extensions 清理缓存后重试。")
    raise e

print("编译成功！开始 Benchmark...")

# ==========================================
# 2. Benchmark 工具函数
# ==========================================
def benchmark_h2d(tensor, label, n_warmup=5, n_iter=20):
    size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
    
    # 创建 CUDA Events 用于精确计时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # 预热
    for _ in range(n_warmup):
        _ = tensor.to('cuda', non_blocking=True)
    
    torch.cuda.synchronize()
    
    # 开始测试
    start_event.record()
    for _ in range(n_iter):
        # non_blocking=True 是关键，只有 Pinned Memory 才能真正实现非阻塞
        _ = tensor.to('cuda', non_blocking=True)
    end_event.record()
    
    end_event.synchronize()
    
    total_time_ms = start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / n_iter
    bandwidth = size_mb / (avg_time_ms / 1000) # MB/s
    
    print(f"[{label:20s}] 耗时: {avg_time_ms:.3f} ms | 带宽: {bandwidth:.2f} MB/s")
    return bandwidth

def run_test():
    # 保持 40MB 以适应 Docker
    N = 100 * 1024 * 1024
    dtype = torch.float32
    print(f"=== Testing with Tensor Size: 40.00 MB ===")

    # --- Case 1 ---
    print("\nRunning Case 1...")
    t_shared = torch.randn(N, dtype=dtype)
    t_shared.share_memory_()
    benchmark_h2d(t_shared, "Shared (Unpinned)")

    # 关键修复：立即删除并回收内存，释放 /dev/shm
    del t_shared
    gc.collect()

    # --- Case 2 ---
    print("\nRunning Case 2...")
    # 重新创建 t_shared
    t_shared = torch.randn(N, dtype=dtype)
    t_shared.share_memory_()

    start_copy = time.time()
    t_pinned_native = t_shared.pin_memory() # 这里产生了 Copy
    copy_overhead = (time.time() - start_copy) * 1000
    print(f"   >>> Native Pin Copy Overhead: {copy_overhead:.3f} ms")
    print(f"tensor is shared: {t_pinned_native.is_shared()}")
    benchmark_h2d(t_pinned_native, "Shared -> Native Pin")

    # 再次清理
    del t_shared
    del t_pinned_native
    gc.collect()

    # --- Case 3 ---
    print("\nRunning Case 3...")
    t_custom = torch.randn(N, dtype=dtype)
    t_custom.share_memory_()

    start_register = time.time()
    # 原地 Pin，不产生 Copy
    custom_ops.pin_buffer(t_custom)
    register_overhead = (time.time() - start_register) * 1000
    print(f"   >>> In-Place Pin Overhead:    {register_overhead:.3f} ms (Expecting < 1ms)")
    print(f"tensor is shared: {t_custom.is_shared()}")

    benchmark_h2d(t_custom, "Shared + InPlace Pin")

    custom_ops.unpin_buffer(t_custom)

if __name__ == "__main__":
    run_test()