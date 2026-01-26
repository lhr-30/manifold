#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

// 检查 CUDA 错误的宏
#define CHECK_CUDA(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA Error: %s at %s:%d\\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    throw std::runtime_error("CUDA Error"); \
  } \
} while (0)

// 原地 Pin 住内存 (In-Place Pin)
void pin_buffer(torch::Tensor t) {
    // 必须确保 Tensor 在内存中是连续的
    if (!t.is_contiguous()) {
        throw std::runtime_error("Tensor must be contiguous to register.");
    }
    
    void* ptr = t.data_ptr();
    size_t size = t.numel() * t.element_size();
    
    // cudaHostRegisterPortable: 让内存对所有 CUDA Context 可见
    // 这对于多进程环境非常重要
    CHECK_CUDA(cudaHostRegister(ptr, size, cudaHostRegisterPortable));
}

// 解锁内存
void unpin_buffer(torch::Tensor t) {
    void* ptr = t.data_ptr();
    CHECK_CUDA(cudaHostUnregister(ptr));
}

// 手动绑定模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pin_buffer", &pin_buffer, "In-place cudaHostRegister");
    m.def("unpin_buffer", &unpin_buffer, "cudaHostUnregister");
}