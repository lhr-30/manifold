#include <cstdio>
#include <cstdint>
#include <atomic>
#include <dlfcn.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <mutex> 
#include <iostream>

// --- 配置 ---
#define SHM_NAME "/gpu_pcie_scoreboard"
#define MAX_GPUS 16
#define RING_BUFFER_SIZE 16384  
#define MIN_BYTES_TO_TRACK 0    

// --- 共享内存结构 ---
struct alignas(64) GpuCounter {
    std::atomic<int64_t> h2d_pending;
    std::atomic<int64_t> d2h_pending;
    std::atomic<int64_t> h2d_total;
    std::atomic<int64_t> d2h_total;
    char padding[32];
};

struct Scoreboard {
    GpuCounter gpus[MAX_GPUS];
};

static Scoreboard* shm_board = nullptr;

// --- 环形缓冲区项 ---
struct TraceItem {
    cudaEvent_t event;
    size_t size;
    int gpu_id;
    int direction;
    std::atomic<int> status; 
};

TraceItem g_ring_buffer[RING_BUFFER_SIZE];
std::atomic<uint64_t> g_head_idx{0}; 
std::atomic<uint64_t> g_tail_idx{0}; 
std::atomic<bool> g_running{false};

// --- 原始函数类型定义 ---
typedef cudaError_t (*cudaMemcpy_func_t)(void*, const void*, size_t, cudaMemcpyKind); // [新增]
typedef cudaError_t (*cudaMemcpyAsync_func_t)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
typedef cudaError_t (*cudaEventRecord_func_t)(cudaEvent_t, cudaStream_t);
typedef cudaError_t (*cudaEventQuery_func_t)(cudaEvent_t);
typedef cudaError_t (*cudaEventCreate_func_t)(cudaEvent_t*, unsigned int);

// --- 原始函数指针 ---
static cudaMemcpy_func_t real_memcpy_sync = nullptr; // [新增]
static cudaMemcpyAsync_func_t real_memcpy_async = nullptr;
static cudaEventRecord_func_t real_event_record = nullptr;
static cudaEventQuery_func_t real_event_query = nullptr;

// --- 初始化 ---
void init_library() {
    if (shm_board) return;
    
    // 加载符号
    real_memcpy_sync = (cudaMemcpy_func_t)dlsym(RTLD_NEXT, "cudaMemcpy"); // [新增]
    real_memcpy_async = (cudaMemcpyAsync_func_t)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    real_event_record = (cudaEventRecord_func_t)dlsym(RTLD_NEXT, "cudaEventRecord");
    real_event_query = (cudaEventQuery_func_t)dlsym(RTLD_NEXT, "cudaEventQuery");
    auto real_event_create = (cudaEventCreate_func_t)dlsym(RTLD_NEXT, "cudaEventCreateWithFlags");

    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (fd != -1) {
        if (ftruncate(fd, sizeof(Scoreboard)) == -1) {
            perror("ftruncate failed");
        }
        shm_board = (Scoreboard*)mmap(0, sizeof(Scoreboard), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }

    // 预填充 Event 池
    for (int i = 0; i < RING_BUFFER_SIZE; ++i) {
        g_ring_buffer[i].status = 0;
        real_event_create(&g_ring_buffer[i].event, cudaEventDisableTiming);
    }
}

// --- 后台消费者线程 ---
void worker_loop() {
    while (g_running) {
        bool busy = false;
        uint64_t current_tail = g_tail_idx.load(std::memory_order_relaxed);
        uint64_t current_head = g_head_idx.load(std::memory_order_acquire);

        while (current_tail < current_head) {
            uint64_t idx = current_tail & (RING_BUFFER_SIZE - 1);
            TraceItem& item = g_ring_buffer[idx];

            if (item.status.load(std::memory_order_acquire) != 2) {
                break; 
            }

            if (real_event_query(item.event) == cudaSuccess) {
                if (item.direction == 0) shm_board->gpus[item.gpu_id].h2d_pending.fetch_sub(item.size, std::memory_order_relaxed);
                else                     shm_board->gpus[item.gpu_id].d2h_pending.fetch_sub(item.size, std::memory_order_relaxed);
                
                item.status.store(0, std::memory_order_release);
                current_tail++;
                busy = true;
            } else {
                break; 
            }
        }
        
        if (current_tail > g_tail_idx.load(std::memory_order_relaxed)) {
            g_tail_idx.store(current_tail, std::memory_order_release);
        }

        if (!busy) std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}

// --- 启动线程 ---
void ensure_initialized() {
    static std::once_flag flag;
    std::call_once(flag, [](){
        init_library();
        g_running = true;
        std::thread(worker_loop).detach();
    });
}

extern "C" {

// [新增] 1. 拦截同步 cudaMemcpy
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    if (!real_memcpy_sync) ensure_initialized();

    int direction = -1;
    if (kind == cudaMemcpyHostToDevice) direction = 0;
    else if (kind == cudaMemcpyDeviceToHost) direction = 1;

    int gpu_id = 0;
    bool tracked = false;

    // 如果满足追踪条件，先记账
    if (direction != -1 && count > MIN_BYTES_TO_TRACK) {
        cudaGetDevice(&gpu_id);
        if (shm_board) {
            auto& gpu = shm_board->gpus[gpu_id];
            if (direction == 0) {
                gpu.h2d_pending.fetch_add(count, std::memory_order_relaxed);
                gpu.h2d_total.fetch_add(count, std::memory_order_relaxed);
            } else {
                gpu.d2h_pending.fetch_add(count, std::memory_order_relaxed);
                gpu.d2h_total.fetch_add(count, std::memory_order_relaxed);
            }
            tracked = true;
        }
    }

    // 执行真正的同步传输（会阻塞直到完成）
    cudaError_t result = real_memcpy_sync(dst, src, count, kind);

    // 传输完成，立即销账
    if (tracked) {
        auto& gpu = shm_board->gpus[gpu_id];
        if (direction == 0) {
            gpu.h2d_pending.fetch_sub(count, std::memory_order_relaxed);
        } else {
            gpu.d2h_pending.fetch_sub(count, std::memory_order_relaxed);
        }
    }

    return result;
}

// 2. 拦截异步 cudaMemcpyAsync
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    if (!real_memcpy_async) ensure_initialized();

    int direction = -1;
    if (kind == cudaMemcpyHostToDevice) direction = 0;
    else if (kind == cudaMemcpyDeviceToHost) direction = 1;

    if (direction != -1 && count > MIN_BYTES_TO_TRACK) {
        uint64_t idx = g_head_idx.fetch_add(1, std::memory_order_relaxed);
        uint64_t mask_idx = idx & (RING_BUFFER_SIZE - 1);
        TraceItem& item = g_ring_buffer[mask_idx];

        int expected = 0;
        if (item.status.compare_exchange_strong(expected, 1, std::memory_order_acquire)) {
            int gpu_id = 0;
            cudaGetDevice(&gpu_id);

            auto& gpu = shm_board->gpus[gpu_id];
            if (direction == 0) {
                gpu.h2d_pending.fetch_add(count, std::memory_order_relaxed);
                gpu.h2d_total.fetch_add(count, std::memory_order_relaxed);
            } else {
                gpu.d2h_pending.fetch_add(count, std::memory_order_relaxed);
                gpu.d2h_total.fetch_add(count, std::memory_order_relaxed);
            }

            item.size = count;
            item.gpu_id = gpu_id;
            item.direction = direction;

            real_event_record(item.event, stream);
            item.status.store(2, std::memory_order_release);
        }
    }

    return real_memcpy_async(dst, src, count, kind, stream);
}

}