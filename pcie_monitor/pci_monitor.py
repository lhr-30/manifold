import ctypes
import mmap
import os
import time
import csv
# --- 配置 ---
SHM_NAME = "/gpu_pcie_scoreboard"
MAX_GPUS = 16
CACHE_LINE_SIZE = 64
LOG_FILE = "pcie_traffic.csv"
THRESHOLD_MB = 0.1  # 只有大于这个值才记录
class GpuCounter(ctypes.Structure):
    """必须严格匹配 C++ 的内存布局"""
    _fields_ = [
        ("h2d_bytes", ctypes.c_int64), # 8 bytes
        ("d2h_bytes", ctypes.c_int64), # 8 bytes
        # Padding: 64 - 16 = 48 bytes
        ("padding", ctypes.c_char * (CACHE_LINE_SIZE - 16))
    ]

class Scoreboard(ctypes.Structure):
    _fields_ = [("gpus", GpuCounter * MAX_GPUS)]

class PCIeMonitor:
    def __init__(self):
        self.mem = None
        self.board = None
        self._connect_shm()
        self.shm_path = f"/dev/shm{SHM_NAME}"

    def _connect_shm(self):
        """尝试连接共享内存"""
        try:
            # 修正 1: 将 os.O_RDONLY 改为 os.O_RDWR (读写模式)
            fd = os.open(f"/dev/shm{SHM_NAME}", os.O_RDWR)
            
            # 修正 2: 将 prot=mmap.PROT_READ 改为 prot=mmap.PROT_READ | mmap.PROT_WRITE
            # ctypes.from_buffer 需要可写的 memoryview
            self.mem = mmap.mmap(fd, ctypes.sizeof(Scoreboard), mmap.MAP_SHARED, 
                                 prot=mmap.PROT_READ | mmap.PROT_WRITE)
            
            self.board = Scoreboard.from_buffer(self.mem)
            print(f"[PCIeMonitor] 成功连接到共享内存: {SHM_NAME}")
        except FileNotFoundError:
            # 这是一个正常的等待过程，不用报错
            print("[PCIeMonitor] 等待共享内存创建... (请先启动被监控的程序)")
            pass
        except PermissionError:
             print("[PCIeMonitor] 权限错误: 无法以写入模式打开共享内存。")

    def get_load(self, gpu_id):
        """
        获取指定 GPU 的 Pending 流量 (单位: MB)
        返回: (h2d_mb, d2h_mb)
        """
        if not self.board:
            self._connect_shm()
            return (0.0, 0.0)
            
        if gpu_id >= MAX_GPUS:
            return (0.0, 0.0)

        # 直接读取原子变量 (64位读取在硬件上是原子的)
        # max(0, val) 是为了过滤掉极短瞬间的负值显示（多线程读写的时间差导致）
        h2d_bytes = max(0, self.board.gpus[gpu_id].h2d_bytes)
        d2h_bytes = max(0, self.board.gpus[gpu_id].d2h_bytes)

        return (h2d_bytes / 1024**2, d2h_bytes / 1024**2)

    def close(self):
        if self.mem:
            self.mem.close()
            self.mem = None

    def cleanup_shm(self):
        try:
            os.unlink(self.shm_path)
            print(f"[PCIeMonitor] 已删除共享内存: {self.shm_path}")
        except FileNotFoundError:
            pass
        except PermissionError:
            print(f"[PCIeMonitor] 权限错误: 无法删除共享内存 {self.shm_path}")


def main():
    monitor = PCIeMonitor()
    
    # 检查文件是否存在，如果不存在则写入表头
    file_exists = os.path.isfile(LOG_FILE)
    
    # 使用 'a' (append) 模式打开，这样重启脚本不会覆盖旧数据
    # buffering=1 表示行缓冲，即每写一行就 flush 到磁盘（防止程序崩溃丢失数据）
    with open(LOG_FILE, 'a', newline='', buffering=1) as f:
        writer = csv.writer(f)
        
        # 如果是新文件，写入表头
        if not file_exists:
            writer.writerow(["Timestamp", "GPU_ID", "H2D_MB", "D2H_MB"])
            print(f"创建日志文件: {LOG_FILE}")
        else:
            print(f"追加到日志文件: {LOG_FILE}")

        try:
            while True:
                current_time_str = time.strftime('%H:%M:%S')
                # 获取高精度时间戳用于日志 (方便后续画图排序)
                timestamp = time.time()
                
                # --- 屏幕显示部分 (可选：为了不闪屏太快，可以降低刷新率) ---
                # print("\033[H\033[J", end="") 
                # print(f"=== PCIe Traffic Monitor (Logging to {LOG_FILE}) ===")
                # print(f"Time: {current_time_str}")
                # print(f"{'GPU':<5} | {'H2D (Up)':<15} | {'D2H (Down)':<15}")
                # print("-" * 45)

                has_data = False
                
                # 遍历 GPU
                for i in range(8): # 假设监控 8 张卡，根据实际情况调整
                    h2d, d2h = monitor.get_load(i)
                    
                    # 只要有流量，就记录
                    if h2d > THRESHOLD_MB or d2h > THRESHOLD_MB:
                        has_data = True
                        
                        # 1. 打印到屏幕 (保留原有逻辑)
                        print(f"#{i:<4} | {h2d:>8.2f} MB    | {d2h:>8.2f} MB")
                        
                        # 2. 【核心修改】写入文件
                        # 格式: [Unix时间戳, GPU号, H2D, D2H]
                        writer.writerow([f"{timestamp:.3f}", i, f"{h2d:.4f}", f"{d2h:.4f}"])
                
                if not has_data:
                    continue
                # 调整采样频率
                # 注意：如果写文件太频繁导致IO瓶颈，可以适当调大 sleep
                time.sleep(0.0005) 

        except KeyboardInterrupt:
            print(f"\n监控停止。数据已保存至 {LOG_FILE}")
        finally:
            monitor.close()
            monitor.cleanup_shm()

if __name__ == "__main__":
    main()
