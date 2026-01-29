import torch
import time

# 确保 LD_PRELOAD 生效
# export LD_PRELOAD=./libgpu_spy.so 

def flood_test():
    device = torch.device("cuda:0")
    # 500MB 数据
    x = torch.randn(1024*1024*125, dtype=torch.float32) 
    
    print("开始洪水测试：瞬间提交 50 个大包传输...")
    print("Monitor 应该会显示巨大的 Pending 数据量 (约 25GB)")

    # 连续提交，不 sleep，不同步
    # 这会让 Driver 队列瞬间积压，Scoreboard 的计数器会瞬间暴涨
    for i in range(50):
        # 此时只是把命令推进队列，Python 不会阻塞
        _ = x.to(device, non_blocking=True)
        # print(f"提交第 {i+1} 个...", end="\r")

    print("\n所有请求已提交 Driver 队列。")
    print("现在的 PCIe 应该是满载状态。")
    print("主线程开始 Sleep 5秒，请观察 Monitor...")
    
    # 在这 5秒内，GPU 会在后台拼命搬运这 25GB 数据
    # Monitor 应该会显示 Pending 数量从 25000MB 慢慢减少到 0
    time.sleep(5)
    
    torch.cuda.synchronize()
    print("测试结束。")

if __name__ == "__main__":
    flood_test()