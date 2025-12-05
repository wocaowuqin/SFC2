import scipy.io as sio
import numpy as np
from pathlib import Path
import sys

# === 关键: 确保 hyperparameters.py 在同一个文件夹中 ===
try:
    import hyperparameters as H
except ImportError:
    print("错误: 找不到 hyperparameters.py。")
    print("请将此脚本与 hyperparameters.py 放在同一个文件夹中再运行。")
    sys.exit()


# === 复制 expert_msfce.py 中的解析器 ===
# (我们需要这个来读取 .mat 结构)
def parse_mat_request(req_obj) -> dict:
    req = req_obj
    try:
        parsed = {
            'id': int(req['id'][0, 0]),
            'source': int(req['source'][0, 0]),
            'dest': [int(d) for d in req['dest'].flatten()],
            'vnf': [int(v) for v in req['vnf'].flatten()],
            'bw_origin': float(req['bw_origin'][0, 0]),
            'cpu_origin': [float(c) for c in req['cpu_origin'].flatten()],
            'memory_origin': [float(m) for m in req['memory_origin'].flatten()],
            'arrival_time': int(req['arrival_time'][0, 0]),
            'leave_time': int(req['leave_time'][0, 0]),
        }
    except (IndexError, TypeError):
        parsed = {
            'id': int(req[0][0][0]),
            'source': int(req[0][1][0]),
            'dest': [int(x) for x in req[0][2].flatten()],
            'vnf': [int(x) for x in req[0][3].flatten()],
            'cpu_origin': [float(x) for x in req[0][4].flatten()],
            'memory_origin': [float(x) for x in req[0][5].flatten()],
            'bw_origin': float(req[0][6][0][0])
        }
    return parsed


# === 主检查逻辑 ===
def check_requests():
    print("--- 正在检查 'sorted_requests.mat' 与 'hyperparameters.py' 的匹配性 ---")

    # 1. 加载容量
    try:
        caps = H.CAPACITIES
        bw_cap = caps['bandwidth']
        cpu_cap = caps['cpu']
        mem_cap = caps['memory']
        print(f"\n容量 (来自 hyperparameters.py):")
        print(f"  带宽: {bw_cap}")
        print(f"  CPU:  {cpu_cap}")
        print(f"  内存: {mem_cap}\n")
    except Exception as e:
        print(f"错误: 无法从 hyperparameters.py 加载 CAPACITIES: {e}")
        return

    # 2. 加载请求
    try:
        reqs_path = H.INPUT_DIR / "sorted_requests.mat"
        reqs_mat = sio.loadmat(reqs_path)['sorted_requests']
        requests = [parse_mat_request(r) for r in reqs_mat]
        print(f"成功加载 {len(requests)} 个请求 (来自 {reqs_path})\n")
    except FileNotFoundError:
        print(f"错误: 找不到 'sorted_requests.mat' 文件于: {reqs_path}")
        return
    except Exception as e:
        print(f"错误: 加载 'sorted_requests.mat' 失败: {e}")
        return

    # 3. 逐个比较 (检查前10个)
    print("--- 检查前 10 个请求 ---")

    max_bw_req = 0
    max_cpu_req = 0
    max_mem_req = 0

    any_request_failed = False

    for i, req in enumerate(requests):
        bw_req = req['bw_origin']
        cpu_reqs = req['cpu_origin']
        mem_reqs = req['memory_origin']

        # 更新最大值
        max_bw_req = max(max_bw_req, bw_req)
        if cpu_reqs:
            max_cpu_req = max(max_cpu_req, max(cpu_reqs))
        if mem_reqs:
            max_mem_req = max(max_mem_req, max(mem_reqs))

        if i < 10:  # 只打印前10个的详情
            print(f"请求 #{req['id']}:")

            # 检查带宽
            bw_ok = bw_req <= bw_cap
            print(f"  带宽: 需求={bw_req}, 容量={bw_cap} -> {'OK' if bw_ok else '失败'}")
            if not bw_ok: any_request_failed = True

            # 检查CPU
            cpu_ok = all(c <= cpu_cap for c in cpu_reqs)
            print(f"  CPU:  需求={cpu_reqs}, 容量={cpu_cap} -> {'OK' if cpu_ok else '失败'}")
            if not cpu_ok: any_request_failed = True

            # 检查内存
            mem_ok = all(m <= mem_cap for m in mem_reqs)
            print(f"  内存: 需求={mem_reqs}, 容量={mem_cap} -> {'OK' if mem_ok else '失败'}")
            if not mem_ok: any_request_failed = True

    print("\n--- 检查完毕 ---")

    # 4. 打印最终摘要
    print("\n--- 摘要 ---")
    print(f"最大带宽需求 (所有请求): {max_bw_req} (容量: {bw_cap})")
    print(f"最大CPU需求 (单个VNF): {max_cpu_req} (容量: {cpu_cap})")
    print(f"最大内存需求 (单个VNF): {max_mem_req} (容量: {mem_cap})")

    print("\n--- 结论 ---")
    if not any_request_failed and max_bw_req <= bw_cap and max_cpu_req <= cpu_cap and max_mem_req <= mem_cap:
        print("✅ 所有被检查的请求似乎都在容量范围内。问题可能在别处。")
    else:
        print("❌ 致命错误: 发现请求的资源 *超过* 了 'hyperparameters.py' 中定义的容量。")
        print("这就是你的专家返回 '空轨迹' 并且预训练 0 步的原因。")
        print("\n请执行以下操作之一:")
        print("1. 增加 hyperparameters.py 中的 CAPACITIES (例如，将 'bandwidth' 增加到 {max_bw_req} 以上)。")
        print("2. 重新生成 sorted_requests.mat 文件，使其请求更少的资源。")


if __name__ == "__main__":
    check_requests()