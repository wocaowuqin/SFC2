#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
matlab_to_python_generator.py
功能：复刻 MATLAB 文件逻辑，生成 Python/Matlab 都能较好读取的 .mat 数据集。
改良点：随机种子、最小时间间隔、字段形状一致性、MATLAB 风格 struct/cell 兼容性等。
"""

import numpy as np
import scipy.io as sio
from pathlib import Path

# ---------------- 全局参数 ----------------
OUTPUT_DIR = Path("./data_test")
TOTAL_TIME = 5000          # 仿真总时间（时间轴上最后一个可能到达时刻上限）
LAMBDA = 5.0               # 每 100 时间单位的平均到达数（脚本里采用平均间隔 = 100/LAMBDA）
AVG_DURATION = 500         # 业务平均持续时间（指数分布的均值）
RANDOM_SEED = 42           # 可设为 None 则不固定随机性

# 拓扑与 VNF 设置
NODE_NUM = 14
TYPE_NUM = 8

# 资源范围（模拟 MATLAB 的 randi）
CPU_RANGE = (1, 24)
MEM_RANGE = (1, 16)
BW_RANGE = (4, 8)
DELAY_RANGE = (20, 100)
SFC_LEN_RANGE = (3, 6)
DEST_NUM_RANGE = (2, 4)

# ---------- 辅助生成函数 ----------
def set_seed(seed):
    if seed is not None:
        np.random.seed(seed)
        print(f"[INFO] Random seed set to {seed}")

def generate_vnfs_catalog(type_num):
    """生成 VNF catalog，返回 dict: {type_id: {'cpu':..., 'memory':...}}"""
    print("正在生成 VNF Catalog...")
    catalog = {}
    for i in range(1, type_num + 1):
        cpu = int(np.random.randint(CPU_RANGE[0], CPU_RANGE[1] + 1))
        mem = int(np.random.randint(MEM_RANGE[0], MEM_RANGE[1] + 1))
        catalog[i] = {'cpu': cpu, 'memory': mem}
    return catalog

def generate_poisson_arrival_times(total_time, lam):
    """
    生成泊松到达时间点（整数时间）。
    这里将 LAMBDA 理解为“每100时间单位的平均到达个数”，因此平均间隔 = 100/lam。
    """
    print("正在生成泊松到达时间...")
    arrive_times = []
    current_time = 0
    mean_interval = 100.0 / lam

    while True:
        inter_arrival = np.random.exponential(mean_interval)
        gap = max(1, int(round(inter_arrival)))   # 保证至少 1
        current_time += gap
        if current_time >= total_time:
            break
        arrive_times.append(int(current_time))
    return np.array(arrive_times, dtype=int)

def generate_requests(arrive_times, vnf_catalog, node_num, type_num):
    """
    生成请求列表，每个请求为一个 dict。为提高 MATLAB 兼容性，尽量把可变长度字段
    以 object array / list 的形式保存，保存前再统一成结构数组。
    """
    print(f"正在生成 {len(arrive_times)} 个请求...")
    requests = []

    for i, arrive_t in enumerate(arrive_times):
        duration = max(1, int(np.random.exponential(AVG_DURATION)))  # 至少 1
        leave_t = int(arrive_t + duration)

        source = int(np.random.randint(1, node_num + 1))

        num_dests = int(np.random.randint(DEST_NUM_RANGE[0], DEST_NUM_RANGE[1] + 1))
        candidates = list(range(1, node_num + 1))
        if source in candidates:
            candidates.remove(source)
        dests = np.random.choice(candidates, size=num_dests, replace=False).astype(int)

        chain_len = int(np.random.randint(SFC_LEN_RANGE[0], SFC_LEN_RANGE[1] + 1))
        vnf_list = np.random.randint(1, type_num + 1, size=chain_len).astype(int)

        cpu_reqs = np.array([vnf_catalog[int(t)]['cpu'] for t in vnf_list], dtype=int)
        mem_reqs = np.array([vnf_catalog[int(t)]['memory'] for t in vnf_list], dtype=int)

        bw = int(np.random.randint(BW_RANGE[0], BW_RANGE[1] + 1))
        delay = int(np.random.randint(DELAY_RANGE[0], DELAY_RANGE[1] + 1))

        req = {
            'id': int(i),
            'arrive_time': int(arrive_t),
            'leave_time': int(leave_t),
            'duration': int(duration),
            'source': int(source),
            # 将可变长度向量保留为列向量 (N,1) 以便 MATLAB 读取为列
            'dest': dests.reshape(-1, 1).astype(int),
            'vnf': vnf_list.reshape(-1, 1).astype(int),
            'cpu_origin': cpu_reqs.reshape(-1, 1).astype(int),
            'memory_origin': mem_reqs.reshape(-1, 1).astype(int),
            'bw_origin': np.array([bw]),           # 标量以数组形式保存
            'delay_limit': np.array([delay])
        }
        requests.append(req)

    return requests

def generate_event_list(requests, total_time):
    """
    生成 event_list：每个时间步为一个 struct，含 arrive_event 和 leave_event。
    最终返回形状为 (max_t, 1) 的 object 数组，每个元素是 dict。
    """
    print("正在生成 Event List...")
    # 预留若干时间以包含离开事件
    max_t = total_time + int(AVG_DURATION * 3)
    event_list = np.empty((max_t, 1), dtype=object)

    # 初始化
    for t in range(max_t):
        event_list[t, 0] = {'arrive_event': [], 'leave_event': []}

    # 填充
    for req in requests:
        a_t = int(req['arrive_time'])
        l_t = int(req['leave_time'])
        if 0 <= a_t < max_t:
            event_list[a_t, 0]['arrive_event'].append(int(req['id']))
        if 0 <= l_t < max_t:
            event_list[l_t, 0]['leave_event'].append(int(req['id']))

    # 把每个 list 转成 numpy array（1 x K）或空的 (0,0)
    for t in range(max_t):
        arr = event_list[t, 0]['arrive_event']
        lev = event_list[t, 0]['leave_event']
        event_list[t, 0]['arrive_event'] = (np.array(arr, dtype=int).reshape(1, -1)
                                            if len(arr) > 0 else np.zeros((0, 0), dtype=int))
        event_list[t, 0]['leave_event'] = (np.array(lev, dtype=int).reshape(1, -1)
                                            if len(lev) > 0 else np.zeros((0, 0), dtype=int))
    return event_list

def convert_requests_to_mat_struct(requests):
    """
    把 Python 的 list-of-dict 转换为 scipy.savemat 更可靠处理的结构：
    - 生成一个 numpy object 数组，每个元素为一个 dict（MATLAB 端会看到 cell array of structs）
    这比直接保存 list-of-dict 能在 MATLAB 端更好兼容（尽管两者在很多情形下都能被 savemat 处理）。
    """
    n = len(requests)
    mat_requests = np.empty((n, 1), dtype=object)
    for i, r in enumerate(requests):
        # 把标量字段设为标量数组（1,1）或列向量；可变长度的字段保持为 object (列向量 ndarray)
        mat_requests[i, 0] = {
            'id': np.array([r['id']]),
            'arrive_time': np.array([r['arrive_time']]),
            'leave_time': np.array([r['leave_time']]),
            'duration': np.array([r['duration']]),
            'source': np.array([r['source']]),
            'dest': r['dest'],               # (k,1) ndarray
            'vnf': r['vnf'],                 # (m,1) ndarray
            'cpu_origin': r['cpu_origin'],   # (m,1)
            'memory_origin': r['memory_origin'], # (m,1)
            'bw_origin': r['bw_origin'],     # (1,)
            'delay_limit': r['delay_limit']  # (1,)
        }
    return mat_requests

# ---------------- 主流程 ----------------
def main():
    set_seed(RANDOM_SEED)

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = generate_vnfs_catalog(TYPE_NUM)
    arrive_times = generate_poisson_arrival_times(TOTAL_TIME, LAMBDA)
    requests = generate_requests(arrive_times, catalog, NODE_NUM, TYPE_NUM)
    event_list = generate_event_list(requests, TOTAL_TIME)

    # 转换为 MATLAB 友好的 struct/cell 表示
    mat_requests = convert_requests_to_mat_struct(requests)

    print(f"正在保存数据到 {OUTPUT_DIR} ...")
    # savemat 会把 Python object arrays / dict 转成 MATLAB cell/struct
    sio.savemat(OUTPUT_DIR / "sorted_requests.mat", {'sorted_requests': mat_requests})
    sio.savemat(OUTPUT_DIR / "event_list.mat", {'event_list': event_list})

    print(f"成功生成 {len(requests)} 个请求。")
    print("文件已保存：")
    print(f"  - {OUTPUT_DIR / 'sorted_requests.mat'}")
    print(f"  - {OUTPUT_DIR / 'event_list.mat'}")

if __name__ == "__main__":
    main()
