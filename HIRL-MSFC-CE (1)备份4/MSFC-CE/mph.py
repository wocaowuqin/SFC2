# mph.py (修正版)
import numpy as np
import scipy.io as sio
import os
import time
import config

from mph_strategy import serve_request_mph
from serve_leave_request import serve_leave_request


class Request:
    def __init__(self, data):
        keys = data.dtype.names
        self.id = data['id'][0][0]
        self.source = int(data['source'][0][0]) - 1
        if 'dests' in keys:
            raw_dest = data['dests'][0]
        elif 'dest' in keys:
            raw_dest = data['dest'][0]
        else:
            raw_dest = np.array([])
        self.dest = [int(x) - 1 for x in raw_dest.flatten()]
        if 'vnfs' in keys:
            raw_vnf = data['vnfs'][0]
        elif 'vnf' in keys:
            raw_vnf = data['vnf'][0]
        else:
            raw_vnf = np.array([1, 2, 3])
        self.vnf = [int(x) - 1 for x in raw_vnf.flatten()]
        self.bw_origin = float(data['bw_origin'][0][0]) if 'bw_origin' in keys else 50.0
        if 'cpu_origin' in keys:
            self.cpu_origin = data['cpu_origin'][0].flatten()
        else:
            self.cpu_origin = np.ones(len(self.vnf)) * 100.0
        if 'memory_origin' in keys:
            self.memory_origin = data['memory_origin'][0].flatten()
        else:
            self.memory_origin = np.ones(len(self.vnf)) * 200.0
        self.arrive_time_step = int(data['arrive_time_step'][0][0]) if 'arrive_time_step' in keys else 0
        self.leave_time_step = int(data['leave_time_step'][0][0]) if 'leave_time_step' in keys else 100
        if 'arrive_time' in keys: self.arrive_time_step = int(data['arrive_time'][0][0])
        if 'leave_time' in keys: self.leave_time_step = int(data['leave_time'][0][0])


print("=" * 60)
print("MPH 策略仿真")
print("=" * 60)

# ========== 配置检查 ==========
print("\n[1] 配置检查:")
print(f"  节点数: {config.node_num}")
print(f"  链路数: {config.link_num}")
print(f"  DC节点数: {len(config.DC)}")
print(f"  DC节点: {config.DC}")
print(f"  CPU容量: {config.cpu_capacity}")
print(f"  MEM容量: {config.memory_capacity}")
print(f"  BW容量: {config.bandwidth_capacity}")

# ========== 加载数据 ==========
print("\n[2] 加载数据...")
try:
    path_data = sio.loadmat(os.path.join(config.DATA_PATH, 'US_Backbone_paths.mat'))
    config.path = path_data['Paths']
    print(f"  ✓ 路径数据: shape={config.path.shape}")

    req_data = sio.loadmat(os.path.join(config.DATA_PATH, 'sorted_requests.mat'))
    req_key = 'sorted_requests' if 'sorted_requests' in req_data else 'requests'
    requests_list = [Request(r) for r in req_data[req_key].flatten()]
    print(f"  ✓ 请求数据: {len(requests_list)} 个请求")

    event_data = sio.loadmat(os.path.join(config.DATA_PATH, 'event_list.mat'))
    event_list = event_data['event_list'].flatten()
    print(f"  ✓ 事件数据: {len(event_list)} 个时间步")

except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    exit()

# ========== 请求样本检查 ==========
print("\n[3] 请求样本检查 (前3个):")
for i, req in enumerate(requests_list[:3]):
    print(f"  Request {req.id}:")
    print(f"    Source: {req.source}, Dests: {req.dest}")
    print(f"    VNFs: {req.vnf}, BW: {req.bw_origin}")
    print(f"    CPU: {req.cpu_origin}, MEM: {req.memory_origin}")

# ========== 初始化 ==========
T = min(400, len(event_list))
Bandwidth_status = np.full((T, config.link_num), config.bandwidth_capacity, dtype=float)
CPU_status = np.full((T, config.node_num), config.cpu_capacity, dtype=float)
Memory_status = np.full((T, config.node_num), config.memory_capacity, dtype=float)
hvt_all = np.zeros((config.node_num, config.type_num), dtype=int)

blocking_rate = np.zeros(T)
block_flag = 0
arrival_request_num = 0
served_records = []

# ========== 核心仿真 ==========
print("\n[4] 开始仿真...")
start_time = time.time()

# 🔥 调试模式：只处理前5个请求
DEBUG_MODE = True
debug_count = 0
MAX_DEBUG = 5

for t in range(T):
    if (t + 1) % 50 == 0:
        rate = block_flag / arrival_request_num if arrival_request_num else 0
        print(f"\n=== Time {t + 1} ===")
        print(f"  Requests: {arrival_request_num}, Blocked: {block_flag}, Rate: {rate:.4f}")

    # 继承上一时刻状态
    if t > 0:
        Bandwidth_status[t] = Bandwidth_status[t - 1].copy()
        CPU_status[t] = CPU_status[t - 1].copy()
        Memory_status[t] = Memory_status[t - 1].copy()

    curr_event = event_list[t]

    # 处理离开事件
    try:
        raw_leave = curr_event['leave_event']
        l_ids = raw_leave.flatten().tolist() if raw_leave.size > 0 else []
    except:
        l_ids = []

    if l_ids:
        rem = []
        for rec in served_records:
            if rec['id'] in l_ids:
                Bandwidth_status[t], CPU_status[t], Memory_status[t], hvt_all = \
                    serve_leave_request(rec['req'], rec['tree'], Bandwidth_status[t],
                                      CPU_status[t], Memory_status[t], hvt_all)
            else:
                rem.append(rec)
        served_records = rem

    # 处理到达事件
    try:
        raw_arrive = curr_event['arrive_event']
        a_ids = raw_arrive.flatten().tolist() if raw_arrive.size > 0 else []
    except:
        a_ids = []

    for rid in a_ids:
        req = next((r for r in requests_list if r.id == rid), None)
        if req:
            arrival_request_num += 1

            # 🔥 只在调试模式下详细输出
            if DEBUG_MODE and debug_count < MAX_DEBUG:
                print(f"\n{'='*60}")
                print(f"[DEBUG {debug_count+1}/{MAX_DEBUG}] Request {req.id} at t={t}")
                print(f"{'='*60}")

            # 🔥 调用MPH策略
            req_updated, bw_status, cpu_status, mem_status, hvt_all_updated, \
                res_req, res_tree, bw_consumed, cpu_consumed, mem_consumed, success = \
                serve_request_mph(req.id, req, Bandwidth_status[t], CPU_status[t],
                                Memory_status[t], hvt_all,
                                config.node_num, config.link_num, config.type_num)

            if DEBUG_MODE and debug_count < MAX_DEBUG:
                if success:
                    print(f"\n✓ SUCCESS!")
                    print(f"  Resources consumed: BW={bw_consumed:.2f}, CPU={cpu_consumed:.2f}, MEM={mem_consumed:.2f}")
                    print(f"  Tree links used: {np.sum(res_tree > 0)}")
                else:
                    print(f"\n✗ BLOCKED!")
                debug_count += 1

            if success:
                # 🔥 修复：使用更新后的资源状态
                served_records.append({'id': req.id, 'req': res_req, 'tree': res_tree})
                Bandwidth_status[t] = bw_status
                CPU_status[t] = cpu_status
                Memory_status[t] = mem_status
                hvt_all = hvt_all_updated
            else:
                block_flag += 1

    if arrival_request_num > 0:
        blocking_rate[t] = block_flag / arrival_request_num

    # 🔥 调试模式：处理完前5个请求后提前退出
    if DEBUG_MODE and debug_count >= MAX_DEBUG:
        print(f"\n[DEBUG] Processed {MAX_DEBUG} requests, exiting early...")
        break

end_time = time.time()
print(f"\n{'='*60}")
print(f"仿真完成")
print(f"{'='*60}")
print(f"耗时: {end_time - start_time:.2f}s")
print(f"总请求: {arrival_request_num}")
print(f"成功: {arrival_request_num - block_flag}")
print(f"阻塞: {block_flag}")
print(f"最终阻塞率: {blocking_rate[t]:.4f}")

# 保存结果
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
sio.savemat(os.path.join(output_dir, 'mph_blocking_rate.mat'), {'blocking_rate': blocking_rate})
print(f"\n结果已保存到 {output_dir}/mph_blocking_rate.mat")