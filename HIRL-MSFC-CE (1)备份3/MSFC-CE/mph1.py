# MPH1.py
import numpy as np
import scipy.io as sio
import os
import time
from collections import namedtuple

# 导入自定义模块
from serve_request import serve_request
from serve_leave_request import serve_leave_request
# 确保 calc_atnp, calc_eval1, calc_eval 都在同一目录下
# 并且 __main__ 中的全局变量能被它们访问 (Python hack)
import builtins

# ================= 配置路径 =================
# 请根据实际情况修改
DATA_PATH = r"E:\pycharmworkspace\SFC-master\HIRL-MSFC-CE (1)\MSFC-CE\generate_requests_depend"
# 如果文件在当前目录，直接用 '.'
# DATA_PATH = '.'

# ================= 全局参数设置 =================
# 定义类似于 MATLAB global 的变量供子模块使用
builtins.node_num = 28
builtins.link_num = 300  # 假设为 300，需与 paths.mat 匹配
builtins.dc_num = 20
builtins.type_num = 8
builtins.dest_num = 5
builtins.cpu_capacity = 2000.0
builtins.memory_capacity = 1100.0
builtins.bandwidth_capacity = 500.0

# 节点编号 0-27 (Python 0-based)
builtins.node_numbering = list(range(28))

# DC 节点 (对应 MATLAB ran_gen)
# 注意：MATLAB 是 1-based，这里转为 0-based
ran_gen = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 27, 28]
builtins.DC = [x - 1 for x in sorted(ran_gen)]

# ================= 数据加载 =================
print("Loading data...")


# 模拟 Request 类
class Request:
    def __init__(self, data):
        # 假设 mat 文件结构，按需解析
        # 这里简化处理，假设 loadmat 返回的已经是对象或字典
        self.id = data['id'][0][0]
        self.source = int(data['source'][0][0]) - 1  # 0-based
        self.dest = [int(x) - 1 for x in data['dests'][0]]  # 0-based list
        self.vnf = [int(x) for x in data['vnfs'][0]]  # vnf types
        self.bw_origin = float(data['bw_origin'][0][0]) if 'bw_origin' in data.dtype.names else 10.0
        # 假设 cpu/mem origin 是数组
        self.cpu_origin = data['cpu_origin'][0] if 'cpu_origin' in data.dtype.names else np.ones(len(self.vnf)) * 10
        self.memory_origin = data['memory_origin'][0] if 'memory_origin' in data.dtype.names else np.ones(
            len(self.vnf)) * 10
        self.arrive_time = int(data['arrive_time'][0][0])
        self.leave_time = int(data['leave_time'][0][0])


try:
    # 加载 Paths
    path_data = sio.loadmat(os.path.join(DATA_PATH, 'US_Backbone_paths.mat'))
    builtins.path = path_data['Paths']  # 存入 builtins 供 calc_eval1 使用

    # 加载 Requests
    req_data = sio.loadmat(os.path.join(DATA_PATH, 'sorted_requests.mat'))
    # 将 numpy void object 转换为 Request 对象列表
    raw_requests = req_data['sorted_requests'][0]
    requests_list = [Request(r) for r in raw_requests]

    # 加载 Event List
    # 假设 event_list 结构: event_list[t].arrive_event (array), .leave_event (array)
    event_data = sio.loadmat(os.path.join(DATA_PATH, 'event_list.mat'))
    event_list = event_data['event_list'][0]  # struct array

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please check DATA_PATH.")
    exit()

# ================= 初始化资源 =================
T = 400
Bandwidth_status = np.full((T, builtins.link_num), builtins.bandwidth_capacity)
CPU_status = np.full((T, builtins.node_num), builtins.cpu_capacity)
Memory_status = np.full((T, builtins.node_num), builtins.memory_capacity)
hvt_all = np.zeros((builtins.node_num, builtins.type_num), dtype=int)

# 统计变量
cpu_load_var = np.zeros(T)
blocking_rate = np.zeros(T)
cpu_resource_comp = np.zeros(T)
memory_resource_comp = np.zeros(T)
bandwidth_resource_comp = np.zeros(T)

# 运行状态变量
block_flag = 0  # 累积被阻塞的请求数
arrival_request_num = 0

# 已服务的业务列表 (Python List 存储)
# 用于 Leave 事件查找
# 结构: {'id': event_id, 'req': request_obj, 'tree': tree_obj}
served_records = []

print("Start Simulation...")
start_time = time.time()

for t in range(T):
    if (t + 1) % 10 == 0:
        print(f"Time Step: {t + 1}")

    # 继承上一时刻状态 (如果 t > 0)
    if t > 0:
        Bandwidth_status[t] = Bandwidth_status[t - 1].copy()
        CPU_status[t] = CPU_status[t - 1].copy()
        Memory_status[t] = Memory_status[t - 1].copy()

    # 获取当前时刻事件
    # event_list[t] 可能是一个 void 对象
    # MATLAB: arrive_event_index = event_list(t).arrive_event
    curr_event = event_list[t]

    # 提取离开和到达事件 ID 列表 (处理可能为空的情况)
    try:
        leave_ids = curr_event['leave_event'][0] if curr_event['leave_event'].size > 0 else []
        arrive_ids = curr_event['arrive_event'][0] if curr_event['arrive_event'].size > 0 else []
    except IndexError:
        leave_ids = []
        arrive_ids = []

    # ================= 1. 处理离开事件 =================
    if len(leave_ids) > 0:
        remaining_records = []
        for record in served_records:
            if record['id'] in leave_ids:
                # 执行资源释放
                Bandwidth_status[t], CPU_status[t], Memory_status[t], hvt_all = \
                    serve_leave_request(record['req'], record['tree'],
                                        Bandwidth_status[t], CPU_status[t], Memory_status[t],
                                        hvt_all)
            else:
                remaining_records.append(record)
        served_records = remaining_records  # 更新列表，移除已离开的

    # ================= 2. 处理到达事件 =================
    sum_bw = 0
    sum_cpu = 0
    sum_mem = 0

    if len(arrive_ids) > 0:
        # 找到对应的 Request 对象
        # 假设 requests_list 中的 id 与 arrive_ids 对应
        # 效率优化：预先建立 id -> request 映射更佳，这里简单遍历
        current_requests = [r for r in requests_list if r.id in arrive_ids]

        # 临时列表用于传递给 serve_request
        # 注意：MATLAB 中 serve_request 会递归更新 in_serving
        # 这里简化：每次循环处理一个，成功则加入 served_records

        for req in current_requests:
            arrival_request_num += 1

            # 临时容器，仅为了适配接口
            temp_reqs = []
            temp_trees = []

            # 调用部署算法
            req, bw_s, cpu_s, mem_s, hvt_all, \
                _, res_trees, bw_c, cpu_c, mem_c, success = serve_request(
                req.id, req,
                Bandwidth_status[t], CPU_status[t], Memory_status[t], hvt_all,
                temp_reqs, temp_trees,
                builtins.node_num, builtins.link_num, builtins.type_num
            )

            if success:
                # 记录成功部署
                served_records.append({
                    'id': req.id,
                    'req': req,
                    'tree': res_trees[0]  # 取出新生成的树
                })
                sum_bw += bw_c
                sum_cpu += cpu_c
                sum_mem += mem_c
            else:
                # 阻塞
                block_flag += 1

    # ================= 3. 统计与记录 =================
    cpu_resource_comp[t] = sum_cpu
    memory_resource_comp[t] = sum_mem
    bandwidth_resource_comp[t] = sum_bw

    # 计算负载均衡度 (Variance)
    # 负载率 = 1 - 剩余/总容量
    cpu_load = 1 - (CPU_status[t] / builtins.cpu_capacity)
    cpu_load_var[t] = np.var(cpu_load)

    # 计算阻塞率
    blocking_rate[t] = block_flag / arrival_request_num if arrival_request_num > 0 else 0

end_time = time.time()
print(f"Simulation Finished in {end_time - start_time:.2f} seconds")
print(f"Final Blocking Rate: {blocking_rate[-1]:.4f}")

# ================= 结果保存 =================
# 保存为 .mat 文件供 MATLAB 绘图或分析
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sio.savemat(os.path.join(output_dir, 'blocking_rate.mat'), {'blocking_rate': blocking_rate})
sio.savemat(os.path.join(output_dir, 'cpu_load_var.mat'), {'cpu_load_var': cpu_load_var})
sio.savemat(os.path.join(output_dir, 'resource_consumption.mat'), {
    'cpu_resource_comp': cpu_resource_comp,
    'memory_resource_comp': memory_resource_comp,
    'bandwidth_resource_comp': bandwidth_resource_comp
})