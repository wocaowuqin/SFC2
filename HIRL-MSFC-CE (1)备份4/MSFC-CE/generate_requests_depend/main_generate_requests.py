# main_generate_requests.py
import numpy as np
import scipy.io as sio
from poisson_arrival import generate_poisson_arrive_time_list
from generate_node_requests import generate_node_requests
import os

# 参数（和原 MATLAB 完全一致）
T = 720
Lamda_important = 0.6
node_important = [16, 21, 22, 24, 25, 26, 27, 28]  # 非数据中心热点节点

# 加载 VNF 目录
all_vnf = sio.loadmat('data/all_vnf.mat')['all_vnf'][0]

# 生成所有请求
all_requests = []
request_id_counter = 1

print("正在生成泊松请求...")
for source in node_important:
    print(f"生成节点 {source} 的请求...")
    arrive_times = generate_poisson_arrive_time_list(T, Lamda_important)
    node_reqs = generate_node_requests(source, node_important, arrive_times, all_vnf)

    # 重新编号 id
    for req in node_reqs:
        req['id'] = request_id_counter
        request_id_counter += 1
    all_requests.extend(node_reqs)

# 按到达时间排序
all_requests.sort(key=lambda x: x['arrive_time'])
sorted_requests = all_requests

# 保存
os.makedirs('output/requests', exist_ok=True)
sio.savemat('output/requests/requests.mat', {'requests': np.array(sorted_requests, dtype=object)})
sio.savemat('output/requests/sorted_requests.mat', {'sorted_requests': np.array(sorted_requests, dtype=object)})

print(f"总请求数: {len(sorted_requests)}")
print("生成完成！")