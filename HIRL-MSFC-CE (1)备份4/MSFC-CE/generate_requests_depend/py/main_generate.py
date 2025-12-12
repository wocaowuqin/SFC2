# main_generate.py
import os
import pickle
import numpy as np
import scipy.io as sio
# 确保 data_generator.py 在同一目录下
from data_generator import generate_vnfs_catalog, generate_poisson_arrive_time_list, generate_node_requests


def save_as_mat(requests, filename):
    """将 Python 字典列表转换为 MATLAB 兼容的 .mat 文件"""
    if not requests:
        return

    # 定义结构化数组的数据类型
    dtype = [
        ('id', 'O'), ('source', 'O'), ('dest', 'O'),
        ('vnf', 'O'), ('cpu_origin', 'O'), ('memory_origin', 'O'),
        ('bw_origin', 'O'), ('arrive_time', 'O'), ('leave_time', 'O'),
        ('arrive_time_step', 'O'), ('leave_time_step', 'O'),
        ('lifetime', 'O')
    ]

    arr = np.zeros((len(requests),), dtype=dtype)

    for i, req in enumerate(requests):
        arr[i]['id'] = req['id']
        arr[i]['source'] = req['source']
        arr[i]['dest'] = np.array(req['dest']).reshape(1, -1)
        arr[i]['vnf'] = np.array(req['vnf']).reshape(1, -1)
        arr[i]['cpu_origin'] = np.array(req['cpu_origin']).reshape(1, -1)
        arr[i]['memory_origin'] = np.array(req['memory_origin']).reshape(1, -1)
        arr[i]['bw_origin'] = req['bw_origin']
        arr[i]['arrive_time'] = req['arrive_time']
        arr[i]['leave_time'] = req['leave_time']
        arr[i]['arrive_time_step'] = req['arrive_time_step']
        arr[i]['leave_time_step'] = req['leave_time_step']
        arr[i]['lifetime'] = req['lifetime']

    sio.savemat(filename, {'sorted_requests': arr})
    print(f"已保存 MATLAB 格式: {filename}")


def main():
    # --- 参数配置 ---
    T = 400
    node_important = [16, 21, 22, 24, 25, 26, 27, 28]
    lamda_important = 0.6

    # --- [修改] 设置保存路径为 ./out ---
    output_dir = './out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"输出目录: {os.path.abspath(output_dir)}")

    print("开始生成 VNF 目录...")
    all_vnf = generate_vnfs_catalog()
    requests = []

    print(f"正在生成业务请求 (T={T})...")
    for source in node_important:
        arrive_times = generate_poisson_arrive_time_list(T, lamda_important)
        node_reqs = generate_node_requests(source, node_important, arrive_times, all_vnf)
        requests.extend(node_reqs)

    # 按到达时间排序
    requests.sort(key=lambda x: x['arrive_time'])

    # 重新编号 ID
    for idx, req in enumerate(requests):
        req['id'] = idx + 1

    # --- 保存文件 ---
    # 1. 保存 Pickle
    pkl_path = os.path.join(output_dir, 'sorted_requests.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(requests, f)

    # 2. 保存 MAT
    mat_path = os.path.join(output_dir, 'sorted_requests.mat')
    save_as_mat(requests, mat_path)

    print(f"生成完成！共 {len(requests)} 条请求。")


if __name__ == '__main__':
    main()