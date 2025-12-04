import os
import pickle
import numpy as np
from data_generator import generate_vnfs_catalog, generate_poisson_arrive_time_list, generate_node_requests


def main():
    # --- 参数配置 ---
    T = 400  # 时间步
    node_important = [16, 21, 22, 24, 25, 26, 27, 28]  # US Backbone 非DC节点
    lamda_important = 0.6

    # 创建保存目录
    output_dir = './data_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("开始生成 VNF 目录...")
    all_vnf = generate_vnfs_catalog()

    requests = []

    print(f"开始为 {len(node_important)} 个节点生成业务 (T={T})...")

    for source in node_important:
        # 生成到达时间
        arrive_times = generate_poisson_arrive_time_list(T, lamda_important)

        # 生成该节点的请求
        node_reqs = generate_node_requests(source, node_important, arrive_times, all_vnf)

        requests.extend(node_reqs)
        print(f"节点 {source} 生成了 {len(node_reqs)} 条请求。")

    # --- 排序与重编号 ---
    print("正在排序并重新编号...")
    # 按 arrive_time 排序
    requests.sort(key=lambda x: x['arrive_time'])

    # 重新分配ID (从1开始，或从0开始，根据你的习惯，这里保持从1开始)
    for idx, req in enumerate(requests):
        req['id'] = idx + 1

    # --- 保存文件 ---
    # 保存为 Python 的 pickle 格式 (.pkl)，方便后续 Python 程序读取
    save_path = os.path.join(output_dir, 'sorted_requests.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(requests, f)

    print(f"生成完成！共 {len(requests)} 条请求。")
    print(f"文件已保存至: {save_path}")


if __name__ == '__main__':
    main()