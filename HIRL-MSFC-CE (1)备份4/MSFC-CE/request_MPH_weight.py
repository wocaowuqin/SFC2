import numpy as np
import scipy.io as sio
import os


def generate_requests_for_mph(num_requests=1000000, node_num=28, dest_per_req=5):
    """
    完全对应 request_MPH_weight.m
    生成 100万条随机多播请求，用于后续计算 MPH 节点权重
    """
    requests = []
    # 保持 1-based 索引，与 MATLAB 兼容
    nodes = np.arange(1, node_num + 1)

    print(f"Generating {num_requests} requests...")

    for i in range(1, num_requests + 1):
        # 1. 随机源节点
        source = np.random.choice(nodes)

        # 2. 从剩余节点中选目的节点
        remaining_nodes = np.setdiff1d(nodes, source)
        dests = np.sort(np.random.choice(remaining_nodes, dest_per_req, replace=False))

        req = {
            'id': i,
            'source': source,
            'dest': dests.reshape(-1, 1),  # 列向量，保持 MATLAB 格式兼容性
            # 额外添加 VNF 信息以保持 Request 对象完整性，虽然后续计算权重可能不用
            'vnf': np.random.randint(1, 9, size=3)
        }
        requests.append(req)

    # 3. 确保保存目录存在
    output_dir = 'Input/Input1'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 4. 保存文件
    output_path = os.path.join(output_dir, 'request_MPH_100.mat')
    sio.savemat(output_path, {'requests': requests})
    print(f"Successfully saved to {output_path}")


if __name__ == "__main__":
    generate_requests_for_mph()