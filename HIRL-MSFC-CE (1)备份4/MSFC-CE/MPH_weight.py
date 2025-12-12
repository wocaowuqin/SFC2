import numpy as np
import scipy.io as sio
import os


def compute_mph_weight(requests_mat_path='sorted_requests.mat',
                       paths_mat_path='US_Backbone_paths.mat'):
    """
    完全复刻 MPH_weight.m 的逻辑
    注意：MATLAB 代码中实际上统计了目的节点，但排除了源节点。
    """

    # 1. 路径检查
    if not os.path.exists(requests_mat_path) or not os.path.exists(paths_mat_path):
        print(f"Error: 数据文件未找到。请检查路径:\n{requests_mat_path}\n{paths_mat_path}")
        return np.zeros(28)

    # 2. 加载数据
    print("Loading data...")
    # 加载预计算的 K-Shortest Paths
    # 注意：根据之前的对话，文件名可能是 US_Backbone_paths.mat (带s)
    paths_data = sio.loadmat(paths_mat_path)
    path_struct = paths_data['Paths']  # 28x28 struct array

    # 加载请求数据 (使用 sorted_requests 或 request_MPH_100，根据你实际有的文件)
    try:
        requests_data = sio.loadmat(requests_mat_path)
        # 尝试适配不同的变量名
        if 'sorted_requests' in requests_data:
            all_requests = requests_data['sorted_requests'][0]
        elif 'requests' in requests_data:
            all_requests = requests_data['requests'][0]
        else:
            raise ValueError("mat文件中未找到 requests 或 sorted_requests 变量")
    except Exception as e:
        print(f"Error loading requests: {e}")
        return np.zeros(28)

    n = 28
    D = np.zeros(n)  # 节点计数器

    print(f"Processing {len(all_requests)} requests...")

    # 3. 遍历所有请求
    for i, req in enumerate(all_requests):
        # 解析源节点 (转换为 0-based)
        source = int(req['source'][0, 0]) - 1

        # 解析目的节点 (转换为 0-based set)
        dests = set([int(d) - 1 for d in req['dests'][0].flatten()])  # 兼容 dests 或 dest

        # Vt: 当前多播树包含的节点集合 (初始只有源节点)
        Vt = {source}

        # Z: 尚未加入树的目的节点
        Z = dests.copy()

        # Prim-like 算法构建多播树
        # 只要 Z 中还有节点不在 Vt 中
        while not Z.issubset(Vt):
            shortest_path = []
            shortest_dist = float('inf')

            # 遍历每一个尚未覆盖的目的节点 z
            targets = Z - Vt

            # 在 Vt (树上节点) 和 Targets (未覆盖目的节点) 之间找最短路
            # 对应 MATLAB: for z = Z; if ~ismember...; for s1 = Vt...
            for z in targets:
                for s1 in Vt:
                    # 获取 s1 -> z 的路径数据
                    # path_struct[s1, z] 是一个 void object
                    # 需要非常小心地解析 scipy.io 加载的结构
                    try:
                        # 假设 path_struct 是 (28,28) 的 numpy object array
                        # 对应 MATLAB: path(s1, z)
                        entry = path_struct[s1, z]

                        # MATLAB: pathsdistance(1) -> 获取第一条路径的距离
                        # Python: ['pathsdistance'][0,0] 可能是一个数组，取第一个元素
                        dist_val = entry['pathsdistance'][0][0]
                        if isinstance(dist_val, np.ndarray):
                            dist = float(dist_val[0])  # 取 k=1
                        else:
                            dist = float(dist_val)

                        if dist < shortest_dist:
                            shortest_dist = dist

                            # 获取具体路径: paths(1, 1:dist+1)
                            # Python: ['paths'][0] 取出路径矩阵, 再取第0行
                            full_paths = entry['paths']
                            # 确保是二维数组取第一行
                            if full_paths.ndim == 2:
                                p_raw = full_paths[0]
                            else:
                                p_raw = full_paths.flat[0][0]  # 处理嵌套情况

                            # 截取有效部分 (距离+1 个节点)
                            valid_len = int(dist) + 1
                            shortest_path = [int(x) - 1 for x in p_raw[:valid_len]]

                    except Exception as e:
                        # 路径不存在或解析错误
                        continue

            # 更新 Vt
            if shortest_path:
                Vt.update(shortest_path)
            else:
                # 无法到达剩余节点 (图不连通?)，跳出防止死循环
                break

        # 4. 统计节点频率 (修正后的逻辑)
        # MATLAB 逻辑: passing_node2 = setdiff(Vt, request.source)
        # 即: 统计树上除了 Source 以外的所有节点 (包含 Dests)
        nodes_to_count = Vt - {source}

        for v in nodes_to_count:
            D[v] += 1

    # 5. 计算权重
    if D.sum() == 0:
        weight = np.zeros(n)
    else:
        weight = D / D.sum()

    return weight


if __name__ == "__main__":
    # 确保文件名与你实际上传的一致
    w = compute_mph_weight(
        requests_mat_path='sorted_requests.mat',
        paths_mat_path='US_Backbone_paths.mat'
    )

    print("Calculated MPH Weights:")
    # 格式化输出方便对比
    print(np.array2string(w, formatter={'float_kind': lambda x: "%.4f" % x}))

    # 保存结果
    sio.savemat('MPH_weight100.mat', {'weight': w})
    print("Saved to MPH_weight100.mat")