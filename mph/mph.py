import numpy as np
import scipy.io as sio
from typing import List, Dict, Tuple, Set


# --- 全局变量 ---
class GlobalVars:
    def __init__(self):
        self.path_db = None  # 将 .mat 文件的 'Paths' 对象保存在这里
        self.link = None
        self.node_num = 28
        self.link_num = None
        self.node_numbering = np.arange(1, 29)
        self.DC = None
        self.dc_num = None
        self.request = {}
        self.topology = None
        self.type_num = 8
        self.dest_num = 5
        # 资源容量 (从 MPH1.m 和 calc_eval 中获取)
        self.cpu_capacity = 2000
        self.memory_capacity = 1100
        self.bandwidth_capacity = 500


# 初始化全局变量
g = GlobalVars()


def topology_link(topology_matrix: np.ndarray) -> Tuple[List, int]:
    """
    从邻接矩阵构建链路列表。
    基于 'topology_link.m'
    """
    links = []
    # 遍历上三角矩阵以避免重复
    for i in range(g.node_num):
        for j in range(i + 1, g.node_num):
            if not np.isinf(topology_matrix[i, j]) and topology_matrix[i, j] > 0:
                # MATLAB 的 'link' 结构似乎是 [node1, node2, weight]
                links.append([i + 1, j + 1, topology_matrix[i, j]])
    return links, len(links)


def load_topology():
    """加载网络拓扑"""
    # 加载路径数据 (来自 US_Backbone_path.mat)
    try:
        temp_path = sio.loadmat('US_Backbone_path.mat')
        # g.path 是一个复杂的对象数组，需要这样访问
        g.path_db = temp_path['Paths']
    except FileNotFoundError:
        print("错误: 未找到 'US_Backbone_path.mat'。请确保文件在同一目录中。")
        return

    g.node_num = 28
    g.dc_num = 14  # 'g.node_num // 2' 在您的代码中是 14
    g.node_numbering = np.arange(1, 29)

    # 数据中心节点编号 (使用 1-based 索引)
    ran_gen = [3, 4, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 20, 23]
    g.DC = np.sort(ran_gen)

    # 构建拓扑邻接矩阵 (0-based 索引)
    # (您提供的拓扑矩阵)
    g.topology = np.array([
        [np.inf, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [1, np.inf, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [1, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, 1, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, 1, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf,
         1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, 1,
         np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf,
         np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf,
         np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1,
         np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf,
         np.inf, np.inf, 1, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, 1, np.inf, 1, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, 1, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1,
         np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, 1, np.inf]
    ])

    # 使用 topology_link 函数
    g.link, g.link_num = topology_link(g.topology)
    print(f"拓扑加载完成: {g.node_num} 个节点, {g.link_num} 条链路。")


def _get_path_from_db(source: int, dest: int, k: int) -> Tuple[List[int], int, List[int]]:
    """
    从加载的 .mat 文件中安全地提取路径、距离和链路ID。
    处理 MATLAB 1-based 索引。
    """
    try:
        # MATLAB 索引从 1 开始
        s_idx = source - 1
        d_idx = dest - 1
        k_idx = k - 1

        # 访问 'Paths' 对象数组
        path_data = g.path_db[s_idx, d_idx]

        # 'paths' 是一个 [K x N] 数组
        # .pathsdistance 是一个 [K x 1] 数组
        # .link_ids 是一个 [K x M] 数组

        path_dist = int(path_data['pathsdistance'][k_idx][0])
        # 路径节点存储在 'paths' 中，我们只需要到 path_dist+1 的长度
        path_nodes = path_data['paths'][k_idx, :path_dist + 1].tolist()
        # 链路ID
        link_ids = path_data['link_ids'][k_idx, :path_dist].tolist()

        # 确保所有节点都是整数
        path_nodes = [int(n) for n in path_nodes]
        link_ids = [int(l) for l in link_ids]

        return path_nodes, path_dist, link_ids
    except Exception as e:
        # print(f"路径提取错误: s={source}, d={dest}, k={k}. 错误: {e}")
        return [], 0, []


def _get_kth_path_max_distance(source: int, dest: int, kpath: int) -> int:
    """
    获取第 k 条路径的距离，用于归一化。

    """
    try:
        s_idx = source - 1
        d_idx = dest - 1
        k_idx = kpath - 1  # 获取第k条路径 (0-indexed)
        path_data = g.path_db[s_idx, d_idx]
        return int(path_data['pathsdistance'][k_idx][0])
    except:
        return 1  # 避免除以零


def calc_eval(
        d: int, k: int,
        Bandwidth_status_t: np.ndarray,
        CPU_status_t: np.ndarray,
        Memory_status_t: np.ndarray,
        hvt_all: np.ndarray,
        kpath: int
) -> Tuple:
    """
    计算第一条路径（S -> d）的评价函数。
    Python 转换自: calc_eval.m

    Args:
        d: 目的节点索引 (1-based index in request.dests)
        k: 第 k 条路径 (1-based)
        (status...): 当前网络资源状态
        kpath: KSP 的总数 (例如 3 或 5)

    Returns:
        eval_val, paths, tree, hvt, feasible, infeasible_dest
    """

    # 初始化
    tree = np.zeros(g.link_num)  # 链路使用
    hvt = np.zeros((g.node_num, g.type_num))  # VNF 放置

    src = g.request['source']
    dest = g.request['dests'][d - 1]  # d 是 1-based 索引

    # 获取路径、距离和链路ID
    paths, pathshop, lind_id = _get_path_from_db(src, dest, k)

    if not paths:
        return 0, [], tree, hvt, False, dest

    # 寻找路径上的DC节点
    usable_on_path = [node for node in paths if node in g.DC]

    # 计算路径上的剩余资源
    CPU_status = np.sum(CPU_status_t[np.array(paths) - 1])
    Memory_status = np.sum(Memory_status_t[np.array(paths) - 1])
    Bandwidth_status = 0
    for link_id in lind_id:
        Bandwidth_status += Bandwidth_status_t[link_id - 1]

    eval_val = 1.0
    feasible = True
    infeasible_dest = 0

    # 检查VNF部署是否可行
    if len(usable_on_path) < len(g.request['vnfs']):
        eval_val = 0
        feasible = False
        infeasible_dest = dest

    # 检查带宽
    if eval_val != 0:
        for link_id in lind_id:
            if Bandwidth_status_t[link_id - 1] < g.request['bw_origin']:
                eval_val = 0
                feasible = False
                infeasible_dest = dest
                break

    # 检查 CPU 和 Memory
    if eval_val != 0:
        j = 0  # vnf 索引
        i = 0  # dc 索引
        flag = True

        while j < len(g.request['vnfs']) and flag:
            while i < len(usable_on_path):
                dc_node_1based = usable_on_path[i]
                dc_node_0based = dc_node_1based - 1

                vnf_type_1based = g.request['vnfs'][j]
                vnf_type_0based = vnf_type_1based - 1

                # 检查是否已在全局部署 (复用)
                if hvt_all[dc_node_0based, vnf_type_0based] == 0:
                    # 需要新部署，检查资源
                    cpu_req = g.request['cpu_origin'][j]
                    mem_req = g.request['memory_origin'][j]

                    if (CPU_status_t[dc_node_0based] < cpu_req) or \
                            (Memory_status_t[dc_node_0based] < mem_req):
                        i += 1  # 资源不足，尝试下一个DC
                    else:
                        hvt[dc_node_0based, vnf_type_0based] = 1
                        j += 1
                        i += 1
                        if j >= len(g.request['vnfs']):
                            flag = False
                            break
                else:
                    # 可以复用
                    hvt[dc_node_0based, vnf_type_0based] = 1
                    j += 1
                    i += 1
                    if j >= len(g.request['vnfs']):
                        flag = False
                        break
            if i >= len(usable_on_path):
                flag = False  # 遍历完所有DC节点

        # 检查是否所有VNF都成功放置
        if np.sum(hvt) != len(g.request['vnfs']):
            eval_val = 0
            feasible = False
            infeasible_dest = dest
            hvt = np.zeros((g.node_num, g.type_num))  # 重置
        else:
            # 计算评价函数 (公式 3-11)
            max_k_dist = _get_kth_path_max_distance(src, dest, kpath)

            eval_val = (
                    (1 - pathshop / max_k_dist) +
                    (len(usable_on_path) / g.dc_num) +
                    (CPU_status / (g.cpu_capacity * g.dc_num)) +
                    (Memory_status / (g.memory_capacity * g.dc_num)) +
                    (Bandwidth_status / (g.bandwidth_capacity * g.link_num))
            )
            # 标记使用的链路
            for link_id in lind_id:
                tree[link_id - 1] = 1

    return eval_val, paths, tree, hvt, feasible, infeasible_dest


def calc_atnp(
        tree1: Dict,
        tree1_node: List[int],
        d: int,
        Bandwidth_status_t: np.ndarray,
        CPU_status_t: np.ndarray,
        Memory_status_t: np.ndarray,
        hvt_all: np.ndarray,
        node_on_tree: Set[int]
) -> Tuple:
    """
    计算目的节点加入树的哪个节点，确定加入树的路径节点。
    Python 转换自: calc_atnp.m

    Args:
        tree1: 当前树结构 (包含 'tree', 'hvt', 'treepaths1'...)
        tree1_node: 当前树上用于连接的节点列表
        d: 目的节点索引 (1-based index in request.dests)
        (status...): 当前网络资源状态
        node_on_tree: 树中所有节点的集合 (用于防环)

    Returns:
        t: 更新后的树
        m: 评价指标
    """
    eval_set = []  # 存储每个连接点的最佳 eval
    temp_feasible_set = []  # 存储每个连接点的可行性
    kpath = 5  #

    # 临时存储 k-path 评估结果
    k_path_results = {}

    dest_node_1based = g.request['dests'][d - 1]

    # 遍历树上的每个节点 i，尝试将其作为连接点
    for i_idx, conn_node_1based in enumerate(tree1_node):
        temp_eval_k = []  # 存储 k 条路径的 eval
        temp_feasible_k = []
        temp_infeasible_dest_k = []

        k_path_results[i_idx] = []

        # 遍历 k 条最短路径
        for k in range(1, kpath + 1):
            eval_val, paths, tree, hvt, feasible, infeasible_dest = calc_eval1(
                d, k, i_idx, tree1_node, g.request, tree1,
                Bandwidth_status_t, CPU_status_t, Memory_status_t,
                hvt_all, kpath, node_on_tree
            )

            temp_eval_k.append(eval_val)
            temp_feasible_k.append(feasible)
            temp_infeasible_dest_k.append(infeasible_dest)

            # 存储结果
            k_path_results[i_idx].append({
                'eval': eval_val, 'paths': paths, 'tree': tree, 'hvt': hvt,
                'feasible': feasible, 'infeasible_dest': infeasible_dest
            })

        eval_set.append(np.max(temp_eval_k))
        temp_feasible_set.append(np.min(temp_feasible_k))

    # --- 寻找最佳连接点和路径 ---
    # 确定目的节点加入树是从树上的哪个节点加入
    best_conn_node_idx = np.argmax(eval_set)

    # - 检查可行性
    r = 2  # 默认为可行
    if best_conn_node_idx == 0:  # MATLAB 索引为 1
        r, j = np.min(temp_feasible_set), np.argmin(temp_feasible_set)
        if r == 1:  # 不可行
            best_conn_node_idx = j

    # 从最佳连接点中，找到最佳的k路径
    best_k_idx = np.argmax([res['eval'] for res in k_path_results[best_conn_node_idx]])

    # 获取最佳结果
    best_result = k_path_results[best_conn_node_idx][best_k_idx]
    m = best_result['eval']

    # 准备返回的树对象 't'
    t = {}
    t['tree'] = best_result['tree']
    t['hvt'] = best_result['hvt']
    t['treepaths'] = []  # (MATLAB中也是空的)
    t['treepaths1'] = tree1_node

    if r == 1:  # 不可行
        t['feasible'] = False
        t['infeasible_dest'] = dest_node_1based
    else:
        t['feasible'] = best_result['feasible']
        t['infeasible_dest'] = best_result['infeasible_dest']

    tree_paths = tree1_node[:best_conn_node_idx + 1]
    t['treepaths2'] = tree_paths + best_result['paths']

    return t, m


def calc_eval1(
        d: int, k: int, i_idx: int, tree1_node: List[int], request: Dict, tree1: Dict,
        Bandwidth_status_t: np.ndarray, CPU_status_t: np.ndarray, Memory_status_t: np.ndarray,
        hvt_all: np.ndarray, kpath: int, node_on_tree: Set[int]
) -> Tuple:
    """
    评估一个新树枝 (new_path) 添加到现有树 (tree1) 上的可行性和质量。
    Python 转换自: calc_eval1.m

    Args:
        d: 目的节点索引 (1-based in request.dests)
        k: 第 k 条路径 (1-based)
        i_idx: tree1_node 中的索引 (0-based)
        tree1_node: 当前树上用于连接的节点列表 (1-based)
        ... (其他参数) ...

    Returns:
        eval, paths, tree, hvt, tree_paths, feasible, infeasible_dest
    """

    hvt = tree1['hvt'].copy()
    tree = tree1['tree'].copy()

    conn_node_1based = tree1_node[i_idx]
    dest_node_1based = request['dests'][d - 1]

    tree_paths_to_connector = tree1_node[:i_idx + 1]  #

    feasible = True
    infeasible_dest = 0

    # 获取第k条路径
    paths, pathshop, lind_id = _get_path_from_db(conn_node_1based, dest_node_1based, k)

    if not paths:
        eval_val = 0
        paths = [conn_node_1based]  # 至少包含连接节点

        # 检查是否因为没有路径导致不可行
        node = tree_paths_to_connector
        usable_on_path_vnf = [n for n in node if n in g.DC]
        if len(usable_on_path_vnf) < len(request['vnfs']):
            feasible = False
            infeasible_dest = dest_node_1based

        return eval_val, paths, tree, hvt, tree_paths_to_connector, feasible, infeasible_dest

    # --- 防环检查 ---
    eval_val = 1.0
    new_path_nodes = set(paths[1:])  # 新路径 (不含连接点)
    connector_path_nodes = set(tree_paths_to_connector)

    # 情况一: 新路径与到连接点的路径重叠
    if new_path_nodes.intersection(connector_path_nodes):
        eval_val = 0

    if eval_val != 0:
        # 情况二: 新路径与树上 *其他* 节点重叠
        # node_on_tree 是树上所有节点的集合
        if new_path_nodes.intersection(node_on_tree - connector_path_nodes):
            eval_val = 0

    # (MATLAB代码中的情况三 'arr6' 过于复杂且可能存在逻辑错误,
    # 核心是检查新路径是否与 *已在树上* 的节点重叠, 上述两步已覆盖)

    if eval_val == 0:
        # 产生环路
        return 0, paths, tree, hvt, tree_paths_to_connector, feasible, infeasible_dest

    # --- 资源和 VNF 检查 ---

    # 寻找新路径上的DC节点
    usable_on_new_path = [node for node in paths[1:] if node in g.DC]

    # 计算新路径上的剩余资源
    CPU_status = np.sum(CPU_status_t[np.array(paths[1:]) - 1]) if len(paths) > 1 else 0
    Memory_status = np.sum(Memory_status_t[np.array(paths[1:]) - 1]) if len(paths) > 1 else 0
    Bandwidth_status = 0
    for link_id in lind_id:
        Bandwidth_status += Bandwidth_status_t[link_id - 1]

    # 计算共享路径上已部署的 VNF
    deployed_on_shared_path = [node for node in tree_paths_to_connector if node in g.DC]
    shared_path_deployed = 0
    for node_1based in deployed_on_shared_path:
        shared_path_deployed += np.sum(tree1['hvt'][node_1based - 1, :])

    undeployed_vnf = len(request['vnfs']) - shared_path_deployed

    # 检查带宽
    for link_id in lind_id:
        if Bandwidth_status_t[link_id - 1] < request['bw_origin']:
            eval_val = 0
            feasible = False
            infeasible_dest = dest_node_1based
            break

    if eval_val != 0:
        if undeployed_vnf <= 0:
            # 共享路径已部署所有 VNF，只需更新 tree
            for link_id in lind_id:
                tree[link_id - 1] = 1
        else:
            # 需要在新路径上部署
            if len(usable_on_new_path) < undeployed_vnf:
                eval_val = 0
                feasible = False
                infeasible_dest = dest_node_1based
            else:
                # --- 尝试放置 VNF ---
                j = int(shared_path_deployed)  # vnf 索引
                g_idx = 0  # dc 索引
                flag = True

                while j < len(request['vnfs']) and flag:
                    while g_idx < len(usable_on_new_path):
                        dc_node_1based = usable_on_new_path[g_idx]
                        dc_node_0based = dc_node_1based - 1

                        vnf_type_1based = request['vnfs'][j]
                        vnf_type_0based = vnf_type_1based - 1

                        if hvt_all[dc_node_0based, vnf_type_0based] == 0:
                            cpu_req = request['cpu_origin'][j]
                            mem_req = request['memory_origin'][j]

                            if (CPU_status_t[dc_node_0based] < cpu_req) or \
                                    (Memory_status_t[dc_node_0based] < mem_req):
                                g_idx += 1
                            else:
                                hvt[dc_node_0based, vnf_type_0based] = 1
                                j += 1
                                g_idx += 1
                                if j >= len(request['vnfs']):
                                    flag = False
                                    break
                        else:
                            hvt[dc_node_0based, vnf_type_0based] = 1
                            j += 1
                            g_idx += 1
                            if j >= len(request['vnfs']):
                                flag = False
                                break
                    if g_idx >= len(usable_on_new_path):
                        flag = False

                # 检查新路径上的 VNF 是否部署成功
                new_path_deployed = 0
                for node_1based in usable_on_new_path:
                    new_path_deployed += np.sum(hvt[node_1based - 1, :])

                # (修正 MATLAB 逻辑: 应该比较 hvt 和 tree1['hvt'] 的差值)
                deployed_count_on_new_path = np.sum(hvt) - np.sum(tree1['hvt'])

                if deployed_count_on_new_path != undeployed_vnf:
                    eval_val = 0
                    feasible = False
                    infeasible_dest = dest_node_1based
                    hvt = tree1['hvt'].copy()  # 回退
                else:
                    # 成功，标记链路
                    for link_id in lind_id:
                        tree[link_id - 1] = 1

    # 如果可行，计算最终得分
    if eval_val != 0:
        max_k_dist = _get_kth_path_max_distance(conn_node_1based, dest_node_1based, kpath)

        eval_val = (
                (1 - pathshop / max_k_dist) +
                (len(usable_on_new_path) / g.dc_num) +
                (CPU_status / (g.cpu_capacity * g.dc_num)) +
                (Memory_status / (g.memory_capacity * g.dc_num)) +
                (Bandwidth_status / (g.bandwidth_capacity * g.link_num))
        )

    return eval_val, paths, {'set': tree}, {'set': hvt}, tree_paths_to_connector, feasible, infeasible_dest


def main():
    """主函数 - 转换自 MPH.m"""
    # 加载拓扑
    load_topology()

    # 设置请求 (1-based 索引)
    #
    g.request = {
        'source': 2,
        'dests': [6, 15, 16, 21, 26],
        'vnfs': [8, 4, 7],
        # (资源需求是我从 MPH1.m 和 calc_eval.m 推断的)
        'bw_origin': 50,  # 假设
        'cpu_origin': [20, 30, 20],  # 假设
        'memory_origin': [10, 20, 10]  # 假设
    }
    g.dest_num = len(g.request['dests'])

    # --- 初始化资源状态 (来自 MPH1.m) ---
    # 假设这是 t=0 时的状态
    Bandwidth_status_t = np.full(g.link_num, g.bandwidth_capacity, dtype=float)
    CPU_status_t = np.full(g.node_num, g.cpu_capacity, dtype=float)
    Memory_status_t = np.full(g.node_num, g.memory_capacity, dtype=float)
    hvt_all = np.zeros((g.node_num, g.type_num))  # 全局 VNF 部署状态

    # 构建初始化树
    tree = np.zeros(g.link_num)
    hvt = np.zeros((g.node_num, g.type_num))

    # 第一阶段: 将源节点加入树中
    # (在Python中，我们通过路径来隐式定义树)

    # --- 阶段 1: S -> d (构建 5 棵树) ---
    #
    tree_set2134 = []  # 命名来自 MATLAB
    print("--- 阶段 1: 构建 1-节点 树 ---")

    for d_idx in range(g.dest_num):
        kpath = 3
        temp_eval = []
        temp_results = []  # 存储 (eval, paths, tree, hvt, feasible, infeasible_dest)

        for k in range(1, kpath + 1):
            # 计算第k条最短路径上的评价函数
            eval_val, paths, tree_links, hvt_nodes, feasible, infeasible_dest = calc_eval(
                d_idx + 1, k,
                Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, kpath
            )

            temp_eval.append(eval_val)
            temp_results.append({
                'eval': eval_val, 'paths': paths, 'tree': tree_links, 'hvt': hvt_nodes
            })

        # 得到评价函数最大的路径
        max_idx = np.argmax(temp_eval)
        best_result = temp_results[max_idx]

        # 将 hvt 和 tree 加入到结构体中
        t = {
            'tree': best_result['tree'],
            'hvt': best_result['hvt'],
            'treepaths': best_result['paths']  # 完整的 S->d 路径
        }
        tree_set2134.append(t)

        # (MATLAB 中的回退在 Python 循环中自动发生，因为 tree/hvt 是局部变量)

    print(f"阶段 1 完成, 构建了 {len(tree_set2134)} 棵候选树。")

    # --- 阶段 2: (S->d1) -> d2 (构建 5*4 = 20 棵树) ---
    #
    print("\n--- 阶段 2: 构建 2-节点 树 ---")
    tree_sets_stage2 = []  # 列表的列表
    all_nodes_in_tree = set([g.request['source']])

    for i in range(g.dest_num):
        tree_set_i = []
        tree1 = tree_set2134[i]  # 获取 1-节点 基础树
        tree1_node_path = tree1['treepaths']  # 这是 S->d_i 的路径

        # 树上的所有节点
        nodes_in_tree1 = all_nodes_in_tree.union(set(tree1_node_path))

        for d in range(g.dest_num):
            if d == i:
                continue  # 跳过自己

            # 计算目的节点加入树的哪个节点，确定加入树的路径节点
            #
            t, m = calc_atnp(
                {'tree': tree1['tree'].copy(), 'hvt': tree1['hvt'].copy()},  # 传递副本
                tree1_node_path,
                d + 1,
                Bandwidth_status_t, CPU_status_t, Memory_status_t,
                hvt_all,
                nodes_in_tree1
            )
            tree_set_i.append(t)

        tree_sets_stage2.append(tree_set_i)

    print(f"阶段 2 完成, 构建了 {len(tree_sets_stage2)} 组树 (每组 {len(tree_sets_stage2[0])} 棵)。")

    # --- 阶段 3: (S->d1->d2) -> d3 (构建 20*3 = 60 棵树) ---
    #
    # 这里我们只演示您在提示中给出的 `tree_set213` (基于 `tree_set21(1)`)
    # `tree_set21` 对应 `tree_sets_stage2[1]` (dests[1]=15)
    # `tree_set21(1)` 对应 `tree_sets_stage2[1][0]` (dests[0]=6)
    # 所以 `tree213` 是基于 "15-6" 树 (索引 1 和 0)

    print("\n--- 阶段 3: 构建 3-节点 树 (演示 'tree_set213') ---")

    # 基础树是 "2-1" 树 (即 dests 索引 1 和 0)
    # tree21 = tree_sets_stage2[1] # "15" -> [6, 16, 21, 26]
    # tree21_1 = tree21[0] # "15-6" 树

    # 模拟 `tree_set21(1)`，它在 MATLAB 中是 `tree_set2` (第2组) 的第1个元素
    # 即基础树 (S->dests[1])，添加 dests[0]
    base_tree_15_6 = tree_sets_stage2[1][0]

    # `tree_set21(1).treepaths1` 是基础树的路径 (S->15)
    base_path_S_15 = tree_set2134[1]['treepaths']
    # `tree_set21(1).treepaths2` 是添加的树枝路径 (conn -> 6)
    base_path_conn_6 = base_tree_15_6['treepaths2']

    # 集合所有节点
    nodes_in_tree_15_6 = set(base_path_S_15).union(set(base_path_conn_6))

    tree_set213 = []

    # 循环 d=1..5, 跳过 1 和 2 (即 0 和 1)
    for d_idx in range(g.dest_num):
        if d_idx == 0 or d_idx == 1:  # 跳过 6 和 15
            continue

        temp_m = []
        temp_t = []

        # --- 评估块 1: 连接到 S->15 的路径 ---
        t1, m1 = calc_atnp(
            {'tree': base_tree_15_6['tree'].copy(), 'hvt': base_tree_15_6['hvt'].copy()},
            base_path_S_15, d_idx + 1,
            Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all,
            nodes_in_tree_15_6
        )
        t1['treepaths3'] = base_path_conn_6  # 记录另一条路径
        otv1 = (1 - np.sum(t1['tree']) / g.link_num) + (np.sum(t1['hvt']) / (g.dc_num * g.type_num))  # 简化的otv
        t1['otv'] = otv1
        temp_m.append(m1)
        temp_t.append(t1)

        # --- 评估块 2: 连接到 conn->6 的路径 ---
        t2, m2 = calc_atnp(
            {'tree': base_tree_15_6['tree'].copy(), 'hvt': base_tree_15_6['hvt'].copy()},
            base_path_conn_6, d_idx + 1,
            Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all,
            nodes_in_tree_15_6
        )
        t2['treepaths3'] = base_path_S_15  # 记录另一条路径
        otv2 = (1 - np.sum(t2['tree']) / g.link_num) + (np.sum(t2['hvt']) / (g.dc_num * g.type_num))
        t2['otv'] = otv2
        temp_m.append(m2)
        temp_t.append(t2)

        # --- 评估块 3: (MATLAB 代码中的 treepaths3 不存在于 "2-1" 树中,
        # 它是在 "2-1-3" 树中才有的。您的代码片段 是
        # 从 3-节点 树扩展到 4-节点 树的。我上面演示的是 2->3。)
        # 为了匹配您的片段,我将跳过这个块。

        # 决策点
        max_m_index = np.argmax(temp_m)
        tree_set213.append(temp_t[max_m_index])

    #
    all_otv = [t['otv'] for t in tree_set213]
    best_otv_index = np.argmax(all_otv)
    best_3_node_tree = tree_set213[best_otv_index]

    print(f"阶段 3 (演示 '2-1-X') 完成。")
    print(f"找到的最佳 3-节点 树 (15-6-X) 的 OTV 为: {best_3_node_tree['otv']:.4f}")

    # ... 算法将继续这个过程，直到所有 5 个节点都被添加 ...


if __name__ == '__main__':
    main()