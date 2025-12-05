# calc_atnp.py (完整调试版)
import numpy as np
from collections import namedtuple
from typing import List
import config

Tree = namedtuple('Tree', ['tree', 'hvt', 'treepaths', 'treepaths1', 'treepaths2',
                           'treepaths3', 'treepaths4', 'treepaths5',
                           'feasible', 'infeasible_dest', 'eval', 'dest_seq', 'otv', 'id'])
Tree.__new__.__defaults__ = ([], [], [], [], 1, 0, 0.0, [], 0.0, 0)


def calc_atnp(tree1: Tree, tree1_node: List[int], d: int, request,
              Bandwidth_status_t: np.ndarray, CPU_status_t: np.ndarray, Memory_status_t: np.ndarray,
              hvt_all: np.ndarray, node_on_tree: List[int]):
    """
    计算将目的节点d加入树的最优方式

    🔥 核心问题：tree1_node应该是当前树上的所有节点，而不只是源节点！
    """
    from calc_eval1 import calc_eval1

    print(f"\n      [calc_atnp] Called for dest_idx={d}, dest_node={request.dest[d]}")
    print(f"        tree1_node (nodes in tree): {tree1_node}")
    print(f"        tree1.tree links used: {np.sum(tree1.tree > 0)}")
    print(f"        tree1.hvt VNFs deployed: {np.sum(tree1.hvt > 0)}")
    print(f"        tree1.eval: {tree1.eval}")

    # 检查输入树是否有效
    if tree1.eval == 0:
        print(f"        ⚠️ tree1.eval==0, returning infeasible tree")
        t = tree1._replace(treepaths=[], feasible=0)
        return 0.0, t

    eval_set = []
    temp_feasible_set = []
    kpath = 5

    # 🔥 关键：遍历树上的每个节点，尝试从该节点连接到目的节点
    print(f"        Trying to connect from {len(tree1_node)} tree nodes...")

    for i in range(len(tree1_node)):
        from_node = tree1_node[i]
        to_node = request.dest[d]

        print(f"          [Node {i}] From {from_node} to {to_node}:")

        temp_eval = np.zeros(kpath)
        temp_feasible = np.zeros(kpath, dtype=bool)

        for k in range(1, kpath + 1):
            eval_val, _, _, _, _, feasible, _ = calc_eval1(
                d, k, i + 1, tree1_node, request, tree1,
                Bandwidth_status_t, CPU_status_t, Memory_status_t,
                hvt_all, kpath, node_on_tree)

            temp_eval[k - 1] = eval_val
            temp_feasible[k - 1] = feasible

            if eval_val > 0:
                print(f"            k={k}: eval={eval_val:.4f}, feasible={feasible}")

        max_eval = np.max(temp_eval)
        any_feasible = np.any(temp_feasible)

        print(f"          Best eval for this node: {max_eval:.4f}, any_feasible={any_feasible}")

        eval_set.append(max_eval)
        temp_feasible_set.append(1 if any_feasible else 0)

    if not eval_set:
        print(f"        ✗ No eval_set generated!")
        return 0.0, tree1._replace(feasible=0)

    eval_set = np.array(eval_set)
    temp_feasible_set = np.array(temp_feasible_set)

    print(f"        eval_set: {eval_set}")
    print(f"        feasible_set: {temp_feasible_set}")

    # 选择最佳节点
    y = int(np.argmax(eval_set))

    r = 2
    if y == 0 and eval_set[0] == 0:  # 🔥 修复：如果第一个节点eval=0，检查是否有可行的
        if np.sum(temp_feasible_set) > 0:
            # 有可行的，选择第一个可行的
            i_selected = int(np.argmax(temp_feasible_set))
            r = 1
        else:
            # 没有可行的
            r = 0
            i_selected = 0
    else:
        i_selected = y
        r = 2

    print(f"        Selected node index: {i_selected} (node {tree1_node[i_selected]}), r={r}")

    # 🔥 重新计算选中节点的K条路径
    temp_eval = np.zeros(kpath)
    temp_tree = [None] * kpath
    temp_hvt = [None] * kpath
    temp_feasible = [False] * kpath
    temp_infeasible_dest = [0] * kpath
    temp_paths_list = [None] * kpath

    for k in range(1, kpath + 1):
        eval_val, paths, tree_tmp, hvt_tmp, t_paths, feas, inf_d = calc_eval1(
            d, k, i_selected + 1, tree1_node, request, tree1,
            Bandwidth_status_t, CPU_status_t, Memory_status_t,
            hvt_all, kpath, node_on_tree)

        temp_eval[k - 1] = eval_val
        temp_tree[k - 1] = tree_tmp
        temp_hvt[k - 1] = hvt_tmp
        temp_feasible[k - 1] = feas
        temp_infeasible_dest[k - 1] = inf_d
        temp_paths_list[k - 1] = paths

    p = int(np.argmax(temp_eval))
    m = temp_eval[p]

    print(f"        Best path: k={p + 1}, eval={m:.4f}, feasible={temp_feasible[p]}")

    # 🔥 路径合并
    best_paths = temp_paths_list[p]

    # 🔥 关键修复：过滤负值
    best_paths_cleaned = [n for n in best_paths if n >= 0]

    tree_paths_final = tree1_node[:i_selected + 1]

    print(f"        tree_paths_final: {tree_paths_final}")
    print(f"        best_paths (raw): {best_paths[:10]}...")  # 只显示前10个
    print(f"        best_paths (cleaned): {best_paths_cleaned}")

    # 合并路径：树路径 + 新路径（去掉新路径的起点）
    if len(best_paths_cleaned) > 1:
        combined = tree_paths_final[:-1] + best_paths_cleaned
    else:
        combined = tree_paths_final

    print(f"        combined path: {combined}")

    # 🔥 确定可行性
    if r == 0:
        final_feasible = 0
        final_inf = request.dest[d]
    else:
        final_feasible = 1 if temp_feasible[p] else 0
        final_inf = temp_infeasible_dest[p]

    print(f"        Final: feasible={final_feasible}, eval={m:.4f}")

    # 🔥 创建新树
    t = Tree(
        tree=temp_tree[p],
        hvt=temp_hvt[p],
        treepaths=[],
        treepaths1=tree1_node.copy(),
        treepaths2=combined,
        treepaths3=[], treepaths4=[], treepaths5=[],
        feasible=final_feasible,
        infeasible_dest=final_inf,
        eval=m,
        dest_seq=getattr(tree1, 'dest_seq', []) + [d] if m > 0 else [],
        otv=0.0, id=0
    )

    return m, t