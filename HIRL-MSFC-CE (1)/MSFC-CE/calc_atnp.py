# calc_atnp.py
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
    from calc_eval1 import calc_eval1

    if tree1.eval == 0:
        t = tree1._replace(treepaths=[], feasible=0)
        return 0.0, t

    eval_set = []
    temp_feasible_set = []
    kpath = 5

    for i in range(len(tree1_node)):
        temp_eval = np.zeros(kpath)
        temp_feasible = np.zeros(kpath, dtype=bool)
        for k in range(1, kpath + 1):
            eval_val, _, _, _, _, feasible, _ = calc_eval1(
                d, k, i + 1, tree1_node, request, tree1,
                Bandwidth_status_t, CPU_status_t, Memory_status_t,
                hvt_all, kpath, node_on_tree)
            temp_eval[k - 1] = eval_val
            temp_feasible[k - 1] = feasible
        eval_set.append(np.max(temp_eval))
        temp_feasible_set.append(1 if np.any(temp_feasible) else 0)

    if not eval_set: return 0.0, tree1._replace(feasible=0)

    eval_set = np.array(eval_set)
    temp_feasible_set = np.array(temp_feasible_set)
    y = int(np.argmax(eval_set))

    r = 2
    if y == 0:
        j = int(np.argmin(temp_feasible_set))
        r = temp_feasible_set[j]
        i_selected = j
    else:
        i_selected = y

    temp_eval = np.zeros(kpath)
    temp_tree = [None] * kpath
    temp_hvt = [None] * kpath
    temp_feasible = [False] * kpath
    temp_infeasible_dest = [0] * kpath
    tree_paths_final = []

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
        tree_paths_final = t_paths

    p = int(np.argmax(temp_eval))
    m = temp_eval[p]

    # 路径合并 logic
    best_path_segment = []
    # calc_eval1 返回的 paths 已经是 0-based 列表
    # tree_paths_final 是树上的路径
    # 我们需要的是新加入的那一段。calc_eval1 返回的 paths 是完整路径
    # 实际上 MATLAB 逻辑是把 paths 直接加进去。
    # 但 Python 列表中 paths 是一维的。
    # 我们假设 calc_atnp 需要将新的一段路径记录下来。
    # MATLAB: t.treepaths2=[tree_paths(1:x-1),best_paths.set];
    # 这里简单化：直接存储新路径

    # 从 temp_paths 中取
    # calc_eval1 返回 (eval, paths, ...)
    # 这里 paths 就是 list
    # 注意：上面的循环没有把 paths 存入 temp_paths 列表，这里修复
    # 重新调用一次获取 paths，或者在循环里存下来
    # 为了代码简洁，我们在循环里用 temp_tree 存了，但 paths 没存。
    # 让我们修正循环：

    # 重新获取 p 的 paths
    _, best_paths, _, _, _, _, _ = calc_eval1(
        d, p + 1, i_selected + 1, tree1_node, request, tree1,
        Bandwidth_status_t, CPU_status_t, Memory_status_t,
        hvt_all, kpath, node_on_tree)

    # 拼接
    combined = tree_paths_final[:-1] + best_paths if tree_paths_final else best_paths

    final_feasible = 0 if r == 0 else (1 if temp_feasible[p] else 0)
    final_inf = request.dest[d] if r == 0 else temp_infeasible_dest[p]

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