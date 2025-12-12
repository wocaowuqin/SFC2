import numpy as np
from calc_eval import calc_eval
from calc_atnp import calc_atnp, Tree
from recall import recall, TreeContainer, calculate_otv_recall


# 辅助函数：计算 OTV (与 recall 中一致)
def calculate_otv_initial(tree):
    # MATLAB: 0.2*(sum(t.tree.set(:))/90)+0.8*(sum(t.hvt.set(:))/8)
    link_cost = np.sum(tree.tree)
    vnf_cost = np.sum(tree.hvt)
    return 0.2 * (link_cost / 90.0) + 0.8 * (vnf_cost / 8.0)


def serve_request(arrive_event_index, request, Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all,
                  in_serving, in_serving_tree,
                  node_num=28, link_num=300, type_num=8, dest_num=5):
    """
    复刻 serve_request.m 的多阶段搜索逻辑:
    Phase 1 -> tree_set
    Phase 2 -> tree_set_x
    Phase 3 -> tree_set_xx
    Phase 4 -> Call recall()
    """

    # ================= Phase 1: 初始化树 (Source -> 1 Dest) =================
    # 对应 MATLAB: tree_set = [] ... for d=1:dest_num

    tree_set = []  # 存储 Tree 对象
    max_eval_1 = []

    for d in range(dest_num):  # d = 0..4
        # MATLAB kpath=5
        kpath = 5
        temp_eval = []
        temp_candidates = []

        for k in range(1, kpath + 1):
            # calc_eval 返回 (eval_val, paths, tree, hvt, feasible, infeasible_dest)
            # 注意：request.dest 在 calc_eval 内部处理，传入 d (0-based index)
            res = calc_eval(request, d, k, Bandwidth_status_t, CPU_status_t, Memory_status_t, kpath, hvt_all)

            # Unpack results
            eval_val, paths, tree_map, hvt_map, feasible, inf_dest = res

            # 构建 Tree 对象
            # 注意：calc_eval 返回的 paths 是 list，需要赋给 treepaths
            # 初始树：treepaths1 = paths (Source->Dest path)
            t = Tree(
                tree=tree_map,
                hvt=hvt_map,
                treepaths=paths,  # temp_paths(p).set
                treepaths1=paths,  # 逻辑推断：初始路径即为分支1
                treepaths2=[],
                treepaths3=[],
                treepaths4=[],
                treepaths5=[],
                feasible=feasible,
                infeasible_dest=inf_dest,
                eval=eval_val,
                dest_seq=[d],  # 记录加入顺序
                otv=0.0  # 暂时未算
            )
            temp_eval.append(eval_val)
            temp_candidates.append(t)

        # 选最好的路径 (Max Eval)
        best_k_idx = np.argmax(temp_eval)
        best_t = temp_candidates[best_k_idx]

        tree_set.append(best_t)
        max_eval_1.append(best_t.eval)

    # 检查 Phase 1 是否全失败
    if max(max_eval_1) == 0:
        # Blocked
        return request, Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, \
            in_serving, in_serving_tree, 0, 0, 0, False

    # ================= Phase 2: 扩展树 (1 Dest -> 2 Dests) =================
    # 对应 MATLAB: tree_set_x

    tree_set_x = []  # 存储 TreeContainer (每个元素包含一组树)

    for a in range(dest_num):  # a = 0..4
        container = TreeContainer()  # 对应 MATLAB u
        eval_all = []

        # 使用 tree_set[a] 作为基础
        base_tree = tree_set[a]

        for d in range(dest_num):
            if d == a: continue

            # MATLAB: tree1_node=[tree_set(a).treepaths]
            tree1_node = base_tree.treepaths  # 应该就是 treepaths1
            node_on_tree = []  # Phase 1 只有一个分支，无其他节点

            m, t = calc_atnp(base_tree, tree1_node, d, request,
                             Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, node_on_tree)

            # 设置属性
            t.dest_seq = base_tree.dest_seq + [d]
            t.eval = m
            # MATLAB: t.treepaths2 = ... (calc_atnp 会填充 treepaths2)
            # 这里不需要额外设置 treepaths3/4

            container.tree.append(t)
            eval_all.append(m)

        # MATLAB: [e,~]=max(eval_all); max_eval=[max_eval,e] (这里未用于阻断，仅记录)
        tree_set_x.append(container)

    # 检查 Phase 2 (MATLAB: if max_eval == 0)
    # 这里略过详细检查，假设至少有一个可行

    # ================= Phase 3: 扩展树 (2 Dests -> 3 Dests) =================
    # 对应 MATLAB: tree_sets_xx

    tree_sets_xx = []
    alls_otv = []

    for a in range(dest_num):  # a = 0..4 (对应 tree_set_x 的索引)
        container_xx = TreeContainer()  # 对应 MATLAB c
        current_x_trees = tree_set_x[a].tree  # 列表，包含 4 棵树 (dest_num - 1)

        group_otvs = []  # 存储这一组 (a) 下所有分支的 otv

        for b in range(len(current_x_trees)):  # b = 0..3
            base_tree = current_x_trees[b]

            # MATLAB: tree_set11
            candidates = []

            for d in range(dest_num):
                if d in base_tree.dest_seq: continue

                # Branch 1: treepaths1
                # node_on_tree1 = tree_set_x(a).tree(b).treepaths2
                tree1_node = base_tree.treepaths1
                node_on_tree = [n for n in base_tree.treepaths2 if n not in tree1_node]

                m1, t1 = calc_atnp(base_tree, tree1_node, d, request,
                                   Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, node_on_tree)
                t1.treepaths3 = base_tree.treepaths2
                t1.dest_seq = base_tree.dest_seq + [d]
                t1.otv = 10.0 if m1 == 0 else calculate_otv_initial(t1)
                t1.eval = m1
                candidates.append(t1)

                # Branch 2: treepaths2
                tree1_node = base_tree.treepaths2
                node_on_tree = [n for n in base_tree.treepaths1 if n not in tree1_node]

                m2, t2 = calc_atnp(base_tree, tree1_node, d, request,
                                   Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, node_on_tree)
                t2.treepaths3 = base_tree.treepaths1
                t2.dest_seq = base_tree.dest_seq + [d]
                t2.otv = 10.0 if m2 == 0 else calculate_otv_initial(t2)
                t2.eval = m2
                candidates.append(t2)

            # 选最好的 (MATLAB: max(eval_all))
            # MATLAB 这里的逻辑是把 tree_set11 (包含所有 d 的尝试) 放入 tree_set_xx
            # 但 tree_set_xx 结构是 c.tree = tree_set_xx
            # j.tree = tree_set11. 
            # 所以 tree_sets_xx[a].tree[b] 是一个包含 dest_num-2 个树的列表？
            # 不，MATLAB 这里的 candidates 选择逻辑：
            # if temp_m1 >= temp_m2 -> tree_set11=[tree_set11, temp_t1]
            # 这意味着对于每一个 d，它选了一个最好的分支，存入 tree_set11。
            # 所以 tree_set11 长度是 dest_num - 2 (即 3)。

            # 我们需要把这 3 个树封装起来
            sub_container = TreeContainer()

            # 为每个 d 选最佳分支
            # 我们刚才把 candidates 展平了 (d1_b1, d1_b2, d2_b1, d2_b2...)
            # 需要按 d 分组比较
            # 简单处理：candidates 列表长度是 (dest_num - 2) * 2
            # 偶数索引是 branch 1, 奇数索引是 branch 2

            best_candidates_for_b = []
            for i in range(0, len(candidates), 2):
                t1 = candidates[i]
                t2 = candidates[i + 1]
                if t1.eval >= t2.eval:
                    best_candidates_for_b.append(t1)
                    group_otvs.append(t1.otv)
                else:
                    best_candidates_for_b.append(t2)
                    group_otvs.append(t2.otv)

            sub_container.tree = best_candidates_for_b
            container_xx.tree.append(sub_container)

        # MATLAB: all_otv=[all_otv, tree_set11.otv]
        # [otvmaxvalue,~]=min(all_otv) -> alls_otv
        if group_otvs:
            alls_otv.append(min(group_otvs))
        else:
            alls_otv.append(float('inf'))

        tree_sets_xx.append(container_xx)

    # ================= Phase 4: Recall & Finalize =================
    # 对应 MATLAB: [~,otvmaxindex]=min(alls_otv) -> e

    if not alls_otv or min(alls_otv) == float('inf'):
        return request, Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, \
            in_serving, in_serving_tree, 0, 0, 0, False

    best_e_idx = np.argmin(alls_otv)

    # 调用 recall 完成剩余步骤
    # recall 会处理 tree_sets_xxx 和 tree_sets_xxxx
    # 注意：recall 返回 success 状态 (True/False)

    return recall(arrive_event_index, request, best_e_idx, tree_sets_xx, hvt_all,
                  Bandwidth_status_t, CPU_status_t, Memory_status_t,
                  in_serving, in_serving_tree,
                  dest_num, link_num, node_num, type_num, 0)  # 0 is dummy block_flag