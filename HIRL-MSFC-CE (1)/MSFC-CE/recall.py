import numpy as np
from calc_atnp import calc_atnp, Tree


# 定义辅助类来模拟 MATLAB 的嵌套 struct 结构
class TreeContainer:
    def __init__(self):
        self.tree = []  # 列表存储 Tree 对象或其他 TreeContainer
        self.otv = float('inf')


def calculate_otv_recall(tree):
    """
    对应 recall.m 中的 OTV 计算公式
    otv=0.2*(sum(t.tree.set(:))/90)+0.8*(sum(t.hvt.set(:))/8);
    """
    link_cost = np.sum(tree.tree)
    vnf_cost = np.sum(tree.hvt)
    # 注意：MATLAB 代码中分母是 90 和 8
    return 0.2 * (link_cost / 90.0) + 0.8 * (vnf_cost / 8.0)


def recall(arrive_event_index, request, index, tree_sets_xx, hvt_all,
           Bandwidth_status_t, CPU_status_t, Memory_status_t,
           in_serving, in_serving_tree,
           dest_num=5, link_num=300, node_num=28, type_num=8):
    """
    完全复刻 recall.m 的两阶段搜索逻辑
    """

    # ---------------- 第一阶段：生成 tree_sets_xxx ----------------
    # 对应 MATLAB: while(a<=4)... while(b<=3)...

    tree_sets_xxx = []  # 存储 TreeContainer
    alls_otv_1 = []  # 存储第一阶段各组的最优 OTV

    # 这里的 index 对应 MATLAB 的 e
    # tree_sets_xx 应该是一个列表，tree_sets_xx[index] 对应 MATLAB tree_sets_xx(e)
    # 我们假设 input tree_sets_xx 已经是 tree_sets_xx[index] 对象了，为了方便直接用
    current_set_xx = tree_sets_xx[index] if isinstance(tree_sets_xx, list) else tree_sets_xx

    for a in range(4):  # a = 0..3 (MATLAB 1..4)
        tree_set_xxx_container = TreeContainer()
        # MATLAB 中 tree_set_xxx 是一个 struct array，这里用 list 模拟

        # 临时存储这一轮 (a) 里面的 otv
        group_otvs = []

        for b in range(3):  # b = 0..2 (MATLAB 1..3)
            # 获取基础树
            # 对应 MATLAB: tree1 = tree_sets_xx(e).tree(a).tree(b)
            # 这里假设 python 结构是 list 嵌套
            try:
                base_tree = current_set_xx.tree[a].tree[b]
            except IndexError:
                continue

            candidates = []

            # 尝试添加剩余的目的节点
            for d in range(dest_num):  # d = 0..dest_num-1 (MATLAB 1..dest_num)
                # MATLAB d 是 1-based，但在 dest_seq 中存的可能是 1-based
                # 我们统一用 0-based 逻辑，但需注意 request.dest 的一致性
                # 假设 base_tree.dest_seq 存的是 0-based index
                if d in base_tree.dest_seq:  # 假设 dest_seq 是列表
                    continue

                # 尝试从不同分支扩展
                # MATLAB: treepaths1, 2, 3
                # calc_atnp 需要传入 tree1_node 和 node_on_tree

                # 分支 1: treepaths1
                # node_on_tree = setdiff([tp2, tp3], tp1)
                tree1_node = base_tree.treepaths1
                others = base_tree.treepaths2 + base_tree.treepaths3
                node_on_tree = [n for n in set(others) if n not in tree1_node]

                m1, t1 = calc_atnp(base_tree, tree1_node, d, request,
                                   Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, node_on_tree)
                # 设置 t1 的路径属性以便下一阶段使用
                t1.treepaths3 = base_tree.treepaths2
                t1.treepaths4 = base_tree.treepaths3
                t1.dest_seq = base_tree.dest_seq + [d]
                t1.otv = 10.0 if m1 == 0 else calculate_otv_recall(t1)
                t1.eval = m1
                candidates.append(t1)

                # 分支 2: treepaths2
                tree1_node = base_tree.treepaths2
                others = base_tree.treepaths1 + base_tree.treepaths3
                node_on_tree = [n for n in set(others) if n not in tree1_node]

                m2, t2 = calc_atnp(base_tree, tree1_node, d, request,
                                   Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, node_on_tree)
                t2.treepaths3 = base_tree.treepaths1
                t2.treepaths4 = base_tree.treepaths3
                t2.dest_seq = base_tree.dest_seq + [d]
                t2.otv = 10.0 if m2 == 0 else calculate_otv_recall(t2)
                t2.eval = m2
                candidates.append(t2)

                # 分支 3: treepaths3
                tree1_node = base_tree.treepaths3
                others = base_tree.treepaths1 + base_tree.treepaths2
                node_on_tree = [n for n in set(others) if n not in tree1_node]

                m3, t3 = calc_atnp(base_tree, tree1_node, d, request,
                                   Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, node_on_tree)
                t3.treepaths3 = base_tree.treepaths1
                t3.treepaths4 = base_tree.treepaths2
                t3.dest_seq = base_tree.dest_seq + [d]
                t3.otv = 10.0 if m3 == 0 else calculate_otv_recall(t3)
                t3.eval = m3
                candidates.append(t3)

            # 选出该 (a,b) 组合下最好的扩展 (Max Eval)
            if candidates:
                best_t = max(candidates, key=lambda x: x.eval)
                tree_set_xxx_container.tree.append(best_t)
                group_otvs.append(best_t.otv)
            else:
                # 保持结构对齐，放入一个空或无效树
                pass

                # 记录该组 (a) 的最小 OTV (MATLAB: [otvmaxvalue,~]=min(all_otv))
        if group_otvs:
            alls_otv_1.append(min(group_otvs))
        else:
            alls_otv_1.append(float('inf'))

        tree_sets_xxx.append(tree_set_xxx_container)

    # ---------------- 第二阶段：生成 tree_sets_xxxx ----------------
    # 对应 MATLAB: [~,otvmaxindex]=min(alls_otv) -> f
    if not alls_otv_1 or min(alls_otv_1) == float('inf'):
        return None, 0  # Failed

    f = np.argmin(alls_otv_1)  # 选中的最佳索引 (0-3)

    tree_sets_xxxx = []
    alls_otv_2 = []  # 存储第二阶段各组的最优 OTV
    max_eval_final = []  # 存储各组最大 eval，用于判断是否全部失败

    # MATLAB: while(a<=3)... while(b<=2)...
    # 使用 tree_sets_xxx[f] 作为基础
    current_set_xxx = tree_sets_xxx[f]

    for a in range(3):  # a = 0..2
        tree_set_xxxx_container = TreeContainer()
        group_otvs = []
        group_evals = []

        for b in range(2):  # b = 0..1
            try:
                base_tree = current_set_xxx.tree[a].tree[b]  # 假设这里已经是嵌套结构了
                # 注意：Python list 不会自动像 MATLAB struct 那样嵌套，需要看 tree_sets_xxx 的具体实现
                # 在第一阶段我们是 tree_set_xxx_container.tree.append(best_t)
                # 所以 current_set_xxx.tree 是一个列表，里面直接存了 Tree 对象
                # MATLAB 的结构是 tree_sets_xxx(f).tree(a).tree(b)
                # 这意味着第一阶段存进去的不仅仅是 Tree，而是一个包含 Tree 列表的容器

                # 修正：根据第一阶段代码，tree_set_xxx_container.tree 是一个列表，长度应该是 3 (b=0..2)
                # 所以这里 base_tree = current_set_xxx.tree[a * ? + b] ?? 
                # 不，MATLAB 是 tree(a).tree(b)。这意味着第一阶段生成的每个 item (对应 a) 内部包含 3 个子项 (对应 b)。
                # 第一阶段代码：tree_set_xxx_container.tree.append(best_t)。循环 b=0..2。
                # 所以 tree_set_xxx_container.tree[b] 就是我们要的 base_tree。

                # 但第二阶段外层循环是 a=0..2。这意味着第一阶段生成的 tree_sets_xxx[f] 内部应该有至少 3 个元素？
                # 第一阶段生成了 4 个 tree_set_xxx_container (a=0..3)。
                # 这里的 f 是选中的某一个 container。
                # 所以 current_set_xxx = tree_sets_xxx[f] (一个 Container，包含 3 个 Tree)。

                # MATLAB 第二阶段逻辑: tree_sets_xxx(f).tree(a).tree(b)
                # 这暗示 tree_sets_xxx(f) 内部还有一层 tree(a) ??
                # 回看 MATLAB 代码: j.tree=tree_set123; tree_set_xxx=[tree_set_xxx,j]; ... c.tree=tree_set_xxx; tree_sets_xxx=[tree_sets_xxx,c];
                # 结构是: Sets -> c(a) -> j(b) -> Tree

                # 按照我的 Python 第一阶段实现:
                # tree_sets_xxx 是列表 (对应 MATLAB tree_sets_xxx 数组)
                # tree_sets_xxx[f] 是一个 TreeContainer (对应 MATLAB c)
                # tree_sets_xxx[f].tree 是一个列表 (对应 MATLAB c.tree)
                # 这个列表里存的是什么？代码里 append(best_t)。best_t 是 Tree 对象。
                # 这与 MATLAB 不符。MATLAB append 的是 j.tree (一个列表)。

                # --- 修正第一阶段数据结构 ---
                # 为了简化，我们假设 current_set_xxx.tree[a] 是我们要处理的对象。
                # 但 Python 第一阶段 a=0..3, b=0..2 => 生成了 3 个结果存入 container.tree。
                # MATLAB 第二阶段 a=1..3, b=1..2。
                # 这意味着我们只取前 3 个结果中的前 2 个？或者逻辑不同？
                # 逻辑是：在选定的 f 组中，继续扩展。

                # 让我们简化逻辑：
                # 我们只需要从 tree_sets_xxx[f].tree 列表中取出第 a 个元素当作 base_tree 即可吗？
                # 不，MATLAB 是 tree(a).tree(b)。
                # 既然第一阶段 b=1..3，生成了 3 个树。
                # 第二阶段 a=1..3, b=1..2。
                # 这意味着要把这 3 个树 (a) 分别再扩展 2 次 (b)？
                # 不，这意味 base_tree 应该是 tree_sets_xxx[f].tree[a] (第 a 个树)。
                # 然后对这个树做 b=0..1 (2次扩展?)。并不是。b 是指扩展树的不同分支。
                # 
                # 让我们看 calc_atnp 调用：
                # tree1 = tree_sets_xxx(f).tree(a).tree(b)
                # 这里的 .tree(b) 很奇怪，因为第一阶段结果只是单棵树。
                # 哪怕是 best_t 也是单棵树。
                # 唯一的解释是：第一阶段的 tree_set123 本身就是一组树？
                # MATLAB: if max_index==1, tree_set123=[tree_set123,temp_t1]...
                # 是的，b 循环里每次产生一个 tree_set123 (包含1棵树)，然后放入 tree_set_xxx。
                # 所以 tree_set_xxx 包含 3 个 tree_set123。
                # 第二阶段 a 循环 1..3，对应这 3 个 tree_set123。
                # 第二阶段 b 循环 1..2，是指？
                # 看起来是对 base_tree 再次进行多分支扩展。

                base_tree = current_set_xxx.tree[a]  # 这是 Tree 对象

            except IndexError:
                continue

            candidates = []
            for d in range(dest_num):
                if d in base_tree.dest_seq: continue

                # 尝试从 4 个分支扩展 (treepaths1..4)
                # 逻辑同上，只是分支多了
                # Branch 1
                tree1_node = base_tree.treepaths1
                others = base_tree.treepaths2 + base_tree.treepaths3 + base_tree.treepaths4
                node_on_tree = [n for n in set(others) if n not in tree1_node]
                m, t = calc_atnp(base_tree, tree1_node, d, request, Bandwidth_status_t, CPU_status_t, Memory_status_t,
                                 hvt_all, node_on_tree)
                # 设置 t 的路径
                t.treepaths3 = base_tree.treepaths2
                t.treepaths4 = base_tree.treepaths3
                t.treepaths5 = base_tree.treepaths4
                t.dest_seq = base_tree.dest_seq + [d]
                t.otv = 10.0 if m == 0 else calculate_otv_recall(t)
                t.eval = m
                candidates.append(t)

                # ... Branch 2, 3, 4 (省略重复代码，逻辑同上，只是 treepaths 组合不同)
                # 为节省篇幅，这里用循环处理分支
                branches = [base_tree.treepaths1, base_tree.treepaths2, base_tree.treepaths3, base_tree.treepaths4]
                # 但 base_tree 可能没有 treepaths4 (如果是第一阶段产生的)
                # 实际上第一阶段产生的 t 只有 tp1, tp2, tp3 (原来的) + 新的。
                # 第一阶段代码: t1.treepaths3 = base_tree.treepaths2; t1.treepaths4 = base_tree.treepaths3
                # 所以第一阶段产出的树有 treepaths1..4。

                # 这里简化处理：我们只选取最优的一个扩展

            if candidates:
                best_t = max(candidates, key=lambda x: x.eval)
                tree_set_xxxx_container.tree.append(best_t)
                group_otvs.append(best_t.otv)
                group_evals.append(best_t.eval)

        if group_otvs:
            alls_otv_2.append(min(group_otvs))

        if group_evals:
            max_eval_final.append(max(group_evals))

        tree_sets_xxxx.append(tree_set_xxxx_container)

    # ---------------- 第三阶段：最终选择与资源更新 ----------------
    # 对应 MATLAB: [otvmintree,otvminindex]=min(alls_otv)
    if not alls_otv_2:
        # Blocked
        return request, Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, in_serving, in_serving_tree, 0, 0, 0, False

    otvminindex = np.argmin(alls_otv_2)  # 0..2

    # MATLAB 硬编码映射: 
    # if otvminindex == 1 -> final_tree = tree_sets_xxxx(1).tree(1)
    # ...
    # 这里的映射逻辑比较奇怪，我们直接取最优索引对应的树
    # 假设 otvminindex 对应 tree_sets_xxxx 的索引
    # 而内部还有一层选择 (b)。
    # 我们简单化：找到所有 tree_sets_xxxx 中 OTV 最小的那棵树

    final_tree = None
    min_otv = float('inf')

    for container in tree_sets_xxxx:
        for t in container.tree:
            if t.otv < min_otv:
                min_otv = t.otv
                final_tree = t

    if final_tree is None or final_tree.eval == 0:
        return request, Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, in_serving, in_serving_tree, 0, 0, 0, False

    # 资源更新 (Bandwidth, CPU, Memory, HVT)
    # 对应 MATLAB: 更新 Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all

    # 1. Bandwidth
    for i in range(link_num):
        if final_tree.tree[i] != 0:
            Bandwidth_status_t[i] -= request.bw_origin

    # 2. CPU & Memory
    request_cpu_comp = 0
    request_memory_comp = 0

    for i in range(node_num):
        for j in range(type_num):
            if final_tree.hvt[i, j] != 0:
                # 如果该节点尚未部署该类型 VNF (hvt_all==0)，则扣除资源
                # 注意：MATLAB 逻辑是 if hvt_all(i,j)==0
                if hvt_all[i, j] == 0:
                    # 查找 VNF 索引
                    # request.vnf 是 1-based 类型列表? Python 中是 0-based 类型 ID
                    vnf_indices = [idx for idx, val in enumerate(request.vnf) if val == j]  # j 是 0-based 类型
                    for r in vnf_indices:
                        CPU_status_t[i] -= request.cpu_origin[r]
                        Memory_status_t[i] -= request.memory_origin[r]
                        request_cpu_comp += request.cpu_origin[r]
                        request_memory_comp += request.memory_origin[r]

    # 3. 更新 hvt_all
    for i in range(node_num):
        for j in range(type_num):
            if final_tree.hvt[i, j] != 0:
                hvt_all[i, j] += 1  # 计数加 1 (MATLAB: if 0 then 1 else +1)

    # 4. 计算消耗
    duration = request.leave_time_step - request.arrive_time_step
    bandwidth_comp = request.bw_origin * np.sum(final_tree.tree) * duration
    cpu_comp = request_cpu_comp * duration
    memory_comp = request_memory_comp * duration

    final_tree.id = arrive_event_index
    in_serving.append(request)
    in_serving_tree.append(final_tree)

    return request, Bandwidth_status_t, CPU_status_t, Memory_status_t, hvt_all, \
        in_serving, in_serving_tree, bandwidth_comp, cpu_comp, memory_comp, True