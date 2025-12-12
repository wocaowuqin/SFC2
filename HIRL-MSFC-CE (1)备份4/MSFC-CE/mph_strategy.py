# mph_strategy.py (修正版)
import numpy as np
from calc_atnp import calc_atnp, Tree
import config


def calculate_otv(tree):
    """计算优化目标值"""
    link_count = np.sum(tree.tree > 0)  # 🔥 修复：只统计使用的链路
    vnf_count = np.sum(tree.hvt > 0)  # 🔥 修复：只统计部署的VNF
    # 归一化：链路数/总链路数 + VNF数/总VNF类型数
    return (1.0 - link_count / 45.0) + (vnf_count / 8.0)


def serve_request_mph(event_id, request, Bandwidth_status, CPU_status, Memory_status, hvt_all,
                      node_num, link_num, type_num):
    """
    MPH策略主函数

    返回:
        request: 请求对象
        Bandwidth_status: 更新后的带宽状态
        CPU_status: 更新后的CPU状态
        Memory_status: 更新后的内存状态
        hvt_all: 更新后的VNF部署状态
        res_req: 成功时返回request，失败返回None
        res_tree: 成功时返回tree对象，失败返回空数组
        bw_comp: 带宽消耗
        cpu_comp: CPU消耗
        mem_comp: 内存消耗
        success: 是否成功
    """

    # 🔥 调试输出
    print(f"\n[MPH_STRATEGY] Processing Request {event_id}")
    print(f"  Source: {request.source}, Dests: {request.dest}")
    print(f"  VNFs: {request.vnf}")
    print(f"  Available CPU (min): {CPU_status.min():.2f}")
    print(f"  Available BW (min): {Bandwidth_status.min():.2f}")

    # 🔥 检查全局配置
    if not hasattr(config, 'path'):
        print(f"  ❌ ERROR: config.path not accessible!")
        return request, Bandwidth_status, CPU_status, Memory_status, hvt_all, \
            None, np.zeros(link_num), 0, 0, 0, False

    if not hasattr(config, 'DC') or len(config.DC) == 0:
        print(f"  ❌ ERROR: config.DC not defined or empty!")
        return request, Bandwidth_status, CPU_status, Memory_status, hvt_all, \
            None, np.zeros(link_num), 0, 0, 0, False

    print(f"  DC nodes: {len(config.DC)} available")

    # 初始化树结构
    current_tree = Tree(
        tree=np.zeros(config.link_num, dtype=int),
        hvt=np.zeros((config.node_num, config.type_num), dtype=int),
        treepaths=[],
        treepaths1=[request.source],
        treepaths2=[request.source],
        feasible=1,
        infeasible_dest=0,
        eval=1.0
    )

    tree_nodes = [request.source]
    remaining_indices = list(range(len(request.dest)))
    success = True

    # 🔥 逐个目的节点添加到树中
    iteration = 0
    while remaining_indices:
        iteration += 1
        print(f"  [Iteration {iteration}] Remaining dests: {[request.dest[i] for i in remaining_indices]}")

        candidates = []
        for d_idx in remaining_indices:
            dest_node = request.dest[d_idx]
            print(f"    Evaluating dest {dest_node} (idx={d_idx})...")

            try:
                # 调用calc_atnp计算将该目的节点加入树的最优方式
                m_val, temp_tree = calc_atnp(
                    current_tree, tree_nodes, d_idx, request,
                    Bandwidth_status, CPU_status, Memory_status,
                    hvt_all, tree_nodes
                )

                print(f"      Result: feasible={temp_tree.feasible}, m_val={m_val:.4f}")

                if temp_tree.feasible == 1:
                    otv = calculate_otv(temp_tree)
                    candidates.append({
                        'd_idx': d_idx,
                        'tree': temp_tree,
                        'otv': otv,
                        'm_val': m_val
                    })
                    print(f"      ✓ Added to candidates, OTV={otv:.4f}")
                else:
                    print(f"      ✗ Not feasible")

            except Exception as e:
                print(f"      ❌ Exception: {e}")
                import traceback
                traceback.print_exc()

        # 检查是否有可行候选
        if not candidates:
            print(f"  ❌ No feasible candidates found! Blocking request.")
            success = False
            break

        # 选择OTV最大的候选
        best_candidate = max(candidates, key=lambda x: x['otv'])
        current_tree = best_candidate['tree']
        best_dest = request.dest[best_candidate['d_idx']]

        # 🔥 关键修复：保持eval非零，否则下一次迭代会失败
        # 使用namedtuple的_replace方法更新eval字段
        current_tree = current_tree._replace(eval=best_candidate['m_val'] if best_candidate['m_val'] > 0 else 1.0)

        print(f"  ✓ Selected dest {best_dest} with OTV={best_candidate['otv']:.4f}")
        print(f"    Tree eval after update: {current_tree.eval}")

        # 更新树节点集合
        for node in current_tree.treepaths2:
            if node not in tree_nodes:
                tree_nodes.append(node)

        remaining_indices.remove(best_candidate['d_idx'])

    # 计算资源消耗
    bw_comp = 0
    cpu_comp = 0
    mem_comp = 0

    if success:
        print(f"  ✓ SUCCESS! Deploying resources...")

        # 🔥 修复1: 带宽分配（占用使用的链路）
        occupied_links = (current_tree.tree > 0)
        Bandwidth_status[occupied_links] -= request.bw_origin
        bw_comp = np.sum(occupied_links) * request.bw_origin

        print(f"    Links occupied: {np.sum(occupied_links)}")
        print(f"    BW consumed: {bw_comp:.2f}")

        # 🔥 修复2: CPU/内存分配（部署VNF到节点）
        deployed_nodes, deployed_vnfs = np.where(current_tree.hvt > 0)

        print(f"    VNFs deployed: {len(deployed_nodes)}")

        for node, vnf_type in zip(deployed_nodes, deployed_vnfs):
            # 找到该VNF类型在请求中的索引
            matching_indices = [i for i, v in enumerate(request.vnf) if v == vnf_type]

            if matching_indices:
                idx = matching_indices[0]  # 取第一个匹配

                # 分配资源
                CPU_status[node] -= request.cpu_origin[idx]
                Memory_status[node] -= request.memory_origin[idx]

                cpu_comp += request.cpu_origin[idx]
                mem_comp += request.memory_origin[idx]

                # 更新全局VNF部署状态
                hvt_all[node, vnf_type] += 1

                print(
                    f"      Node {node}: VNF {vnf_type}, CPU={request.cpu_origin[idx]:.2f}, MEM={request.memory_origin[idx]:.2f}")

        print(f"    Total CPU: {cpu_comp:.2f}, MEM: {mem_comp:.2f}")

        # 🔥 修复3: 设置请求ID
        request.id = event_id

        # 🔥 修复4: 返回tree.tree数组而不是Tree对象
        return (request, Bandwidth_status, CPU_status, Memory_status, hvt_all,
                request, current_tree.tree, bw_comp, cpu_comp, mem_comp, True)
    else:
        print(f"  ✗ FAILED! Request blocked.")
        # 失败时返回空数组
        return (request, Bandwidth_status, CPU_status, Memory_status, hvt_all,
                None, np.zeros(link_num), 0, 0, 0, False)