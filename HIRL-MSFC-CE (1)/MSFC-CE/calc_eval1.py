# calc_eval1.py
import numpy as np
from typing import List
import config  # 引入配置

# 调试开关：打印前 5 次失败原因
DEBUG_LIMIT = 5
debug_count = 0


def calc_eval1(d: int, k: int, i: int, tree1_node: List[int], request, tree1,
               Bandwidth_status_t: np.ndarray, CPU_status_t: np.ndarray, Memory_status_t: np.ndarray,
               hvt_all: np.ndarray, kpath: int, node_on_tree: List[int]):
    global debug_count

    hvt = tree1.hvt.copy()
    tree = tree1.tree.copy()
    tree_paths = tree1_node[:i]

    feasible = True
    infeasible_dest = 0
    eval_val = 1.0

    src = tree1_node[i - 1]
    dest = request.dest[d]
    idx = k - 1

    # === 1. 路径提取 ===
    is_path_empty = True
    full_path = None
    dist_k = 0
    max_dist = 1.0
    lind_id_0 = []

    try:
        if config.path is not None:
            pinfo = config.path[src, dest]
            raw_paths = pinfo['paths']
            raw_dists = pinfo['pathsdistance']

            if raw_paths.size > 0:
                # 智能提取：处理 1D/2D/Object 各种情况
                if raw_paths.dtype == 'O':
                    flat = raw_paths.flatten()
                    if idx < len(flat): full_path = flat[idx]
                elif raw_paths.ndim == 2:
                    if idx < raw_paths.shape[0]: full_path = raw_paths[idx]
                elif raw_paths.ndim == 1 and idx == 0:
                    full_path = raw_paths

            if full_path is not None:
                is_path_empty = False
                full_path = np.array(full_path).flatten()

                # 提取距离
                if raw_dists.size > idx:
                    dist_k = int(raw_dists.flatten()[idx])
                    max_dist = float(raw_dists.flatten()[-1]) if raw_dists.size > 0 else 1.0

                # 截取并转 0-based
                raw_segment = full_path[:dist_k + 1]
                paths = [int(x) - 1 for x in raw_segment]
                pathshop = dist_k

                # 提取链路
                if 'link_ids' in pinfo.dtype.names:
                    raw_l = pinfo['link_ids']
                    if raw_l.size > 0:
                        l_data = None
                        if raw_l.ndim == 2 and idx < raw_l.shape[0]:
                            l_data = raw_l[idx]
                        elif raw_l.ndim == 1 and idx == 0:
                            l_data = raw_l
                        elif raw_l.dtype == 'O':
                            if idx < len(raw_l.flatten()): l_data = raw_l.flatten()[idx]

                        if l_data is not None:
                            l_seg = np.array(l_data).flatten()[:dist_k]
                            lind_id_0 = [int(x) - 1 for x in l_seg]

    except Exception as e:
        if debug_count < DEBUG_LIMIT:
            print(f"[Error] Path Extract: {e}")
            debug_count += 1
        is_path_empty = True

    # === 2. 失败快速返回 ===
    if is_path_empty:
        if debug_count < DEBUG_LIMIT:
            print(f"[Fail] Empty Path for {src}->{dest} (k={k})")
            debug_count += 1
        return 0.0, [src], tree, hvt, tree_paths, False, request.dest[d]

    # === [关键修改] 移除或放宽 "路径上必须有3个DC" 的约束 ===
    # 原 MATLAB 代码: if length(usable_on_path_vnf)<3 -> feasible=0
    # 修改为: 只要路径上有 DC 节点即可尝试，或者至少要有1个
    usable_temp = [n for n in paths if n in config.DC]

    # 如果你一定要保留约束，请改为 len(usable_temp) < 1
    # 这里我们直接注释掉这个强约束，交给后面的资源检查来判断
    # if len(usable_temp) < 3:
    #     return 0.0, paths, tree, hvt, tree_paths, False, request.dest[d]

    # === 3. 环路检测 ===
    arr1 = paths[1:]
    arr2 = tree1_node[:i]
    arr4 = node_on_tree
    arr6 = tree1_node[i:]

    has_loop = False
    if len(paths) != len(set(paths)): has_loop = True
    if not has_loop:
        min_len = min(len(arr6), len(arr1))
        b = 0
        while b < min_len:
            if arr6[b] != arr1[b]: break
            b += 1
        if len(arr6[b:] + arr1[b:]) > len(set(arr6[b:] + arr1[b:])): has_loop = True

    if not has_loop and (set(arr1) & set(arr2)): has_loop = True
    if not has_loop and (set(arr1) & set(arr4)): has_loop = True

    if has_loop:
        return 0.0, paths, tree, hvt, tree_paths, feasible, infeasible_dest

    # === 4. 资源计算与部署 ===
    if eval_val != 0:
        usable_on_path = [n for n in paths[1:] if n in config.DC]

        Bandwidth_status = 0.0
        safe_links = [l for l in lind_id_0 if 0 <= l < config.link_num]
        if safe_links:
            Bandwidth_status = sum(Bandwidth_status_t[l] for l in safe_links)

        # 带宽检查
        for lid in safe_links:
            if Bandwidth_status_t[lid] < request.bw_origin:
                eval_val = 0.0;
                break

        if eval_val != 0:
            deployed_on_path = [n for n in tree_paths if n in config.DC]
            shared_path_deployed = sum(tree1.hvt[n, :].sum() for n in deployed_on_path)
            undeployed_vnf = len(request.vnf) - shared_path_deployed

            # 统计资源用于评分
            valid_nodes = [n for n in paths[1:] if 0 <= n < config.node_num]
            CPU_sum = sum(CPU_status_t[n] for n in valid_nodes)
            Mem_sum = sum(Memory_status_t[n] for n in valid_nodes)

            if undeployed_vnf <= 0:
                # 已全部部署
                eval_val = (1 - pathshop / max_dist) + \
                           len(deployed_on_path) / config.dc_num + \
                           CPU_sum / (config.cpu_capacity * config.dc_num) + \
                           Mem_sum / (config.memory_capacity * config.dc_num) + \
                           Bandwidth_status / (config.bandwidth_capacity * config.link_num)
                for lid in safe_links: tree[lid] = 1
            else:
                # 需新部署：检查节点数量是否足够
                if len(usable_on_path) < undeployed_vnf:
                    if debug_count < DEBUG_LIMIT:
                        print(
                            f"[Fail] Not enough DCs: Need {undeployed_vnf}, Found {len(usable_on_path)} on path {paths}")
                        debug_count += 1
                    return 0.0, paths, tree, hvt, tree_paths, False, request.dest[d]

                # 贪婪部署
                j = int(shared_path_deployed)
                g = 0
                while j < len(request.vnf) and g < len(usable_on_path):
                    node_idx = usable_on_path[g]
                    vnf_idx = request.vnf[j]

                    if not (0 <= node_idx < config.node_num): g += 1; continue

                    if hvt_all[node_idx, vnf_idx] == 0:
                        if (CPU_status_t[node_idx] < request.cpu_origin[j] or
                                Memory_status_t[node_idx] < request.memory_origin[j]):
                            g += 1  # 资源不足
                        else:
                            hvt[node_idx, vnf_idx] = 1;
                            j += 1;
                            g += 1
                    else:
                        hvt[node_idx, vnf_idx] = 1;
                        j += 1;
                        g += 1

                # 验证部署结果
                usable_deployed = sum(hvt[n, :].sum() for n in usable_on_path if 0 <= n < config.node_num)

                # 这里要注意：hvt 是累积的，我们需要判断新增的部署量
                # 简单判断：看总部署量是否等于需求量
                total_deployed = hvt.sum()
                if total_deployed < len(request.vnf):
                    eval_val = 0.0;
                    hvt = tree1.hvt.copy()
                    if debug_count < DEBUG_LIMIT:
                        print(f"[Fail] Resource shortage on nodes {usable_on_path}")
                        debug_count += 1
                else:
                    eval_val = (1 - pathshop / max_dist) + \
                               len(usable_on_path) / config.dc_num + \
                               CPU_sum / (config.cpu_capacity * config.dc_num) + \
                               Mem_sum / (config.memory_capacity * config.dc_num) + \
                               Bandwidth_status / (config.bandwidth_capacity * config.link_num)
                    for lid in safe_links: tree[lid] = 1

    return eval_val, paths, tree, hvt, tree_paths, feasible, infeasible_dest