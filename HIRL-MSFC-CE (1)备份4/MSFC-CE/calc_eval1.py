# calc_eval1.py (终极修复版)
import numpy as np
from typing import List
import config

DEBUG_LIMIT = 3  # 🔥 临时开启调试
debug_count = 0


def calc_eval1(d: int, k: int, i: int, tree1_node: List[int], request, tree1,
               Bandwidth_status_t: np.ndarray, CPU_status_t: np.ndarray, Memory_status_t: np.ndarray,
               hvt_all: np.ndarray, kpath: int, node_on_tree: List[int]):
    """
    🔥 终极修复版：正确处理MATLAB路径数据中的填充值
    """
    global debug_count

    # 从当前树继承状态
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
    paths = []
    dist_k = 0
    max_dist = 1.0
    lind_id_0 = []

    try:
        if config.path is not None and src != dest:
            pinfo = config.path[src, dest]

            if 'paths' not in pinfo.dtype.names:
                is_path_empty = True
            else:
                raw_paths = pinfo['paths']

                if raw_paths.size == 0:
                    is_path_empty = True
                else:
                    # 提取路径
                    full_path = None
                    if raw_paths.dtype == 'O':
                        flat = raw_paths.flatten()
                        if idx < len(flat):
                            full_path = flat[idx]
                    elif raw_paths.ndim == 2:
                        if idx < raw_paths.shape[0]:
                            full_path = raw_paths[idx]
                    elif raw_paths.ndim == 1 and idx == 0:
                        full_path = raw_paths

                    if full_path is not None:
                        is_path_empty = False
                        full_path = np.array(full_path).flatten()

                        # 提取距离
                        raw_dists = pinfo['pathsdistance'].flatten()
                        if raw_dists.size > idx:
                            dist_k = int(raw_dists[idx])
                            max_dist = float(raw_dists[-1]) if raw_dists.size > 0 else 1.0

                        # 🔥 关键修复：截取有效部分并过滤负值
                        raw_segment = full_path[:dist_k + 1]

                        # 🔥 过滤负值和0（MATLAB填充值）
                        valid_nodes = []
                        for x in raw_segment:
                            node_val = int(x) - 1  # 转0-based
                            if node_val >= 0:  # 过滤负值
                                valid_nodes.append(node_val)

                        paths = valid_nodes
                        pathshop = len(paths) - 1 if len(paths) > 1 else dist_k  # 🔥 使用实际跳数或原始dist_k

                        # 🔥 修复：如果max_dist为0，使用pathshop+1避免除零
                        if max_dist == 0 or max_dist == 1.0:
                            max_dist = max(dist_k, 1.0)

                        # 提取链路ID
                        if 'link_ids' in pinfo.dtype.names:
                            raw_l = pinfo['link_ids']
                            if raw_l.size > 0:
                                l_data = None
                                if raw_l.ndim == 2 and idx < raw_l.shape[0]:
                                    l_data = raw_l[idx]
                                elif raw_l.ndim == 1 and idx == 0:
                                    l_data = raw_l
                                elif raw_l.dtype == 'O':
                                    flat_l = raw_l.flatten()
                                    if idx < len(flat_l):
                                        l_data = flat_l[idx]

                                if l_data is not None:
                                    l_seg = np.array(l_data).flatten()[:dist_k]
                                    # 🔥 过滤负值
                                    lind_id_0 = [int(x) - 1 for x in l_seg if int(x) > 0]

    except Exception as e:
        if debug_count < DEBUG_LIMIT:
            print(f"[Error] Path Extract {src}->{dest}: {e}")
            debug_count += 1
        is_path_empty = True

    # === 2. 快速失败返回 ===
    if is_path_empty or len(paths) < 2:
        return 0.0, [src], tree, hvt, tree_paths, False, request.dest[d]

    # === 3. 环路检测 ===
    arr1 = paths[1:]
    arr2 = tree1_node[:i]
    arr4 = node_on_tree
    arr6 = tree1_node[i:]

    has_loop = False

    if len(paths) != len(set(paths)):
        has_loop = True

    if not has_loop and len(arr6) > 0:
        min_len = min(len(arr6), len(arr1))
        b = 0
        while b < min_len:
            if arr6[b] != arr1[b]:
                break
            b += 1
        combined = arr6[b:] + arr1[b:]
        if len(combined) > len(set(combined)):
            has_loop = True

    if not has_loop and (set(arr1) & set(arr2)):
        has_loop = True

    if not has_loop and (set(arr1) & set(arr4)):
        has_loop = True

    if has_loop:
        return 0.0, paths, tree, hvt, tree_paths, feasible, infeasible_dest

    # === 4. 资源计算与部署 ===
    if eval_val != 0:
        usable_on_path = [n for n in paths[1:] if n in config.DC]

        # 带宽检查
        safe_links = [l for l in lind_id_0 if 0 <= l < config.link_num]
        Bandwidth_status = sum(Bandwidth_status_t[l] for l in safe_links) if safe_links else 0.0

        for lid in safe_links:
            if Bandwidth_status_t[lid] < request.bw_origin:
                eval_val = 0.0
                break

        if eval_val != 0:
            # 计算树上已部署的VNF数量
            deployed_on_path = [n for n in tree_paths if n in config.DC]
            shared_path_deployed = 0
            for n in deployed_on_path:
                if 0 <= n < config.node_num:
                    shared_path_deployed += tree1.hvt[n, :].sum()

            undeployed_vnf = len(request.vnf) - shared_path_deployed

            # CPU/Memory资源统计
            valid_nodes = [n for n in paths[1:] if 0 <= n < config.node_num]
            CPU_sum = sum(CPU_status_t[n] for n in valid_nodes)
            Mem_sum = sum(Memory_status_t[n] for n in valid_nodes)

            if undeployed_vnf <= 0:
                # 所有VNF已部署
                eval_val = (1 - pathshop / max_dist) + \
                           len(deployed_on_path) / config.dc_num + \
                           CPU_sum / (config.cpu_capacity * config.dc_num) + \
                           Mem_sum / (config.memory_capacity * config.dc_num) + \
                           Bandwidth_status / (config.bandwidth_capacity * config.link_num)

                for lid in safe_links:
                    tree[lid] = 1
            else:
                # 需要新部署VNF
                if len(usable_on_path) < undeployed_vnf:
                    if debug_count < DEBUG_LIMIT:
                        print(f"[Fail] Not enough DCs: Need {undeployed_vnf}, Found {len(usable_on_path)}")
                        debug_count += 1
                    return 0.0, paths, tree, hvt, tree_paths, False, request.dest[d]

                # 贪婪部署VNF
                j = int(shared_path_deployed)
                g = 0

                while j < len(request.vnf) and g < len(usable_on_path):
                    node_idx = usable_on_path[g]
                    vnf_idx = request.vnf[j]

                    if not (0 <= node_idx < config.node_num):
                        g += 1
                        continue

                    if hvt_all[node_idx, vnf_idx] == 0 and hvt[node_idx, vnf_idx] == 0:
                        if (CPU_status_t[node_idx] < request.cpu_origin[j] or
                                Memory_status_t[node_idx] < request.memory_origin[j]):
                            g += 1
                        else:
                            hvt[node_idx, vnf_idx] = 1
                            j += 1
                            g += 1
                    else:
                        hvt[node_idx, vnf_idx] = 1
                        j += 1
                        g += 1

                # 验证部署结果
                total_deployed = hvt.sum()
                required = len(request.vnf)

                if total_deployed < required:
                    eval_val = 0.0
                    hvt = tree1.hvt.copy()
                    if debug_count < DEBUG_LIMIT:
                        print(f"            [Fail] Deployment: deployed={total_deployed}, need={required}")
                        print(f"              Path DCs: {usable_on_path}")
                        print(f"              Shared deployed: {shared_path_deployed}, Undeployed: {undeployed_vnf}")
                        debug_count += 1
                else:
                    eval_val = (1 - pathshop / max_dist) + \
                               len(usable_on_path) / config.dc_num + \
                               CPU_sum / (config.cpu_capacity * config.dc_num) + \
                               Mem_sum / (config.memory_capacity * config.dc_num) + \
                               Bandwidth_status / (config.bandwidth_capacity * config.link_num)

                    if debug_count < DEBUG_LIMIT:
                        print(f"            [Success] eval={eval_val:.4f}")
                        print(f"              path_term={1 - pathshop / max_dist:.4f}")
                        print(f"              dc_term={len(usable_on_path) / config.dc_num:.4f}")
                        debug_count += 1

                    for lid in safe_links:
                        tree[lid] = 1

    return eval_val, paths, tree, hvt, tree_paths, feasible, infeasible_dest