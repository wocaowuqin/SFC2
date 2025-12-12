# calc_eval.py
import numpy as np
import config


def calc_eval(request, d: int, k: int,
              Bandwidth_status_t: np.ndarray, CPU_status_t: np.ndarray, Memory_status_t: np.ndarray,
              kpath: int, hvt_all: np.ndarray):
    tree = np.zeros(config.link_num, dtype=int)
    hvt = np.zeros((config.node_num, config.type_num), dtype=int)
    src = request.source;
    dest = request.dest[d];
    idx = k - 1

    is_valid = False
    try:
        if config.path is not None:
            pinfo = config.path[src, dest]
            raw_paths = pinfo['paths']

            if raw_paths.size > 0:
                full_path = None
                if raw_paths.dtype == 'O':
                    flat = raw_paths.flatten()
                    if idx < len(flat): full_path = flat[idx]
                elif raw_paths.ndim == 2 and idx < raw_paths.shape[0]:
                    full_path = raw_paths[idx]
                elif raw_paths.ndim == 1 and idx == 0:
                    full_path = raw_paths

                if full_path is not None:
                    is_valid = True
                    # [修复] 强制转为数组
                    full_path = np.array(full_path).flatten()

                    raw_dists = pinfo['pathsdistance'].flatten()
                    dist_k = int(raw_dists[idx]) if idx < len(raw_dists) else 0
                    max_dist = float(raw_dists[-1]) if raw_dists.size > 0 else 1.0

                    # [修复] 截取并转 0-based
                    raw_seg = full_path[:dist_k + 1]
                    paths = [int(x) - 1 for x in raw_seg]
                    pathshop = dist_k

                    lind_id_0 = []
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
    except:
        is_valid = False

    if not is_valid: return 0.0, [], tree, hvt, False, request.dest[d]

    # 资源检查
    usable_on_path = [n for n in paths if n in config.DC]
    safe_links = [l for l in lind_id_0 if 0 <= l < config.link_num]
    Bandwidth_status = sum(Bandwidth_status_t[l] for l in safe_links) if safe_links else 0
    valid_nodes = [n for n in paths if 0 <= n < config.node_num]
    CPU_status = sum(CPU_status_t[n] for n in valid_nodes)
    Memory_status = sum(Memory_status_t[n] for n in valid_nodes)

    eval_val = 1.0;
    feasible = True;
    infeasible_dest = 0

    if len(usable_on_path) < len(request.vnf):
        eval_val = 0.0;
        feasible = False;
        infeasible_dest = request.dest[d]

    if eval_val != 0:
        for lid in safe_links:
            if Bandwidth_status_t[lid] < request.bw_origin: eval_val = 0.0; break

    if eval_val != 0:
        j = 0;
        i = 0
        while j < len(request.vnf) and i < len(usable_on_path):
            node = int(usable_on_path[i])
            vnf = int(request.vnf[j])
            if not (0 <= node < config.node_num): i += 1; continue
            if hvt_all[node, vnf] == 0:
                if (CPU_status_t[node] < request.cpu_origin[j] or
                        Memory_status_t[node] < request.memory_origin[j]):
                    i += 1
                else:
                    hvt[node, vnf] = 1; j += 1; i += 1
            else:
                hvt[node, vnf] = 1; j += 1; i += 1

        if hvt.sum() != len(request.vnf):
            eval_val = 0.0; hvt[:] = 0
        else:
            eval_val = (1 - pathshop / max_dist) + \
                       len(usable_on_path) / config.dc_num + \
                       CPU_status / (config.cpu_capacity * config.dc_num) + \
                       Memory_status / (config.memory_capacity * config.dc_num) + \
                       Bandwidth_status / (config.bandwidth_capacity * config.link_num)
            for lid in safe_links: tree[lid] = 1

    return eval_val, paths, tree, hvt, feasible, infeasible_dest