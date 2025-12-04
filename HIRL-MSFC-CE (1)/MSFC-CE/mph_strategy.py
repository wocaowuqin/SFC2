# mph_strategy.py
import numpy as np
from calc_atnp import calc_atnp, Tree
import config


def calculate_otv(tree):
    link_count = np.sum(tree.tree)
    vnf_count = np.sum(tree.hvt)
    return (1.0 - link_count / 45.0) + (vnf_count / 8.0)


def serve_request_mph(event_id, request, Bandwidth_status, CPU_status, Memory_status, hvt_all,
                      node_num, link_num, type_num):
    current_tree = Tree(
        tree=np.zeros(config.link_num, dtype=int),
        hvt=np.zeros((config.node_num, config.type_num), dtype=int),
        treepaths=[], treepaths1=[request.source], treepaths2=[request.source],
        feasible=1, infeasible_dest=0, eval=1.0
    )

    tree_nodes = [request.source]
    remaining_indices = list(range(len(request.dest)))
    success = True

    while remaining_indices:
        candidates = []
        for d_idx in remaining_indices:
            m_val, temp_tree = calc_atnp(
                current_tree, tree_nodes, d_idx, request,
                Bandwidth_status, CPU_status, Memory_status,
                hvt_all, tree_nodes
            )
            if temp_tree.feasible == 1:
                otv = calculate_otv(temp_tree)
                candidates.append({'d_idx': d_idx, 'tree': temp_tree, 'otv': otv})

        if not candidates:
            success = False;
            break

        best_candidate = max(candidates, key=lambda x: x['otv'])
        current_tree = best_candidate['tree']

        for node in current_tree.treepaths2:
            if node not in tree_nodes: tree_nodes.append(node)
        remaining_indices.remove(best_candidate['d_idx'])

    bw_comp = 0;
    cpu_comp = 0;
    mem_comp = 0

    if success:
        occ = (current_tree.tree > 0)
        Bandwidth_status[occ] -= request.bw_origin
        bw_comp = np.sum(current_tree.tree) * request.bw_origin

        d_nodes, d_types = np.where(current_tree.hvt > 0)
        for node, vnf in zip(d_nodes, d_types):
            indices = [i for i, x in enumerate(request.vnf) if x == vnf]
            for idx in indices:
                CPU_status[node] -= request.cpu_origin[idx]
                Memory_status[node] -= request.memory_origin[idx]
                cpu_comp += request.cpu_origin[idx]
                mem_comp += request.memory_origin[idx]
                hvt_all[node, vnf] += 1

        request.id = event_id
        return request, Bandwidth_status, CPU_status, Memory_status, hvt_all, request, current_tree, bw_comp, cpu_comp, mem_comp, True
    else:
        return request, Bandwidth_status, CPU_status, Memory_status, hvt_all, None, None, 0, 0, 0, False