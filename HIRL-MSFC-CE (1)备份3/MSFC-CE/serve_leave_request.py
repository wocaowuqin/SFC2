# serve_leave_request.py
import numpy as np
import config


def serve_leave_request(request, tree, Bandwidth_status, CPU_status, Memory_status, hvt_all):
    deployed_nodes, deployed_types = np.where(tree.hvt > 0)

    for node, vnf in zip(deployed_nodes, deployed_types):
        if hvt_all[node, vnf] > 1:
            hvt_all[node, vnf] -= 1
        else:
            hvt_all[node, vnf] = 0
            indices = [i for i, x in enumerate(request.vnf) if x == vnf]
            for idx in indices:
                CPU_status[node] += request.cpu_origin[idx]
                Memory_status[node] += request.memory_origin[idx]

            if CPU_status[node] > config.cpu_capacity: CPU_status[node] = config.cpu_capacity
            if Memory_status[node] > config.memory_capacity: Memory_status[node] = config.memory_capacity

    occupied_links = (tree.tree > 0)
    Bandwidth_status[occupied_links] += request.bw_origin
    Bandwidth_status[Bandwidth_status > config.bandwidth_capacity] = config.bandwidth_capacity

    return Bandwidth_status, CPU_status, Memory_status, hvt_all