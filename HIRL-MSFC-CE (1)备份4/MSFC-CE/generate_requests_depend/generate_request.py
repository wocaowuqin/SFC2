# generate_request.py
import numpy as np

def generate_request(id_, source, dests, all_vnf, maxslot=8, minslot=4, arrive_time=None, mean_lifetime=3):
    vnf_type_num = len(all_vnf)
    vnf_num = 3

    # 随机选3种VNF类型
    vnf = np.random.permutation(vnf_type_num)[:vnf_num] + 1  # 1-based

    # 随机带宽需求
    bw_origin = np.random.randint(minslot, maxslot + 1)

    cpu_origin = []
    memory_origin = []
    for v in vnf:
        vnf_data = all_vnf[v - 1]
        cpu_origin.append(round(bw_origin * vnf_data['cpu_need']))
        memory_origin.append(round(bw_origin * vnf_data['memory_need']))

    # 持续时间（截断指数分布）
    lifetime = 1 + np.random.exponential(mean_lifetime - 1)
    while lifetime > 6:
        lifetime = 1 + np.random.exponential(mean_lifetime - 1)

    leave_time = arrive_time + lifetime
    arrive_time_step = np.ceil(arrive_time)
    leave_time_step = np.ceil(leave_time)

    request = {
        'id': int(id_),
        'source': int(source),
        'dest': np.array(dests, dtype=int),
        'vnf': np.array(vnf, dtype=int),
        'cpu_origin': np.array(cpu_origin),
        'memory_origin': np.array(memory_origin),
        'bw_origin': int(bw_origin),
        'arrive_time': float(arrive_time),
        'lifetime': float(lifetime),
        'leave_time': float(leave_time),
        'arrive_time_step': int(arrive_time_step),
        'leave_time_step': int(leave_time_step),
    }
    return request