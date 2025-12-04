# data_generator.py
import random
import math

def generate_vnfs_catalog(vnf_type_num=8):
    all_vnf = []
    for vnf_type in range(1, vnf_type_num + 1):
        # CPU需求系数 [0.25, 3.0]
        cpu_need = random.random() * 2.75 + 0.25
        # 内存需求系数 [0.25, 2.0]
        memory_need = random.random() * 1.75 + 0.25

        vnf = {'type': vnf_type, 'cpu_need': cpu_need, 'memory_need': memory_need}
        all_vnf.append(vnf)
    return all_vnf

def generate_poisson_arrive_time_list(T, lamda):
    """生成服从泊松过程的到达时间列表"""
    time_state = 0
    arrive_time_list = []

    while time_state < T:
        interval = random.expovariate(lamda)
        t = time_state + interval
        if t < T:
            time_state = t
            arrive_time_list.append(t)
        else:
            break
    return arrive_time_list

def generate_single_request(req_id, source, dest, all_vnf, max_bw, min_bw, arrive_time, mean_lifetime):
    """生成单个业务请求"""
    vnf_type_num = len(all_vnf)
    vnf_num = 3  # 固定选择3个VNF

    # 随机选择3个不同的VNF类型 (索引从1开始对应type)
    vnf_indices = random.sample(range(vnf_type_num), vnf_num)
    selected_vnfs = [all_vnf[i] for i in vnf_indices]
    vnf_types = [v['type'] for v in selected_vnfs]

    # 初始带宽
    bw_origin = random.randint(min_bw, max_bw)

    cpu_origin = []
    memory_origin = []

    for v in selected_vnfs:
        cpu = round(bw_origin * v['cpu_need'])
        mem = round(bw_origin * v['memory_need'])
        cpu_origin.append(cpu)
        memory_origin.append(mem)

    # 请求持续时间
    while True:
        lifetime = 1 + random.expovariate(1.0 / (mean_lifetime - 1))
        if lifetime <= 6:
            break

    leave_time = arrive_time + lifetime

    request = {
        'id': req_id,
        'source': source,
        'dest': dest,
        'vnf': vnf_types,
        'cpu_origin': cpu_origin,
        'memory_origin': memory_origin,
        'bw_origin': bw_origin,
        'arrive_time': arrive_time,
        'lifetime': lifetime,
        'leave_time': leave_time,
        'arrive_time_step': math.ceil(arrive_time),
        'leave_time_step': math.ceil(leave_time)
    }
    return request

def generate_node_requests(source, node_important, arrive_time_list, all_vnf):
    """为特定源节点生成一系列请求"""
    node_requests = []
    candidates = [n for n in node_important if n != source]

    max_bandwidth = 8
    min_bandwidth = 4
    multicast_num = 5
    mean_lifetime = 3

    for i, arrive_time in enumerate(arrive_time_list):
        k = min(multicast_num, len(candidates))
        dest = random.sample(candidates, k)
        # ID 暂时设为0，后续统一重排
        req = generate_single_request(0, source, dest, all_vnf,
                                      max_bandwidth, min_bandwidth,
                                      arrive_time, mean_lifetime)
        node_requests.append(req)

    return node_requests