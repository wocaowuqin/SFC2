# generate_node_requests.py
from generate_request import generate_request

def generate_node_requests(source, important_nodes, arrive_times, all_vnf):
    other_important = [n for n in important_nodes if n != source]
    n_dest = 5
    requests = []

    for i, t in enumerate(arrive_times):
        dests = np.random.choice(other_important, n_dest, replace=False)
        req = generate_request(i + 1, source, dests, all_vnf, arrive_time=t)
        requests.append(req)

    return requests