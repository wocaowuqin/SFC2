# generate_vnf_catalog.py
import numpy as np
import scipy.io as sio

def generate_vnf_catalog(save_path='data/all_vnf.mat'):
    vnf_type_num = 8
    all_vnf = []

    for vnf_type in range(1, vnf_type_num + 1):
        vnf = {
            'type': vnf_type,
            # 'dependency': 7 if vnf_type == 3 else (4 if vnf_type == 8 else 0),
            'cpu_need': np.random.uniform(0.25, 3.0),
            'memory_need': np.random.uniform(0.25, 2.0),
        }
        all_vnf.append(vnf)

    all_vnf_array = np.array(all_vnf, dtype=object)
    sio.savemat(save_path, {'all_vnf': all_vnf_array})
    print(f"VNF catalog saved to {save_path}")

if __name__ == "__main__":
    generate_vnf_catalog()