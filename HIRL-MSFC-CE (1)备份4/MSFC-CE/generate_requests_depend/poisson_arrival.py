# poisson_arrival.py
import numpy as np

def generate_poisson_arrive_time_list(T: int, lamda: float):
    """生成服从泊松过程的到达时间列表"""
    arrive_times = []
    t = 0.0
    while t < T:
        inter_arrival = np.random.exponential(1 / lamda)
        t += inter_arrival
        if t < T:
            arrive_times.append(t)
    return np.array(arrive_times)