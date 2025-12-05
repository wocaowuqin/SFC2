# generate_event.py
import pickle
import os
import numpy as np
import scipy.io as sio


def generate_events():
    # --- [修改] 基础路径设为 ./out ---
    base_dir = './out'

    file_path = os.path.join(base_dir, 'sorted_requests.pkl')

    if not os.path.exists(file_path):
        print(f"错误：未找到 {file_path}。请先运行 main_generate.py！")
        return

    print("读取请求数据...")
    with open(file_path, 'rb') as f:
        requests_list = pickle.load(f)

    # 确定最大时间步
    max_time_step = 0
    for req in requests_list:
        if req['leave_time_step'] > max_time_step:
            max_time_step = req['leave_time_step']

    print(f"最大时间步: {max_time_step}")

    # 初始化事件列表
    py_event_list = []
    for t in range(max_time_step + 5):
        py_event_list.append({
            'time_step': t,
            'arrive_event': [],
            'leave_event': []
        })

    # 填充事件
    for req in requests_list:
        t_arr = req['arrive_time_step']
        if t_arr < len(py_event_list):
            py_event_list[t_arr]['arrive_event'].append(req['id'])

        t_leave = req['leave_time_step']
        if t_leave < len(py_event_list):
            py_event_list[t_leave]['leave_event'].append(req['id'])

    # --- 保存为 MAT 格式 ---
    print("正在转换为 MATLAB 格式...")

    dtype = [('time_step', 'O'), ('arrive_event', 'O'), ('leave_event', 'O')]
    mat_event_arr = np.zeros((len(py_event_list),), dtype=dtype)

    for i, evt in enumerate(py_event_list):
        mat_event_arr[i]['time_step'] = evt['time_step']

        if evt['arrive_event']:
            mat_event_arr[i]['arrive_event'] = np.array(evt['arrive_event']).reshape(1, -1)
        else:
            mat_event_arr[i]['arrive_event'] = np.array([]).reshape(1, 0)

        if evt['leave_event']:
            mat_event_arr[i]['leave_event'] = np.array(evt['leave_event']).reshape(1, -1)
        else:
            mat_event_arr[i]['leave_event'] = np.array([]).reshape(1, 0)

    save_path_mat = os.path.join(base_dir, 'event_list.mat')
    sio.savemat(save_path_mat, {'event_list': mat_event_arr})

    print(f"成功！事件列表已保存至: {save_path_mat}")


if __name__ == '__main__':
    generate_events()