import pickle
import os


def generate_events():
    # 读取请求数据
    file_path = './data_output/sorted_requests.pkl'
    if not os.path.exists(file_path):
        print("未找到 sorted_requests.pkl，请先运行 main_generate.py")
        return

    with open(file_path, 'rb') as f:
        requests_list = pickle.load(f)

    print(f"读取到 {len(requests_list)} 条请求，开始生成事件列表...")

    # 找出最大时间步以确定列表长度
    max_time_step = 0
    for req in requests_list:
        if req['leave_time_step'] > max_time_step:
            max_time_step = req['leave_time_step']

    # 初始化事件列表 (索引 0 对应时间步 0，虽然通常从1开始用，多建一点无妨)
    # 结构: event_list[t] = {'time_step': t, 'arrive_event': [], 'leave_event': []}
    event_list = []
    for t in range(max_time_step + 2):
        event_list.append({
            'time_step': t,
            'arrive_event': [],  # 存储请求ID
            'leave_event': []
        })

    # 填充到达事件
    for req in requests_list:
        t_arr = req['arrive_time_step']
        if t_arr < len(event_list):
            event_list[t_arr]['arrive_event'].append(req['id'])

    # 填充离开事件
    for req in requests_list:
        t_leave = req['leave_time_step']
        if t_leave < len(event_list):
            event_list[t_leave]['leave_event'].append(req['id'])

    # 保存
    save_path = './data_output/event_list.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(event_list, f)

    print("事件列表生成完成！")
    print(f"文件已保存至: {save_path}")


if __name__ == '__main__':
    generate_events()