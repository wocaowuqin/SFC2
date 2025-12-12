# generate_event_list.py
import numpy as np
import scipy.io as sio

requests = sio.loadmat('output/requests/sorted_requests.mat')['sorted_requests'][0]
max_time_step = max(int(r['leave_time_step']) for r in requests)

event_list = []
current_events = {}

for req in requests:
    aid = int(req['arrive_time_step'])
    lid = int(req['leave_time_step'])

    # 到达事件
    if aid not in current_events:
        current_events[aid] = {'time_step': aid, 'arrive_event': [], 'leave_event': []}
    current_events[aid]['arrive_event'].append(int(req['id']))

    # 离开事件
    if lid <= max_time_step:
        if lid not in current_events:
            current_events[lid] = {'time_step': lid, 'arrive_event': [], 'leave_event': []}
        current_events[lid]['leave_event'].append(int(req['id']))

# 填满所有时间步
for t in range(1, max_time_step + 1):
    if t not in current_events:
        event_list.append({'time_step': t, 'arrive_event': [], 'leave_event': []})
    else:
        event_list.append(current_events[t])

event_array = np.array(event_list, dtype=object)
sio.savemat('output/requests/event_list.mat', {'event_list': event_array})
print("event_list.mat 生成完成！")