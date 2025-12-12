%%基于requests按时间步生成event_list
clc;
close all;
clear ;

filename = "sorted_requests.mat";
temp_requests = load(filename);
requests_list = temp_requests.sorted_requests;
clear temp_requests;

requests_num = length(requests_list);

event_list = [];
time_step = 0;

% 到达事件
for i = 1 : requests_num
    if(mod(i,5000) == 0)
        i
    end
    temp_arrive_time_step = requests_list(i).arrive_time_step;

    if(time_step < temp_arrive_time_step)
        time_step = time_step + 1;

        while(time_step < temp_arrive_time_step)%某一个时间步没有到达业务
            event.time_step = time_step;
            event.arrive_event = [];
            event.leave_event = [];
            event_list = [event_list;event];
            time_step = time_step + 1;
        end
        event.time_step = time_step;
        event.arrive_event = requests_list(i).id;
        event.leave_event = [];
        event_list = [event_list;event];
    else
        event = event_list(time_step);
        event.arrive_event = [event.arrive_event,requests_list(i).id];
        event_list(time_step) = event;
    end
end

output = "leave"

% 离开事件，最后一个时间步以后离开的暂时忽略
for i = 1 : requests_num

    if(mod(i,5000) == 0)
        i
    end
    temp_leave_time_step = requests_list(i).leave_time_step;
    if(temp_leave_time_step <= time_step)
        event = event_list(temp_leave_time_step);
        event.leave_event = [event.leave_event,requests_list(i).id];
        event_list(temp_leave_time_step) = event;
    end
end

filepath = sprintf('%s.mat','event_list');
save(filepath,'event_list');
