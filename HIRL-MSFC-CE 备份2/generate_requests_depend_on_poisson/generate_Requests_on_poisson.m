function request = generate_Requests_on_poisson(id,source,dest,all_vnf,maxslot,minslot,arrive_time,mean_lifetime)
%%生成业务请求的通用方法


%vnf种类数
vnf_type_num = length(all_vnf);

% 请求的vnf数目
vnf_num=3;

% 随机生成请求的vnf
temp = randperm(vnf_type_num);
vnf = temp(1:vnf_num);
clear temp;

% 初始请求频隙数
bw_origin = randi([minslot,maxslot]);
%初始请求节点资源数
cpu_origin = [];
memory_origin = [];

for i = 1 : vnf_num
    cpu_need = round(bw_origin * all_vnf(vnf(i)).cpu_need);
    cpu_origin = [cpu_origin,cpu_need];
    memory_need = round(bw_origin * all_vnf(vnf(i)).memory_need);
    memory_origin = [memory_origin,memory_need];
end


%请求持续时间，确保大于1小于6
lifetime = 1 + exprnd(mean_lifetime - 1);
while lifetime > 6
    lifetime = 1 + exprnd(mean_lifetime - 1);
end

%请求离开时间
leave_time = arrive_time + lifetime;
%请求到达时间步和请求离开时间步
arrive_time_step = ceil(arrive_time);
leave_time_step = ceil(leave_time);


request.id = id;
request.source = source;
request.dest = dest;
request.vnf = vnf;
request.cpu_origin = cpu_origin;
request.memory_origin = memory_origin;
request.bw_origin = bw_origin;
request.arrive_time = arrive_time;
request.lifetime = lifetime;
request.leave_time = leave_time;
request.arrive_time_step = arrive_time_step;
request.leave_time_step = leave_time_step;