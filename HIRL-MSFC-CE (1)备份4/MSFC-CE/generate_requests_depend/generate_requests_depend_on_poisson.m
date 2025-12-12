%%main方法，生成服从泊松过程的节点对业务请求
clc;
close all;
clear ;

node_num = 28;

tic
% all_vnf = generate_VNFs_catalog();
temp_all_vnf = load('all_vnf.mat');
all_vnf = temp_all_vnf.all_vnf;
clear temp_all_vnf;

%非数据中心节点
node_important = [16 21 22 24 25 26 27 28];


% %热门节点
% node_important = [1 4 6 8 14];
% %非热门节点
% node_all = [1:node_num];
% node_normal = setdiff(node_all, node_important);%两个数组的差集

%到达率
Lamda_important = 0.6;
Lamda_normal = 2;

%事件步
% T = 2016;%以五分钟为间隔，一周共2016个时间步
T = 720;%以五秒中为间隔，一个小时一共720个时间步

%业务数
request_num = 0;

%总的业务序列
requests = [];

%为非数据中心节点生成业务
for s = 1 : length(node_important)
    s
    arrive_time_list = generate_poisson_arrive_time_list(T,Lamda_important);
    node_requests = generate_node_Requests_on_poisson(node_important(s),node_important,arrive_time_list,all_vnf);
    filepath = sprintf('%s%d%s.mat','G:\0.6\generate_requests_depend_on_poisson\requests\node',node_important(s),'requests');
    save(filepath,'node_requests');
    request_num = request_num +length(node_requests);
    requests = [requests;node_requests];
end


%为热门节点生成业务
% for s = 1 : length(node_important)
%     s
%     arrive_time_list = generate_poisson_arrive_time_list(T,Lamda_important);
%     node_requests = generate_node_Requests_on_poisson(node_important(s),node_important,node_normal,arrive_time_list,all_vnf);
%     filepath = sprintf('%s%d%s.mat','G:\MPH\generate_requests_depend_on_poisson\requests\node',node_important(s),'requests');
%     save(filepath,'node_requests');
%     request_num = request_num +length(node_requests);
%     requests = [requests;node_requests];
% end
% %为非热门节点生成业务
% for s = 1 : length(node_normal)
%     s
%     arrive_time_list = generate_poisson_arrive_time_list(T,Lamda_normal);
%     node_requests = generate_node_Requests_on_poisson(node_normal(s),node_important,node_normal,arrive_time_list,all_vnf);
%     filepath = sprintf('%s%d%s.mat','G:\MPH\generate_requests_depend_on_poisson\requests\node',node_normal(s),'requests');
%     save(filepath,'node_requests');
%     request_num = request_num +length(node_requests);
%     requests = [requests;node_requests];
% end

%业务按到达时间排序
[sorted_arrive_time,sortedIndex] = sort([requests.arrive_time]);
sorted_requests = requests(sortedIndex);
%重新编号
new_Id_list = num2cell([1 : length(sorted_requests)]);
[sorted_requests.id] = new_Id_list{:};


toc

filepath = sprintf('%s.mat','H:\MPH\generate_requests_depend_on_poisson\requests');
save(filepath,'requests');
filepath = sprintf('%s.mat','H:\MPH\generate_requests_depend_on_poisson\sorted_requests');
save(filepath,'sorted_requests');
