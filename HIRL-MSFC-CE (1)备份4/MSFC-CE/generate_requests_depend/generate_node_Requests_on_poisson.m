function node_requests = generate_node_Requests_on_poisson(source,node_important,arrive_time_list,all_vnf)
%%生成节点source的泊松业务序列

%候选目的节点
temp_node_important = setdiff(node_important,source);%除源节点之外的节点
% temp_node_normal = setdiff(node_normal, source);%除源节点之外的节点

% 初始请求带宽的取值范围
max_bandwidth = 8;
min_bandwidth = 4;

% 初始请求节点资源由带宽和vnf决定

%持续时间负指数分布均值
mean_lifetime = 3;

%业务序列
node_requests=[];


n = 5;
for i = 1 : length(arrive_time_list)
%     dest_pro = rand;
%     if dest_pro < 0.7
        dest = temp_node_important(randperm(numel(temp_node_important),n));
%     else
%         dest = temp_node_normal(randi(length(temp_node_normal)));
%     end

    request = generate_Requests_on_poisson(i,source,dest,all_vnf,max_bandwidth,min_bandwidth,arrive_time_list(i),mean_lifetime);
    node_requests = [node_requests;request];
end

