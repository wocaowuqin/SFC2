function all_vnf = generate_VNFs_catalog()
%% 生成8种类型的vnf,且vnf3依赖于vnf7，vnf8依赖于vnf4
vnf_type_num = 8;
all_vnf=[];

for vnf_type = 1: vnf_type_num
    % type
    vnf.type = vnf_type;

   % dependency
%     if vnf_type==3
%         vnf.dependency = 7;
%     elseif vnf_type == 8
%         vnf.dependency = 4;
%     else 
%         vnf.dependency = 0;
%     end
    
    % bandwidth change ratio
    switch vnf_type
        case 1
%             vnf.bwchangeratio = round(rand*0.1+0.6,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
        case 2
%             vnf.bwchangeratio = round(rand*0.1+0.7,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
        case 3
%             vnf.bwchangeratio = round(rand*0.1+0.8,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
        case 4
%             vnf.bwchangeratio = round(rand*0.1+0.9,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
        case 5
%             vnf.bwchangeratio = round(rand*0.1+1.0,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
        case 6
%             vnf.bwchangeratio = round(rand*0.1+1.1,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
        case 7
%             vnf.bwchangeratio = round(rand*0.1+1.2,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
        otherwise
%             vnf.bwchangeratio = round(rand*0.1+1.3,2);
            vnf.cpu_need = rand * 2.75 + 0.25;
            vnf.memory_need = rand * 1.75 + 0.25;
    end
    
%     vnf.factor = vnf.bwchangeratio;

    all_vnf = [all_vnf,vnf];   
end


save('G:\MPH\generate_requests_depend_on_poisson\all_vnf.mat','all_vnf');