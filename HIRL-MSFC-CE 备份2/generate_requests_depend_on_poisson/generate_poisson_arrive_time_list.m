function arrive_time_list=generate_poisson_arrive_time_list(T,Lamda)
%函数用来产生业务到达时刻，业务服从到达率为Lamda的possion过程

time_state=0;
arrive_time_list = [];


while time_state < T
    t = time_state + exprnd(1/Lamda);%按照泊松过程产生到达时间
    time_state = t;
    arrive_time_list = [arrive_time_list,t];
end

