# config.py
import numpy as np
import os
import scipy.io as sio
from US_Backbone import US_Backbone
from topology_link import topology_link

# ================= 1. 路径配置 =================
# 请确保这里指向正确的文件夹
DATA_PATH = r"E:\pycharmworkspace\SFC-master\HIRL-MSFC-CE (1)\MSFC-CE\generate_requests_depend\py\out"
if not os.path.exists(DATA_PATH):
    DATA_PATH = './out'

# ================= 2. 基础网络参数 =================
node_num = 28
type_num = 8
dest_num = 5

# 动态计算拓扑
topo_matrix = US_Backbone()
link, link_num = topology_link(topo_matrix)

# DC 节点 (0-based)
ran_gen = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 27, 28]
dc_num = len(ran_gen)
DC = [x - 1 for x in sorted(ran_gen)]

# 资源容量
cpu_capacity = 80.0
memory_capacity = 60.0
bandwidth_capacity = 80.0

# ================= 3. 加载路径数据 =================
path = None
try:
    path_file = os.path.join(DATA_PATH, 'US_Backbone_paths.mat')
    if os.path.exists(path_file):
        path_data = sio.loadmat(path_file)
        path = path_data['Paths']
        print("Config: 路径数据加载成功。")
    else:
        print(f"Config 警告: 找不到路径文件 {path_file}")
except Exception as e:
    print(f"Config 错误: 加载路径失败 {e}")