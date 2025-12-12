import numpy as np


def topology_link(W):
    """
    对应 MATLAB 的 topology_link 函数

    参数:
        W: 邻接矩阵 (numpy.ndarray), 表示网络拓扑
           W[i, j] != 0 且 != inf 表示 i 和 j 之间有链路

    返回:
        link: 链路编号矩阵 (numpy.ndarray), link[i, j] 存储链路的唯一 ID (从 1 开始)
        linksum: 链路总数 (int)
    """
    # 获取节点数量 (行数)
    nodenum = W.shape[0]

    # 初始化 link 矩阵为全 0
    link = np.zeros((nodenum, nodenum), dtype=int)

    linksum = 0

    # 遍历矩阵
    for i in range(nodenum):
        for j in range(nodenum):
            # 对应 MATLAB: if ((W(i,j)~=0)&&(W(i,j)~=inf))
            # 检查是否有边 (值不为0且不为无穷大)
            if W[i, j] != 0 and W[i, j] != float('inf'):
                linksum += 1
                link[i, j] = linksum

    return link, linksum


# 使用示例
if __name__ == "__main__":
    # 创建一个示例拓扑矩阵 (inf 表示无连接)
    inf = float('inf')
    W = np.array([
        [0, 10, inf],
        [10, 0, 5],
        [inf, 5, 0]
    ])

    link_matrix, total_links = topology_link(W)

    print(f"Total links: {total_links}")
    print("Link Matrix:")
    print(link_matrix)