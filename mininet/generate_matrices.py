# -*- coding: utf-8 -*-
# @File    : generate_matrices_standalone.py
# @Date    : 2024
# @Author  : Modified for standalone usage (no TMgen dependency)
# -------------------------------------------
# 功能说明：
# 1. 不依赖 TMgen，自实现流量矩阵生成
# 2. 生成符合引力模型和昼夜周期的流量矩阵
# 3. 自动保存所有统计数据和图表
# -------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class TrafficMatrix:
    """自定义流量矩阵类"""

    def __init__(self, matrix):
        self.matrix = matrix

    def at_time(self, t):
        """获取指定时刻的流量矩阵"""
        return self.matrix[:, :, t]


def set_seed(seed):
    """设置随机数种子"""
    np.random.seed(seed)
    print(f"[INFO] Random seed set to {seed}")


def generate_gravity_model(num_nodes, spatial_variance):
    """
    生成基于引力模型的基础流量矩阵

    引力模型假设：流量 ∝ (源节点重要性 × 目标节点重要性) / 距离
    """
    # 生成节点重要性（模拟节点的流量吸引力）
    node_importance = np.random.gamma(2, spatial_variance, num_nodes)

    # 生成距离矩阵（随机距离，可以根据实际拓扑调整）
    distances = np.random.uniform(1, 10, (num_nodes, num_nodes))
    np.fill_diagonal(distances, 0)  # 自己到自己距离为0

    # 应用引力模型
    gravity_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                gravity_matrix[i, j] = (node_importance[i] * node_importance[j]) / (distances[i, j] + 1)

    return gravity_matrix


def generate_diurnal_pattern(num_tms, pm_ratio, t_ratio, diurnal_freq):
    """
    生成昼夜周期调制模式

    参数：
        num_tms: 时间段数量
        pm_ratio: 峰均比（peak-to-mean）
        t_ratio: 谷均比（trough-to-mean）
        diurnal_freq: 昼夜频率
    """
    # 生成时间序列
    time_points = np.linspace(0, 1, int(num_tms), endpoint=False)

    # 生成正弦波形的昼夜模式
    # 使用余弦函数：cos(2π * freq * t)
    # 范围从 -1 到 1，需要映射到 [t_ratio, pm_ratio]
    cosine_wave = np.cos(2 * np.pi * diurnal_freq * num_tms * time_points)

    # 归一化到 [0, 1]
    normalized = (cosine_wave + 1) / 2

    # 映射到 [t_ratio, pm_ratio]
    modulation = t_ratio + (pm_ratio - t_ratio) * normalized

    return modulation


def modulated_gravity_tm(num_nodes, num_tms, mean_traffic, pm_ratio, t_ratio,
                         diurnal_freq, spatial_variance, temporal_variance):
    """
    生成带昼夜调制的引力模型流量矩阵

    返回：TrafficMatrix 对象，包含 (num_nodes × num_nodes × num_tms) 的三维矩阵
    """
    print(f"[INFO] Generating gravity model for {num_nodes} nodes...")

    # 1. 生成基础引力模型
    base_gravity = generate_gravity_model(num_nodes, spatial_variance)

    # 2. 归一化到平均流量
    base_gravity = base_gravity / base_gravity.mean() * mean_traffic

    # 3. 生成昼夜调制模式
    diurnal_pattern = generate_diurnal_pattern(num_tms, pm_ratio, t_ratio, diurnal_freq)

    # 4. 创建三维矩阵 (nodes × nodes × time)
    tm_matrix = np.zeros((num_nodes, num_nodes, num_tms))

    # 5. 对每个时间段应用调制和随机扰动
    for t in range(num_tms):
        # 应用昼夜调制
        modulated = base_gravity * diurnal_pattern[t]

        # 添加时间随机扰动（模拟流量的随机波动）
        noise = np.random.normal(1, temporal_variance, (num_nodes, num_nodes))
        noise = np.maximum(noise, 0)  # 确保非负

        tm_matrix[:, :, t] = modulated * noise

    print(f"[INFO] Traffic matrix generated with shape {tm_matrix.shape}")
    return TrafficMatrix(tm_matrix)


def generate_tm(args):
    """生成流量矩阵主函数"""
    print("[INFO] Generating traffic matrices...")

    # 转换为整数
    num_nodes_int = int(args.num_nodes)
    num_tms_int = int(args.num_tms)

    # 修正频率以避免浮点误差
    fixed_diurnal_freq = 1.0 / num_tms_int

    # 生成流量矩阵
    tm = modulated_gravity_tm(
        num_nodes_int,
        num_tms_int,
        args.mean_traffic,
        args.pm_ratio,
        args.t_ratio,
        fixed_diurnal_freq,
        args.spatial_variance,
        args.temporal_variance,
    )

    # 计算每个时间步的平均流量
    mean_time_tm = []
    for t in range(num_tms_int):
        mean_time_tm.append(tm.at_time(t).mean())
        print(f"[INFO] time {t:2d} h, mean traffic: {mean_time_tm[-1]:.2f}")

    # 生成通信掩码
    _size = (num_nodes_int,) * 2 + (num_tms_int,)
    temp = np.random.random(_size)
    mask = temp < args.communicate_ratio
    communicate_tm = tm.matrix * mask

    mean_communicate_tm = []
    for t in range(num_tms_int):
        mean_communicate_tm.append(communicate_tm[:, :, t].mean())
        print(f"[INFO] time {t:2d} h, mean communicate traffic: {mean_communicate_tm[-1]:.2f}")

    # 保存结果
    np_save(tm.matrix, "traffic_matrix")
    np_save(mean_time_tm, "mean_time_tm")
    np_save(communicate_tm, "communicate_tm")
    np_save(mean_communicate_tm, "mean_communicate_tm")

    # 绘制图表
    plot_tm_mean(mean_time_tm, title="mean_time_tm")
    plot_tm_mean(mean_communicate_tm, title="mean_communicate_tm")

    print("[INFO] Traffic matrices generation complete.")


def np_save(file_data, file_name):
    """保存 numpy 数组"""
    Path("./tm_statistic").mkdir(exist_ok=True)
    np.save(f"./tm_statistic/{file_name}.npy", file_data)
    print(f"[SAVE] {file_name}.npy saved in ./tm_statistic/")


def plot_tm_mean(mean_list, x_label='time', y_label='mean_traffic', title='mean'):
    """绘制流量均值变化图"""
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(f'{title} - Traffic Pattern', fontsize=14)

    x = list(range(len(mean_list)))
    y = mean_list

    # 使用柱状图和折线图结合
    plt.bar(x, y, alpha=0.6, color='skyblue')
    plt.plot(x, y, 'r-', linewidth=2, marker='o', markersize=4)
    plt.grid(True, alpha=0.3)

    Path("./figure").mkdir(exist_ok=True)
    plt.savefig(f"./figure/{title}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"[PLOT] Figure saved to ./figure/{title}.pdf")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate traffic matrices (standalone version)")
    parser.add_argument("--seed", type=int, default=2020, help="random seed")
    parser.add_argument("--num_nodes", type=int, default=14, help="number of nodes in network")
    parser.add_argument("--num_tms", type=int, default=24, help="total number of matrices")
    parser.add_argument("--mean_traffic", type=float, default=5 * 10 ** 3 * 0.75,
                        help="mean volume of traffic (Kbps)")
    parser.add_argument("--pm_ratio", type=float, default=1.5, help="peak-to-mean ratio")
    parser.add_argument("--t_ratio", type=float, default=0.75, help="trough-to-mean ratio")
    parser.add_argument("--diurnal_freq", type=float, default=1 / 24, help="Frequency of modulation")
    parser.add_argument("--spatial_variance", type=float, default=500,
                        help="Variance on the volume between OD pairs")
    parser.add_argument("--temporal_variance", type=float, default=0.03,
                        help="Variance on the volume in time")
    parser.add_argument("--communicate_ratio", type=float, default=0.7,
                        help="percentage of nodes to communicate")
    parser.add_argument("--plot_only", action="store_true",
                        help="if set, only plot existing mean_time_tm.npy")

    args = parser.parse_args()
    set_seed(args.seed)

    # 检测已有文件
    tm_stat_path = Path("./tm_statistic/mean_time_tm.npy")
    if args.plot_only and tm_stat_path.exists():
        print("[INFO] mean_time_tm.npy found, plotting only...")
        mean_time_tm = np.load(tm_stat_path)
        plot_tm_mean(mean_time_tm)
    else:
        generate_tm(args)