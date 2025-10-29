# -*- coding: utf-8 -*-
# @File    : generate_matrices_enhanced.py
# @Date    : 2024
# @Author  : Enhanced version with TMgen
# -------------------------------------------
# 功能说明：
# 1. 使用 TMgen 生成多时段流量矩阵
# 2. 增强的统计分析和可视化
# 3. 支持多种导出格式（NPY, CSV）
# 4. 自动生成详细报告
# -------------------------------------------

import argparse
from pathlib import Path
import os
import numpy as np
import numpy.random
from tmgen.models import modulated_gravity_tm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，避免显示问题


# ================================
# 随机数种子
# ================================
def set_seed(seed):
    numpy.random.seed(seed)
    print(f"[INFO] Random seed set to {seed}")


# ================================
# 生成流量矩阵函数
# ================================
def generate_tm(args):
    print("[INFO] Generating traffic matrices...")
    print(f"[PARAM] Nodes: {args.num_nodes}, Time slots: {args.num_tms}")
    print(f"[PARAM] Mean traffic: {args.mean_traffic:.2f} Kbps")
    print(f"[PARAM] Peak-to-Mean ratio: {args.pm_ratio}, Trough-to-Mean ratio: {args.t_ratio}")

    # 转换为整数
    num_nodes_int = int(args.num_nodes)
    num_tms_int = int(args.num_tms)

    # 修正频率以避免浮点精度误差
    fixed_diurnal_freq = 1.0 / num_tms_int

    # 生成流量矩阵
    print("[INFO] Calling TMgen modulated_gravity_tm...")
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

    print(f"[INFO] Traffic matrix shape: {tm.matrix.shape}")

    # 计算每个时间步的平均流量
    mean_time_tm = []
    max_traffic_per_time = []
    min_traffic_per_time = []

    for t in range(num_tms_int):
        tm_at_t = tm.at_time(t)
        mean_val = tm_at_t.mean()
        max_val = tm_at_t.max()
        min_val = tm_at_t[tm_at_t > 0].min() if np.any(tm_at_t > 0) else 0

        mean_time_tm.append(mean_val)
        max_traffic_per_time.append(max_val)
        min_traffic_per_time.append(min_val)

        print(f"[INFO] Time {t:2d}h | Mean: {mean_val:8.2f} | Max: {max_val:8.2f} | Min: {min_val:8.2f}")

    # 随机生成通信掩码
    print(f"[INFO] Generating communication mask with ratio {args.communicate_ratio}")
    _size = (num_nodes_int,) * 2 + (num_tms_int,)
    temp = np.random.random(_size)
    mask = temp < args.communicate_ratio
    communicate_tm = tm.matrix * mask

    # 计算通信节点的平均流量
    mean_communicate_tm = []
    active_pairs_count = []

    for t in range(num_tms_int):
        comm_matrix = communicate_tm[:, :, t]
        active_pairs = np.sum(comm_matrix > 0)
        mean_val = comm_matrix.mean()

        mean_communicate_tm.append(mean_val)
        active_pairs_count.append(active_pairs)

        print(f"[INFO] Time {t:2d}h | Comm Mean: {mean_val:8.2f} | Active pairs: {active_pairs:3d}")

    # 统计分析
    print("\n" + "=" * 60)
    print("TRAFFIC STATISTICS SUMMARY")
    print("=" * 60)

    peak_time = np.argmax(mean_time_tm)
    trough_time = np.argmin(mean_time_tm)

    print(f"Overall Mean Traffic:     {np.mean(mean_time_tm):10.2f} Kbps")
    print(f"Overall Max Traffic:      {np.max(max_traffic_per_time):10.2f} Kbps")
    print(f"Overall Min Traffic:      {np.min(min_traffic_per_time):10.2f} Kbps")
    print(f"Peak Hour:                Time {peak_time:2d} ({mean_time_tm[peak_time]:.2f} Kbps)")
    print(f"Trough Hour:              Time {trough_time:2d} ({mean_time_tm[trough_time]:.2f} Kbps)")
    print(f"Actual Peak-to-Mean:      {mean_time_tm[peak_time] / np.mean(mean_time_tm):.3f}")
    print(f"Actual Trough-to-Mean:    {mean_time_tm[trough_time] / np.mean(mean_time_tm):.3f}")
    print(f"Average Active Pairs:     {np.mean(active_pairs_count):.1f}")
    print(f"Communication Efficiency: {np.mean(mean_communicate_tm) / np.mean(mean_time_tm) * 100:.2f}%")
    print("=" * 60 + "\n")

    # 保存数据
    print("[INFO] Saving data files...")
    np_save(tm.matrix, "traffic_matrix")
    np_save(mean_time_tm, "mean_time_tm")
    np_save(communicate_tm, "communicate_tm")
    np_save(mean_communicate_tm, "mean_communicate_tm")
    np_save(max_traffic_per_time, "max_traffic_per_time")
    np_save(min_traffic_per_time, "min_traffic_per_time")
    np_save(active_pairs_count, "active_pairs_count")

    # 保存统计摘要
    summary = {
        'num_nodes': num_nodes_int,
        'num_tms': num_tms_int,
        'mean_traffic': float(np.mean(mean_time_tm)),
        'peak_time': int(peak_time),
        'peak_traffic': float(mean_time_tm[peak_time]),
        'trough_time': int(trough_time),
        'trough_traffic': float(mean_time_tm[trough_time]),
        'actual_pm_ratio': float(mean_time_tm[peak_time] / np.mean(mean_time_tm)),
        'actual_t_ratio': float(mean_time_tm[trough_time] / np.mean(mean_time_tm)),
    }
    np_save(summary, "traffic_summary")

    # 导出CSV（可选）
    if args.export_csv:
        print("[INFO] Exporting to CSV format...")
        export_to_csv(mean_time_tm, mean_communicate_tm, max_traffic_per_time,
                      min_traffic_per_time, active_pairs_count)

    # 生成可视化
    print("[INFO] Generating visualizations...")
    plot_tm_mean(mean_time_tm, title="mean_time_tm", ylabel="Mean Traffic (Kbps)")
    plot_tm_mean(mean_communicate_tm, title="mean_communicate_tm", ylabel="Mean Comm Traffic (Kbps)")

    if args.plot_detailed:
        plot_detailed_analysis(mean_time_tm, max_traffic_per_time, min_traffic_per_time)
        plot_active_pairs(active_pairs_count)

        # 绘制热力图
        if args.plot_heatmap:
            plot_heatmap(tm.matrix, peak_time, title=f"Peak_Hour_t{peak_time}")
            plot_heatmap(tm.matrix, trough_time, title=f"Trough_Hour_t{trough_time}")
            plot_heatmap(communicate_tm, peak_time, title=f"Comm_Peak_t{peak_time}")

    print("[INFO] Traffic matrices generation complete!")
    print(f"[INFO] Output saved to: ./tm_statistic/ and ./figure/")


# ================================
# 保存函数
# ================================
def np_save(file_data, file_name):
    """保存 numpy 数组或字典"""
    Path("./tm_statistic").mkdir(exist_ok=True)
    np.save(f"./tm_statistic/{file_name}.npy", file_data)
    print(f"[SAVE] {file_name}.npy")


# ================================
# 导出CSV
# ================================
def export_to_csv(mean_time_tm, mean_communicate_tm, max_traffic, min_traffic, active_pairs):
    """导出统计数据为CSV格式"""
    try:
        import pandas as pd
        Path("./csv_output").mkdir(exist_ok=True)

        df = pd.DataFrame({
            'Time': range(len(mean_time_tm)),
            'Mean_Traffic': mean_time_tm,
            'Mean_Comm_Traffic': mean_communicate_tm,
            'Max_Traffic': max_traffic,
            'Min_Traffic': min_traffic,
            'Active_Pairs': active_pairs
        })

        df.to_csv("./csv_output/traffic_statistics.csv", index=False)
        print(f"[EXPORT] traffic_statistics.csv saved in ./csv_output/")
    except ImportError:
        print("[WARN] pandas not installed, skipping CSV export")


# ================================
# 绘图函数
# ================================
def plot_tm_mean(mean_list, x_label='Time (hours)', ylabel='Traffic', title='mean'):
    """绘制基础流量均值图"""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = list(range(len(mean_list)))
    y = mean_list

    # 柱状图 + 折线图
    ax.bar(x, y, alpha=0.6, color='skyblue', label='Traffic Volume')
    ax.plot(x, y, 'r-o', linewidth=2, markersize=4, label='Trend Line')

    # 标注峰值和谷值
    peak_idx = np.argmax(y)
    trough_idx = np.argmin(y)

    ax.plot(peak_idx, y[peak_idx], 'g^', markersize=12, label=f'Peak (t={peak_idx})')
    ax.plot(trough_idx, y[trough_idx], 'rv', markersize=12, label=f'Trough (t={trough_idx})')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title.replace('_', ' ').title(), fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    Path("./figure").mkdir(exist_ok=True)
    plt.savefig(f"./figure/{title}.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.savefig(f"./figure/{title}.png", dpi=150, bbox_inches='tight', pad_inches=0)
    print(f"[PLOT] {title}.pdf and .png")
    plt.close()


def plot_detailed_analysis(mean_traffic, max_traffic, min_traffic):
    """绘制详细的流量分析图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    x = list(range(len(mean_traffic)))

    # 子图1：均值、最大值、最小值对比
    ax1.plot(x, mean_traffic, 'b-o', linewidth=2, label='Mean', markersize=4)
    ax1.plot(x, max_traffic, 'r--s', linewidth=1.5, label='Max', markersize=3)
    ax1.plot(x, min_traffic, 'g--^', linewidth=1.5, label='Min', markersize=3)
    ax1.fill_between(x, min_traffic, max_traffic, alpha=0.2, color='gray')
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Traffic (Kbps)', fontsize=11)
    ax1.set_title('Traffic Range Analysis', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # 子图2：流量变化率
    traffic_change = np.diff(mean_traffic)
    traffic_change = np.append(traffic_change, 0)  # 补齐长度
    colors = ['green' if tc >= 0 else 'red' for tc in traffic_change]

    ax2.bar(x, traffic_change, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('Traffic Change (Kbps)', fontsize=11)
    ax2.set_title('Hour-to-Hour Traffic Change', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("./figure/detailed_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("./figure/detailed_analysis.png", dpi=150, bbox_inches='tight')
    print(f"[PLOT] detailed_analysis.pdf and .png")
    plt.close()


def plot_active_pairs(active_pairs):
    """绘制活跃节点对数量"""
    fig, ax = plt.subplots(figsize=(12, 5))

    x = list(range(len(active_pairs)))
    ax.fill_between(x, active_pairs, alpha=0.5, color='orange')
    ax.plot(x, active_pairs, 'b-o', linewidth=2, markersize=4)

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Number of Active Pairs', fontsize=12)
    ax.set_title('Active Communication Pairs Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 添加平均线
    avg_pairs = np.mean(active_pairs)
    ax.axhline(y=avg_pairs, color='red', linestyle='--', linewidth=2,
               label=f'Average: {avg_pairs:.1f}')
    ax.legend(fontsize=10)

    plt.savefig("./figure/active_pairs.pdf", dpi=300, bbox_inches='tight')
    plt.savefig("./figure/active_pairs.png", dpi=150, bbox_inches='tight')
    print(f"[PLOT] active_pairs.pdf and .png")
    plt.close()


def plot_heatmap(tm_matrix, time_index, title='Traffic_Heatmap'):
    """绘制指定时刻的流量热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))

    data = tm_matrix[:, :, time_index]
    im = ax.imshow(data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax.set_xlabel('Destination Node', fontsize=12)
    ax.set_ylabel('Source Node', fontsize=12)
    ax.set_title(f'{title.replace("_", " ")} (Time {time_index})', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Traffic (Kbps)', fontsize=11)

    # 添加网格
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.grid(which='major', color='white', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.savefig(f"./figure/heatmap_{title}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"./figure/heatmap_{title}.png", dpi=150, bbox_inches='tight')
    print(f"[PLOT] heatmap_{title}.pdf and .png")
    plt.close()


# ================================
# 主函数入口
# ================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate traffic matrices using TMgen (Enhanced Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 基础参数
    parser.add_argument("--seed", type=int, default=2020, help="Random seed")
    parser.add_argument("--num_nodes", type=int, default=14, help="Number of nodes in network")
    parser.add_argument("--num_tms", type=int, default=24, help="Total number of time matrices")

    # 流量参数
    parser.add_argument("--mean_traffic", type=float, default=5 * 10 ** 3 * 0.75,
                        help="Mean volume of traffic (Kbps)")
    parser.add_argument("--pm_ratio", type=float, default=1.5, help="Peak-to-mean ratio")
    parser.add_argument("--t_ratio", type=float, default=0.75, help="Trough-to-mean ratio")
    parser.add_argument("--diurnal_freq", type=float, default=1 / 24,
                        help="Frequency of modulation (will be auto-adjusted)")
    parser.add_argument("--spatial_variance", type=float, default=500,
                        help="Variance on the volume between OD pairs")
    parser.add_argument("--temporal_variance", type=float, default=0.03,
                        help="Variance on the volume in time")
    parser.add_argument("--communicate_ratio", type=float, default=0.7,
                        help="Percentage of nodes to communicate")

    # 功能开关
    parser.add_argument("--plot_only", action="store_true",
                        help="Only plot existing mean_time_tm.npy")
    parser.add_argument("--plot_detailed", action="store_true",
                        help="Generate detailed analysis plots")
    parser.add_argument("--plot_heatmap", action="store_true",
                        help="Generate traffic heatmaps")
    parser.add_argument("--export_csv", action="store_true",
                        help="Export statistics to CSV format")

    args = parser.parse_args()

    print("=" * 60)
    print("Traffic Matrix Generator (Enhanced with TMgen)")
    print("=" * 60)

    set_seed(args.seed)

    # 自动检测已有文件
    tm_stat_path = Path("./tm_statistic/mean_time_tm.npy")
    if args.plot_only and tm_stat_path.exists():
        print("[INFO] Found existing mean_time_tm.npy, plotting only...")
        mean_time_tm = np.load(tm_stat_path)
        plot_tm_mean(mean_time_tm, title="mean_time_tm")

        # 如果存在其他文件也绘制
        comm_path = Path("./tm_statistic/mean_communicate_tm.npy")
        if comm_path.exists():
            mean_communicate_tm = np.load(comm_path)
            plot_tm_mean(mean_communicate_tm, title="mean_communicate_tm")
    else:
        generate_tm(args)

    print("\n" + "=" * 60)
    print("All tasks completed successfully!")
    print("=" * 60)