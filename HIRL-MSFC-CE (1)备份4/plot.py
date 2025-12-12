import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
import numpy as np

# ==========================================
# 全局绘图风格设置
# ==========================================
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# [新增] 定义图片输出目录
IMAGE_DIR = "image"


def ensure_image_dir():
    """确保 image 目录存在"""
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"📂 已创建图片输出目录: {IMAGE_DIR}")


def find_latest_files(results_dir="results"):
    """自动查找 results 目录中最新的那组 CSV 文件"""
    if not os.path.exists(results_dir):
        print(f"❌ 错误：找不到目录 {results_dir}")
        return None

    # 找所有 resource_utilization 文件，按时间排序
    files = glob.glob(os.path.join(results_dir, "*_resource_utilization.csv"))
    if not files:
        print("❌ 错误：在 results 目录下找不到 CSV 文件")
        return None

    # 获取最新的一个文件的前缀
    latest_file = max(files, key=os.path.getctime)
    prefix = latest_file.replace("_resource_utilization.csv", "")
    print(f"📂 正在分析最新一次实验数据：{os.path.basename(prefix)} ...")

    return {
        "res": f"{prefix}_resource_utilization.csv",
        "dep": f"{prefix}_deployment_details.csv",
        "fail": f"{prefix}_failed_nodes_analysis.csv",
        "sum": f"{prefix}_summary_statistics.csv",
        "time_metrics": f"{prefix}_metrics_by_time_interval.csv"
    }


def plot_acceptance_rate(dep_file):
    """画接受率的移动平均曲线"""
    if not os.path.exists(dep_file):
        print(f"⚠️ 文件不存在，跳过接受率图: {dep_file}")
        return

    try:
        df = pd.read_csv(dep_file)

        df['full_acc_ma'] = df['fully_deployed'].rolling(window=50).mean() * 100
        df['partial_acc_ma'] = df['partial_deployed'].rolling(window=50).mean() * 100

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['full_acc_ma'], label='完全成功率 (Moving Avg)', color='green', linewidth=2)
        plt.plot(df.index, df['partial_acc_ma'], label='部分成功率 (Moving Avg)', color='orange', alpha=0.7)

        plt.title('训练过程中的请求接受率趋势', fontsize=14)
        plt.xlabel('请求序列 (Request ID)', fontsize=12)
        plt.ylabel('接受率 (%)', fontsize=12)
        plt.legend()
        plt.tight_layout()

        # [修改] 保存到 image 目录
        save_path = os.path.join(IMAGE_DIR, 'plot_acceptance_rate.png')
        plt.savefig(save_path)
        print(f"✅ 已生成: {save_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️ 无法绘制接受率: {e}")


def plot_resource_utilization(res_file):
    """画资源利用率曲线"""
    if not os.path.exists(res_file):
        print(f"⚠️ 文件不存在，跳过资源利用率图: {res_file}")
        return

    try:
        df = pd.read_csv(res_file)

        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['cpu_utilization'], label='CPU 利用率', alpha=0.8)
        plt.plot(df.index, df['bw_utilization'], label='带宽 (BW) 利用率', alpha=0.8)
        plt.plot(df.index, df['mem_utilization'], label='内存 (MEM) 利用率', alpha=0.8)

        plt.title('网络资源利用率变化', fontsize=14)
        plt.xlabel('时间步 (Steps)', fontsize=12)
        plt.ylabel('利用率 (0-1)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # [修改] 保存到 image 目录
        save_path = os.path.join(IMAGE_DIR, 'plot_resource_utilization.png')
        plt.savefig(save_path)
        print(f"✅ 已生成: {save_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️ 无法绘制资源利用率: {e}")


def plot_failure_analysis(dep_file):
    """画失败原因统计饼图"""
    if not os.path.exists(dep_file):
        return

    try:
        df = pd.read_csv(dep_file)
        all_reasons = []
        for item in df['failure_reasons'].dropna():
            if item:
                all_reasons.extend(item.split(','))

        if not all_reasons:
            print("⚠️ 没有检测到失败数据，跳过失败分析图")
            return

        from collections import Counter
        reason_counts = Counter(all_reasons)

        plt.figure(figsize=(8, 8))
        plt.pie(reason_counts.values(), labels=reason_counts.keys(), autopct='%1.1f%%', startangle=140)
        plt.title('请求失败原因分布', fontsize=14)
        plt.tight_layout()

        # [修改] 保存到 image 目录
        save_path = os.path.join(IMAGE_DIR, 'plot_failure_reasons.png')
        plt.savefig(save_path)
        print(f"✅ 已生成: {save_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️ 无法绘制失败原因: {e}")


def plot_top_failed_nodes(fail_file):
    """画最容易堵塞的节点 TOP 10"""
    if not os.path.exists(fail_file):
        print(f"⚠️ 文件不存在，跳过失败节点图: {fail_file}")
        return

    try:
        df = pd.read_csv(fail_file)
        if df.empty:
            return

        top_df = df.head(10).sort_values('failure_count', ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(top_df['node_id'].astype(str), top_df['failure_count'], color='salmon')
        plt.xlabel('失败次数 (阻塞次数)', fontsize=12)
        plt.ylabel('节点 ID', fontsize=12)
        plt.title('最容易发生阻塞的节点 Top 10', fontsize=14)
        plt.tight_layout()

        # [修改] 保存到 image 目录
        save_path = os.path.join(IMAGE_DIR, 'plot_failed_nodes.png')
        plt.savefig(save_path)
        print(f"✅ 已生成: {save_path}")
        plt.close()
    except Exception as e:
        print(f"⚠️ 无法绘制失败节点: {e}")


def plot_comparison_chart(msfc_file, stb_file):
    """绘制 HIRL-MSFC-CE vs MSFC-CE 真实资源消耗对比柱状图"""

    df_msfc = pd.read_csv(msfc_file)
    df_stb  = pd.read_csv(stb_file)

    df_msfc = df_msfc.sort_values("Time_Interval")
    df_stb  = df_stb.sort_values("Time_Interval")

    labels = df_msfc['Time_Interval'].astype(str).tolist()

    msfc_data = df_msfc['Total_CPU_Consumed'].tolist()
    stb_data  = df_stb['Total_CPU_Consumed'].tolist()

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))

    plt.bar(x - width/2, stb_data, width, label="STB")
    plt.bar(x + width/2, msfc_data, width, label="MSFC-CE")

    plt.ylabel("CPU 资源消耗总量")
    plt.xlabel("时间间隔")
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("image/resource_consumption_comparison.png", dpi=300)
    plt.close()

    print("✅ STB vs MSFC 真实资源消耗对比图已生成")

if __name__ == "__main__":
    # 1. 确保 image 文件夹存在
    ensure_image_dir()

    print("🚀 开始生成分析图表...")

    # 2. 找到最新文件
    files = find_latest_files()

    if files:
        # 3. 生成所有图表到 image 目录
        plot_acceptance_rate(files['dep'])
        plot_resource_utilization(files['res'])
        plot_failure_analysis(files['dep'])
        plot_top_failed_nodes(files['fail'])
        plot_comparison_chart(files['time_metrics'])

        print(f"\n🎉 所有图表已生成在 '{IMAGE_DIR}' 目录下。")