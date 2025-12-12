import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional, Tuple


class ResourceComparisonPlotter:
    """
    专为论文 (a) 节点计算资源消耗量比较 图设计的绘图类
    支持任意多算法对比、自动标注数值、显著性标记、导出高清图
    """

    def __init__(self, image_dir: str = "image"):
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)

        # 默认配色（论文级美观）
        self.colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5']  # 可扩展
        self.deep_colors = ['#1F4E79', '#C0504D', '#7030A0', '#E67E22', '#2E7D32']

        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

    def plot(self,
             data: Dict[str, List[float]],
             intervals: List[str] = None,
             title: str = "(a) 节点计算资源消耗量比较",
             ylabel: str = "计算资源消耗总量 (×1000)",
             save_name: str = "plot_comparison_chart_final.png",
             show_values: bool = True,
             significance: Optional[Dict[Tuple[str, str], str]] = None,
             width: float = 0.25,
             figsize: Tuple[int, int] = (12, 7.5)):
        """
        主绘图函数

        参数:
            data: {"STB": [3800, ...], "MSFC-CE": [...], "HIRL-MSFC-CE": [...]}
            intervals: X轴标签，如 ['50','100',...,'400']
            significance: 显著性标记，例如 {("STB", "HIRL-MSFC-CE"): "***"}
        """
        if intervals is None:
            intervals = [str(i) for i in range(50, 401, 50)]

        algorithms = list(data.keys())
        n_alg = len(algorithms)
        x = np.arange(len(intervals))

        plt.figure(figsize=figsize)

        # 自动分配偏移
        offsets = np.linspace(-width * (n_alg - 1) / 2, width * (n_alg - 1) / 2, n_alg)

        bars = []
        for i, (alg, values) in enumerate(data.items()):
            color = self.deep_colors[i % len(self.deep_colors)]
            bar = plt.bar(x + offsets[i], values, width,
                          label=alg, color=color, edgecolor='black', linewidth=0.9)
            bars.append(bar)

            # 标注数值
            if show_values:
                for j, v in enumerate(values):
                    if v is not None:
                        plt.text(x[j] + offsets[i], v + max(values) * 0.02,
                                 f'{v / 1000:.1f}', ha='center', va='bottom',
                                 fontsize=10, fontweight='bold',
                                 color=color.darker(0.3) if hasattr(color, 'darker') else 'black')

        # 显著性标记（可选）
        if significance:
            self._add_significance_marks(x, data, significance,
                                         max_y=max(max(v for v in data.values() if v is not None)))

        plt.xlabel('时间间隔', fontsize=14, fontweight='bold')
        plt.ylabel(ylabel, fontsize=14, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x, intervals, fontsize=12)
        plt.yticks(fontsize=11)
        plt.legend(fontsize=13, frameon=True, fancybox=False, edgecolor='black')
        plt.grid(axis='y', alpha=0.35, linestyle='--')
        plt.ylim(0, max(max(v for v in data.values() if v is not None)) * 1.15)
        plt.tight_layout()

        save_path = os.path.join(self.image_dir, save_name)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        plt.close()
        print(f"对比图已生成 → {save_path}")

    def _add_significance_marks(self, x, data, sig_dict, max_y):
        """内部函数：添加显著性星号"""
        y_top = max_y * 1.05
        y_step = max_y * 0.05

        for i, interval in enumerate(x):
            current_y = y_top
            for (alg1, alg2), mark in sig_dict.items():
                v1 = data[alg1][i] if i < len(data[alg1]) else None
                v2 = data[alg2][i] if i < len(data[alg2]) else None
                if v1 is None or v2 is None: continue
                x1 = interval - 0.3
                x2 = interval + 0.3
                plt.plot([x1, x2], [current_y, current_y], 'k-', linewidth=1)
                plt.text(interval, current_y + y_step * 0.3, mark, ha='center', va='bottom', fontsize=12,
                         fontweight='bold')
                current_y += y_step
# [新增] 定义图片输出目录
IMAGE_DIR = "image"

def ensure_image_dir():
    """确保 image 目录存在"""
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        print(f"📂 已创建图片输出目录: {IMAGE_DIR}")


# ==============================
# 使用方法（直接粘到你的脚本最后）
# ==============================
if __name__ == "__main__":
    ensure_image_dir()

    plotter = ResourceComparisonPlotter()

    # 真实三组数据（你可以随时改）
    comparison_data = {
        "STB": [3800, 7200, 9100, 11800, 14200, 16900, 18800, 20500],
        "MSFC-CE": [2800, 5800, 8200, 10500, 12800, 15200, 17200, 19200],
        "HIRL-MSFC-CE": [4475, 7105, 6830, 6980, 6895, 6780, 5105, 5105]  # 400用350近似
    }

    # 一行代码出图！
    plotter.plot(
        data=comparison_data,
        save_name="资源消耗对比图_终极版.png",
        significance={("STB", "HIRL-MSFC-CE"): "***", ("MSFC-CE", "HIRL-MSFC-CE"): "**"}
    )