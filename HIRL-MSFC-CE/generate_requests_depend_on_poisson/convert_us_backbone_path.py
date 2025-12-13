#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_us_backbone_path.py

将 US_Backbone_path.mat 转换为 Python pickle 格式

Author: Claude
Date: 2025-12-13
"""

import scipy.io
import pickle
import numpy as np
import os
from pathlib import Path


def convert_mat_to_pkl(mat_file, output_file=None):
    """
    将 US_Backbone_path.mat 转换为 .pkl 格式

    Args:
        mat_file: 输入的 .mat 文件路径
        output_file: 输出的 .pkl 文件路径（可选，默认为同名.pkl）

    Returns:
        转换后的数据字典
    """

    # 确定输出文件名
    if output_file is None:
        output_file = Path(mat_file).with_suffix('.pkl')

    print("=" * 70)
    print("US_Backbone_path.mat → PKL 转换器")
    print("=" * 70)

    # ========== 加载 MAT 文件 ==========
    print(f"\n[1/5] 加载 MAT 文件: {mat_file}")

    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"文件不存在: {mat_file}")

    mat_data = scipy.io.loadmat(mat_file)
    paths_array = mat_data['Paths']

    print(f"  ✓ 数组形状: {paths_array.shape}")
    print(f"  ✓ 字段: {list(paths_array.dtype.names)}")

    # ========== 文件说明 ==========
    print(f"\n[2/5] 文件说明")
    print("""
  这是一个预计算的路径数据库，包含：
    - 28x28 节点对之间的 K 条最短路径
    - 每条路径的节点序列、跳数、距离
    - 用于加速专家算法的路径查找
    """)

    # ========== 转换数据结构 ==========
    print(f"[3/5] 转换数据结构...")

    path_database = {}
    total_paths = 0

    for i in range(paths_array.shape[0]):
        for j in range(paths_array.shape[1]):
            if i != j:  # 跳过自己到自己
                cell = paths_array[i, j]

                # 提取字段
                paths = cell['paths']
                hops = cell['pathshops']
                distances = cell['pathsdistance']
                link_ids = cell['link_ids']

                if paths.size > 0:
                    k_paths = []

                    # 处理每条路径
                    for k in range(paths.shape[0]):
                        path_nodes = paths[k]
                        path_nodes = path_nodes[path_nodes > 0]  # 移除填充的0

                        if len(path_nodes) > 0:
                            k_paths.append({
                                'nodes': path_nodes.tolist(),
                                'hops': int(hops[k, 0]),
                                'distance': int(distances[k, 0]),
                                'link_ids': link_ids[k][link_ids[k] > 0].tolist()
                            })
                            total_paths += 1

                    if k_paths:
                        path_database[(i + 1, j + 1)] = k_paths  # 节点从1开始编号

    print(f"  ✓ 节点对数: {len(path_database)}")
    print(f"  ✓ 总路径数: {total_paths}")
    print(f"  ✓ 平均路径数: {total_paths / len(path_database):.1f} 条/节点对")

    # ========== 构建输出数据 ==========
    print(f"\n[4/5] 构建输出数据...")

    data = {
        'path_database': path_database,
        'metadata': {
            'num_nodes': 28,
            'num_node_pairs': len(path_database),
            'total_paths': total_paths,
            'avg_paths_per_pair': total_paths / len(path_database),
            'source_file': os.path.basename(mat_file),
            'description': 'Precomputed K-shortest paths for US Backbone topology',
            'format': {
                'path_database': 'dict[(src, dst)] -> list of paths',
                'path': 'dict with keys: nodes, hops, distance, link_ids'
            },
            'usage': 'paths = data["path_database"][(src, dst)]'
        }
    }

    # ========== 保存 PKL 文件 ==========
    print(f"[5/5] 保存 PKL 文件: {output_file}")

    with open(output_file, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(output_file)
    print(f"  ✓ 文件大小: {file_size / 1024:.2f} KB")

    # ========== 验证 ==========
    print(f"\n[验证] 重新加载测试...")

    with open(output_file, 'rb') as f:
        test_data = pickle.load(f)

    print(f"  ✓ 加载成功")
    print(f"  ✓ 节点对: {test_data['metadata']['num_node_pairs']}")
    print(f"  ✓ 路径数: {test_data['metadata']['total_paths']}")

    # ========== 显示示例 ==========
    print(f"\n[示例] 节点 1 → 节点 5 的路径:")

    if (1, 5) in test_data['path_database']:
        paths = test_data['path_database'][(1, 5)]
        for i, path in enumerate(paths[:3]):
            print(f"  路径 {i + 1}: {path['nodes']} "
                  f"(跳数={path['hops']}, 距离={path['distance']})")

    print("\n" + "=" * 70)
    print("✅ 转换完成！")
    print("=" * 70)

    return data


def demo_usage(pkl_file):
    """
    演示如何使用转换后的 PKL 文件
    """
    print("\n" + "=" * 70)
    print("📖 使用说明")
    print("=" * 70)

    print("""
1. 加载路径数据库:

   import pickle

   with open('US_Backbone_path.pkl', 'rb') as f:
       data = pickle.load(f)

   path_db = data['path_database']
   metadata = data['metadata']

2. 查找两节点间的路径:

   src, dst = 1, 5
   if (src, dst) in path_db:
       paths = path_db[(src, dst)]

       # 选择第一条路径（通常是最短路径）
       best_path = paths[0]

       print(f"节点序列: {best_path['nodes']}")
       print(f"跳数: {best_path['hops']}")
       print(f"距离: {best_path['distance']}")

3. 获取所有备选路径:

   for i, path in enumerate(paths):
       print(f"路径 {i+1}: {path['nodes']}")

4. 在代码中使用:

   # 在 expert_msfce.py 中
   class PathFinder:
       def __init__(self, path_db_file):
           with open(path_db_file, 'rb') as f:
               data = pickle.load(f)
           self.path_db = data['path_database']

       def find_path(self, src, dst, k=0):
           '''获取第 k 条路径'''
           if (src, dst) in self.path_db:
               return self.path_db[(src, dst)][k]['nodes']
           return None
""")

    # 实际演示
    print("\n" + "=" * 70)
    print("🔧 实际运行示例")
    print("=" * 70)

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)

    path_db = data['path_database']

    print("\n查找节点 1 到节点 10 的所有路径:")
    if (1, 10) in path_db:
        paths = path_db[(1, 10)]
        print(f"  找到 {len(paths)} 条路径\n")

        for i, path in enumerate(paths):
            print(f"  路径 {i + 1}:")
            print(f"    节点: {path['nodes']}")
            print(f"    跳数: {path['hops']}")
            print(f"    距离: {path['distance']}")


if __name__ == '__main__':
    import sys

    # 默认文件名
    mat_file = 'US_Backbone_path.mat'
    pkl_file = 'data_output/US_Backbone_path.pkl'

    # 命令行参数
    if len(sys.argv) > 1:
        mat_file = sys.argv[1]
    if len(sys.argv) > 2:
        pkl_file = sys.argv[2]

    # 执行转换
    try:
        data = convert_mat_to_pkl(mat_file, pkl_file)

        # 显示使用说明
        demo_usage(pkl_file)

        print("\n✅ 完成！")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()