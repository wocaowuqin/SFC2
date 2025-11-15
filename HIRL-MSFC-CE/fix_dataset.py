import scipy.io as sio
import numpy as np
from pathlib import Path
import hyperparameters as H
from expert_msfce import MSFCE_Solver, parse_mat_request


def analyze_dataset():
    """分析数据集的DC覆盖率"""

    print("=" * 80)
    print("数据集诊断工具")
    print("=" * 80)

    # 加载专家
    expert = MSFCE_Solver(
        H.INPUT_DIR / "US_Backbone_path.mat",
        H.TOPOLOGY_MATRIX,
        H.DC_NODES,
        H.CAPACITIES
    )

    # 加载请求
    reqs = sio.loadmat(H.INPUT_DIR / "sorted_requests.mat")['sorted_requests']
    requests = [parse_mat_request(r) for r in reqs]

    print(f"\n✅ 加载了 {len(requests)} 个请求")
    print(f"✅ DC 节点: {len(expert.DC)} 个")
    print(f"✅ 总节点: {expert.node_num} 个")
    print(f"✅ DC 覆盖率: {len(expert.DC) / expert.node_num * 100:.1f}%")

    # 统计VNF需求分布
    vnf_counts = [len(r['vnf']) for r in requests]
    print(f"\n📊 VNF 需求统计:")
    print(f"   最小: {min(vnf_counts)}")
    print(f"   最大: {max(vnf_counts)}")
    print(f"   平均: {np.mean(vnf_counts):.2f}")
    print(f"   中位数: {np.median(vnf_counts):.0f}")

    # 统计目的地数量分布
    dest_counts = [len(r['dest']) for r in requests]
    print(f"\n📊 目的地数量统计:")
    print(f"   最小: {min(dest_counts)}")
    print(f"   最大: {max(dest_counts)}")
    print(f"   平均: {np.mean(dest_counts):.2f}")

    # 分析路径DC覆盖率
    print(f"\n📊 路径 DC 覆盖率分析 (采样100个请求):")

    sufficient_paths = 0
    total_checks = 0

    for req in requests[:100]:
        for d_idx, dest in enumerate(req['dest']):
            src = req['source']
            vnf_needed = len(req['vnf'])

            has_sufficient_path = False

            for k in range(1, expert.k_path_count + 1):
                paths, _, _ = expert._get_path_from_db(src, dest, k)
                if not paths:
                    continue

                dc_on_path = [n for n in paths if n in expert.DC]

                if len(dc_on_path) >= vnf_needed:
                    has_sufficient_path = True
                    break

            total_checks += 1
            if has_sufficient_path:
                sufficient_paths += 1

    coverage = sufficient_paths / total_checks * 100 if total_checks > 0 else 0
    print(f"   ✅ 有足够DC节点的路径: {sufficient_paths}/{total_checks} ({coverage:.1f}%)")

    if coverage < 50:
        print(f"\n🚨 严重问题: 只有 {coverage:.1f}% 的路径有足够的DC节点!")
        print("\n建议的解决方案:")
        print("1. 增加 DC_NODES (推荐)")
        print("2. 减少请求的 VNF 数量")
        print("3. 重新生成 K-最短路径,使用 DC-aware 算法")
    elif coverage < 80:
        print(f"\n⚠️ 警告: 只有 {coverage:.1f}% 的路径有足够的DC节点")
        print("建议增加 DC_NODES 或优化路径算法")
    else:
        print(f"\n✅ 良好: {coverage:.1f}% 的路径有足够的DC节点")


def suggest_dc_nodes():
    """建议新增的DC节点"""

    print("\n" + "=" * 80)
    print("DC 节点建议")
    print("=" * 80)

    current_dc = set(H.DC_NODES)
    total_nodes = H.TOPOLOGY_MATRIX.shape[0]

    # 计算每个节点的度数 (连接数)
    degrees = []
    for i in range(total_nodes):
        node_id = i + 1
        degree = 0
        for j in range(total_nodes):
            if not np.isinf(H.TOPOLOGY_MATRIX[i, j]) and H.TOPOLOGY_MATRIX[i, j] > 0:
                degree += 1
        degrees.append((node_id, degree))

    # 按度数排序
    degrees.sort(key=lambda x: x[1], reverse=True)

    print(f"\n当前 DC 节点: {sorted(current_dc)}")
    print(f"当前 DC 数量: {len(current_dc)}/{total_nodes}")

    # 建议新增的节点 (高度数且非DC)
    suggestions = []
    for node_id, degree in degrees:
        if node_id not in current_dc and len(suggestions) < 8:  # 建议增加8个
            suggestions.append(node_id)

    if suggestions:
        print(f"\n建议新增的 DC 节点 (按连接数排序):")
        for node in suggestions:
            degree = next(d for n, d in degrees if n == node)
            print(f"  节点 {node}: {degree} 个连接")

        new_dc_list = sorted(list(current_dc) + suggestions)
        print(f"\n完整的 DC_NODES 列表 (复制到 hyperparameters.py):")
        print(f"DC_NODES = {new_dc_list}")
    else:
        print("\n所有节点都已是 DC 节点")


if __name__ == "__main__":
    analyze_dataset()
    suggest_dc_nodes()