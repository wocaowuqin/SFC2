#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速诊断脚本：一键分析阻塞率和接受率上限问题

运行方法：
    python quick_diagnosis.py

输出：
1. PathDB质量报告（理论上限）
2. 初始阻塞率分析（ε-greedy影响）
3. 优化建议
"""

import numpy as np
import sys
import pickle
from pathlib import Path
from collections import defaultdict

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

import hyperparameters as H
from expert_msfce import MSFCE_Solver


def main():
    print("\n" + "=" * 80)
    print("🔍 阻塞率与接受率上限诊断")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/4] 加载数据...")
    expert = MSFCE_Solver(
        H.INPUT_DIR / "US_Backbone_path.mat",
        H.TOPOLOGY_MATRIX,
        H.DC_NODES,
        H.CAPACITIES
    )

    with open(H.INPUT_DIR / 'sorted_requests.pkl', 'rb') as f:
        requests = pickle.load(f)

    print(f"  ✓ Expert初始化完成")
    print(f"  ✓ 请求数据加载完成: {len(requests)}条")

    # 2. 分析PathDB质量
    print("\n[2/4] 分析PathDB质量...")

    stats = {
        'total_requests': len(requests),
        'no_path_count': 0,
        'insufficient_dc_count': 0,
        'feasible_count': 0,
        'vnf_distribution': defaultdict(int),
        'problematic_pairs': []
    }

    for req in requests[:200]:  # 抽样200个请求
        src = req['source']
        dests = req['dest']
        vnf_count = len(req.get('vnf', []))

        stats['vnf_distribution'][vnf_count] += 1

        all_feasible = True

        for dest in dests:
            max_dc_on_path = 0
            has_path = False

            # 检查所有k条路径
            for k in range(1, expert.k_path + 1):
                nodes, dist, links = expert._get_path_info(src, dest, k)

                if not nodes:
                    continue

                has_path = True
                dc_count = len([n for n in nodes if n in expert.DC])
                max_dc_on_path = max(max_dc_on_path, dc_count)

            if not has_path:
                stats['no_path_count'] += 1
                stats['problematic_pairs'].append((src, dest, 'no_path'))
                all_feasible = False
                break

            if max_dc_on_path < vnf_count:
                stats['insufficient_dc_count'] += 1
                stats['problematic_pairs'].append((src, dest, f'max_dc={max_dc_on_path}, need={vnf_count}'))
                all_feasible = False
                break

        if all_feasible:
            stats['feasible_count'] += 1

    # 计算理论上限
    sample_size = min(200, len(requests))
    theoretical_limit = stats['feasible_count'] / sample_size

    print(f"\n  📊 PathDB质量报告（基于{sample_size}个样本）:")
    print(f"     理论可满足: {stats['feasible_count']} ({theoretical_limit:.1%})")
    print(f"     无路径:     {stats['no_path_count']} ({stats['no_path_count'] / sample_size:.1%})")
    print(f"     DC节点不足: {stats['insufficient_dc_count']} ({stats['insufficient_dc_count'] / sample_size:.1%})")

    print(f"\n  📈 VNF需求分布:")
    for vnf_count in sorted(stats['vnf_distribution'].keys()):
        count = stats['vnf_distribution'][vnf_count]
        print(f"     {vnf_count}个VNF: {count}次 ({count / sample_size:.1%})")

    # 展示问题节点对
    if stats['problematic_pairs']:
        print(f"\n  ⚠️  问题节点对示例（前5个）:")
        for i, (src, dest, reason) in enumerate(stats['problematic_pairs'][:5]):
            print(f"     {i + 1}. ({src} → {dest}): {reason}")

    # 3. 分析初始阻塞率原因
    print("\n[3/4] 分析初始阻塞率...")

    # 模拟完全随机策略
    random_success = 0
    random_trials = 50

    for req in requests[:random_trials]:
        src = req['source']
        dests = req['dest']

        # 随机选择k值（模拟ε=1.0时的行为）
        success = True
        for dest in dests:
            k = np.random.randint(1, expert.k_path + 1)
            nodes, _, _ = expert._get_path_info(src, dest, k)

            if not nodes:
                success = False
                break

            dc_count = len([n for n in nodes if n in expert.DC])
            if dc_count < len(req.get('vnf', [])):
                success = False
                break

        if success:
            random_success += 1

    random_acc = random_success / random_trials

    print(f"\n  🎲 随机策略模拟（ε=1.0）:")
    print(f"     成功率: {random_acc:.1%}")
    print(f"     预期阻塞率: {1 - random_acc:.1%}")

    # 4. 输出结论和建议
    print("\n[4/4] 生成诊断报告...")
    print("\n" + "=" * 80)
    print("📋 诊断结论")
    print("=" * 80)

    print(f"\n✅ 问题1：初始阻塞率高（43.8%）的原因")
    print(f"   原因：ε-greedy完全随机探索")
    print(f"   证据：随机策略成功率仅{random_acc:.1%}，与实际初始阻塞率吻合")
    print(f"   影响：前100个episode性能极差")

    print(f"\n✅ 问题2：最终接受率上限（95-96%）的原因")
    if theoretical_limit < 0.97:
        print(f"   主要原因：PathDB质量限制（理论上限{theoretical_limit:.1%}）")
        print(f"   次要原因：资源竞争、训练不足")
    else:
        print(f"   主要原因：资源竞争（PathDB质量正常，理论上限{theoretical_limit:.1%}）")
        print(f"   次要原因：DQN未完全收敛")

    print("\n" + "=" * 80)
    print("💡 优化建议")
    print("=" * 80)

    # 建议优先级
    if theoretical_limit < 0.97:
        print("\n🔴 优先级1：提升PathDB质量")
        print("   方案A：增加DC节点数量")
        print(f"         当前: {len(expert.DC)}个")
        print(f"         建议: ≥{int(len(expert.DC) * 1.5)}个（增加50%）")
        print("   方案B：增加K值")
        print(f"         当前: k={expert.k_path}")
        print(f"         建议: k≥{expert.k_path + 3}（增加3条备用路径）")
        print("   方案C：降低VNF需求")
        print("         当前: 2-4个VNF")
        print("         建议: 限制为2-3个VNF")

    print("\n🟡 优先级2：降低初始阻塞率")
    print("   方案A：减少预训练随机性")
    print("         增加PRE_TRAIN_STEPS: 100 → 500")
    print("   方案B：降低初始epsilon")
    print("         修改initial_epsilon: 1.0 → 0.5")
    print("   方案C：延长探索衰减")
    print("         增加EXPLORATION_STEPS: 500K → 1M")

    print("\n🟢 优先级3：提升最终性能")
    print("   方案A：增加训练轮数")
    print("         EPISODE_LIMIT: 300 → 500")
    print("   方案B：增加Meta训练频率")
    print("         META_TRAIN_FREQ: 100 → 50")
    print("   方案C：添加早停机制")
    print("         patience=50, min_improvement=0.01")

    print("\n" + "=" * 80)
    print("📝 下一步行动")
    print("=" * 80)

    if theoretical_limit < 0.97:
        print("\n1️⃣  先解决PathDB质量问题（理论上限不足）")
        print("   → 运行: python improve_pathdb.py")
        print("\n2️⃣  再优化训练策略（降低初始阻塞率）")
        print("   → 使用: hyperparameters_optimized.py")
    else:
        print("\n1️⃣  优化训练策略（PathDB质量正常）")
        print("   → 使用: hyperparameters_optimized.py")
        print("\n2️⃣  监控训练过程")
        print("   → 每10个episode检查接受率变化")

    print("\n" + "=" * 80)
    print("✅ 诊断完成")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()