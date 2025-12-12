#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断：时间步不足导致的资源竞争问题

分析为什么只有400时间步会导致60%阻塞率
"""

import pickle
import numpy as np
from pathlib import Path

# 加载请求数据
INPUT_DIR = Path(r'E:\pycharmworkspace\SFC-master\HIRL-MSFC-CE (1)\generate_requests_depend_on_poisson\data_output')

with open(INPUT_DIR / 'sorted_requests.pkl', 'rb') as f:
    requests = pickle.load(f)

print("=" * 80)
print("🔍 时间步充足性分析")
print("=" * 80)

# 1. 基本统计
total_requests = len(requests)
arrival_times = [req['time'] for req in requests]
holding_times = [req['holding_time'] for req in requests]

min_time = min(arrival_times)
max_time = max(arrival_times)
time_span = max_time - min_time

print(f"\n📊 请求数据统计:")
print(f"  总请求数: {total_requests}")
print(f"  时间跨度: {min_time:.1f} → {max_time:.1f} (共 {time_span:.1f} 时间步)")
print(f"  平均到达间隔: {time_span / total_requests:.2f} 时间步/请求")
print(f"  平均持续时间: {np.mean(holding_times):.2f} 时间步")

# 2. 分析并发度
max_concurrent = 0
concurrent_at_t = {}

for t in range(int(min_time), int(max_time) + 1):
    active_requests = 0
    for req in requests:
        if req['time'] <= t < req['time'] + req['holding_time']:
            active_requests += 1
    concurrent_at_t[t] = active_requests
    max_concurrent = max(max_concurrent, active_requests)

avg_concurrent = np.mean(list(concurrent_at_t.values()))

print(f"\n📈 并发度分析:")
print(f"  最大并发数: {max_concurrent} 个请求")
print(f"  平均并发数: {avg_concurrent:.1f} 个请求")
print(f"  并发度比率: {avg_concurrent / total_requests:.1%}")

# 3. 资源需求分析
total_cpu_demand = 0
total_mem_demand = 0
total_bw_demand = 0

for req in requests:
    cpu_per_req = sum(req.get('cpu_origin', [0]))
    mem_per_req = sum(req.get('memory_origin', [0]))
    bw_per_req = req.get('bw_origin', 0) * len(req['dest'])

    total_cpu_demand += cpu_per_req
    total_mem_demand += mem_per_req
    total_bw_demand += bw_per_req

# 网络容量（28个节点）
NUM_NODES = 28
CPU_PER_NODE = 80.0
MEM_PER_NODE = 60.0
BW_PER_LINK = 80.0
NUM_LINKS = 45

total_cpu_capacity = NUM_NODES * CPU_PER_NODE
total_mem_capacity = NUM_NODES * MEM_PER_NODE
total_bw_capacity = NUM_LINKS * BW_PER_LINK

print(f"\n💾 资源供需对比:")
print(f"  CPU:")
print(f"    网络总容量: {total_cpu_capacity:.0f}")
print(f"    峰值并发需求: {avg_concurrent * (total_cpu_demand / total_requests):.0f}")
print(f"    利用率: {(avg_concurrent * total_cpu_demand / total_requests) / total_cpu_capacity:.1%}")

print(f"  内存:")
print(f"    网络总容量: {total_mem_capacity:.0f}")
print(f"    峰值并发需求: {avg_concurrent * (total_mem_demand / total_requests):.0f}")
print(f"    利用率: {(avg_concurrent * total_mem_demand / total_requests) / total_mem_capacity:.1%}")

print(f"  带宽:")
print(f"    网络总容量: {total_bw_capacity:.0f}")
print(f"    峰值并发需求: {avg_concurrent * (total_bw_demand / total_requests):.0f}")
print(f"    利用率: {(avg_concurrent * total_bw_demand / total_requests) / total_bw_capacity:.1%}")

# 4. 问题诊断
print("\n" + "=" * 80)
print("🚨 问题诊断")
print("=" * 80)

if time_span < 500:
    print(f"\n❌ 严重问题：时间跨度过短 ({time_span:.0f} < 500)")
    print("   原因：请求过于密集，资源竞争激烈")
    print("   后果：")
    print("     - 平均 {:.1f} 个请求同时存在".format(avg_concurrent))
    print("     - 资源利用率超高，大量请求阻塞")
    print("     - Expert无法找到可用资源，BackupPolicy频繁激活")

if avg_concurrent > 50:
    print(f"\n❌ 严重问题：平均并发度过高 ({avg_concurrent:.0f} > 50)")
    print("   原因：请求持续时间长 + 到达密集")
    print("   后果：")
    print("     - 网络资源长期被占用")
    print("     - 新请求难以找到可用路径")
    print("     - 阻塞率高达60%+")

# 5. 解决方案
print("\n" + "=" * 80)
print("💡 解决方案")
print("=" * 80)

print("\n🔴 方案1：增加时间跨度（最有效）")
print("   修改 data_generator.py:")
print(f"     当前时间跨度: {time_span:.0f}")
print(f"     建议时间跨度: {time_span * 5:.0f} (增加5倍)")
print("   操作：")
print("     找到 arrival_time 生成代码")
print("     降低 lambda 参数（泊松分布）到原来的 1/5")
print("   效果：")
print("     - 平均并发数: {:.1f} → {:.1f}".format(avg_concurrent, avg_concurrent / 5))
print("     - 预期阻塞率: 60% → 20-30%")

print("\n🟡 方案2：减少请求持续时间")
print("   修改 data_generator.py:")
print(f"     当前平均持续: {np.mean(holding_times):.1f}")
print(f"     建议平均持续: {np.mean(holding_times) / 2:.1f} (减半)")
print("   操作：")
print("     找到 holding_time 生成代码")
print("     将均值参数减半")
print("   效果：")
print("     - 资源释放更快")
print("     - 预期阻塞率: 60% → 35-45%")

print("\n🟢 方案3：增加网络容量（治标不治本）")
print("   修改 hyperparameters.py:")
print("   CAPACITIES = {")
print(f"       'cpu': {CPU_PER_NODE} → {CPU_PER_NODE * 1.5} (+50%)")
print(f"       'memory': {MEM_PER_NODE} → {MEM_PER_NODE * 1.5} (+50%)")
print(f"       'bandwidth': {BW_PER_LINK} → {BW_PER_LINK * 1.5} (+50%)")
print("   }")
print("   效果：")
print("     - 预期阻塞率: 60% → 45-50%")
print("     - 但并发度问题仍存在")

print("\n🔵 方案4：减少请求数量")
print("   修改 data_generator.py:")
print(f"     当前请求数: {total_requests}")
print(f"     建议请求数: {total_requests // 3} (减少到1/3)")
print("   操作：")
print("     修改生成循环次数")
print("   效果：")
print("     - 平均并发数大幅降低")
print("     - 预期阻塞率: 60% → 15-25%")

# 6. 优先级建议
print("\n" + "=" * 80)
print("⭐ 推荐行动方案")
print("=" * 80)

print("\n立即执行（必须）：")
print("  1️⃣  方案1：增加时间跨度到2000+ (最关键)")
print("     python data_generator.py --time_span 2000")
print("\n  2️⃣  方案2：减少持续时间到平均50以下")
print("     修改 holding_time 均值参数")

print("\n可选执行（如果效果仍不佳）：")
print("  3️⃣  方案3：增加容量50%")
print("  4️⃣  方案4：减少请求数到600左右")

print("\n预期效果：")
print("  执行方案1+2后：")
print("    - Ep 0 阻塞率: 60% → 25-35%")
print("    - Ep 100 阻塞率: 60% → 10-15%")
print("    - BackupPolicy激活率: 47.9% → <5%")
print("    - 最终接受率: 39% → 90-95%")

print("\n" + "=" * 80)