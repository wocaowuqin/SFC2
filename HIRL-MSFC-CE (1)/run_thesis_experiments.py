#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : run_thesis_experiments.py
# @Desc    : 复现论文第三章 MSFC-CE 算法实验 (带宽/节点资源敏感性分析)

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from tqdm import tqdm  # 进度条库, 如未安装请 pip install tqdm

# 导入环境和专家
import hyperparameters as H
from hirl_sfc_env import SFC_HIRL_Env

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ThesisExp")

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def run_simulation_with_capacity(bw_cap, node_cap, experiment_name="exp"):
    """
    使用指定的资源容量运行一次完整的 MSFC-CE 仿真
    """
    # 1. 临时修改资源配置
    current_capacities = {
        'bandwidth': bw_cap,
        'cpu': node_cap,
        'memory': node_cap * 0.75  # 论文中存储通常比计算少，按比例缩放或保持一致
    }

    # 2. 初始化环境 (加载数据)
    # 注意：必须重新初始化环境以应用新的资源上限
    env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, current_capacities)

    total_requests = 0
    accepted_requests = 0

    # 3. 仿真主循环
    # 这里的逻辑与 train_hirl_sfc.py 的预训练部分类似，但只做推断
    pbar = tqdm(total=env.T, desc=f"Simulating {experiment_name} BW={bw_cap} Node={node_cap}", leave=False)

    current_request, _ = env.reset_request()

    while env.t < env.T:
        if current_request is None:
            # 如果当前没有请求，推进时间
            current_request, _ = env.reset_request()
            pbar.update(env.t - pbar.n)
            if current_request is None and env.t >= env.T:
                break
            continue

        total_requests += 1

        # --- 调用专家算法 ---
        try:
            # 获取当前网络状态
            net_state = env._get_network_state_dict()

            # 专家求解
            # solve_request_for_expert 返回 (tree_struct, trajectory)
            # 如果返回 None，说明专家认为无法部署（阻塞）
            expert_result = env.expert.solve_request_for_expert(current_request, net_state)

            if expert_result and expert_result[0] is not None:
                _, trajectory = expert_result

                # 专家找到了解，我们在环境中执行它
                # 注意：必须一步步执行 trajectory，才能让环境扣除资源并推进状态
                request_success = True
                for goal_idx, action_tuple, _ in trajectory:
                    # action_tuple 格式: (p_idx, k_idx, placement)
                    # 转换动作格式传给 env (env会自动处理)
                    _, _, sub_done, req_done = env.step_low_level(goal_idx, action_tuple)

                    # 如果某一步失败了（理论上专家说行就行，但为了健壮性）
                    if not sub_done and not req_done:
                        # 这种情况很少见，除非专家和环境状态不同步
                        request_success = False
                        break

                if request_success:
                    accepted_requests += 1
            else:
                # 专家无法求解 -> 阻塞
                pass

        except Exception as e:
            logger.error(f"Error processing request {current_request['id']}: {e}")

        # 准备下一个请求
        current_request, _ = env.reset_request()
        pbar.update(env.t - pbar.n)

    pbar.close()

    # 计算阻塞率
    blocking_rate = 1.0 - (accepted_requests / max(1, total_requests))
    logger.info(f"Result: BW={bw_cap}, Node={node_cap} -> Blocking Rate={blocking_rate:.2%}")

    return blocking_rate


def experiment_1_bandwidth_sensitivity():
    """
    实验 1: 验证带宽资源对阻塞率的影响 (复现论文图 3.9)
    固定节点资源充裕 (例如 200)，变化带宽 (20-100)
    """
    logger.info("\n=== 开始实验 1: 带宽资源敏感性分析 ===")

    # 实验参数
    node_cap_fixed = 200  # 充足的节点资源
    bw_caps = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    results = []

    for bw in bw_caps:
        br = run_simulation_with_capacity(bw, node_cap_fixed, "Exp1")
        results.append(br)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(bw_caps, results, 'g-o', linewidth=2, markersize=8, label='MSFC-CE')
    plt.title('业务请求阻塞率随带宽资源容量的变化 (Exp 1)')
    plt.xlabel('网络链路带宽资源量')
    plt.ylabel('业务请求阻塞率')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(H.OUTPUT_DIR / 'exp1_bandwidth_sensitivity.png')
    logger.info(f"实验 1 完成，图表已保存至 {H.OUTPUT_DIR / 'exp1_bandwidth_sensitivity.png'}")

    # 保存数据
    df = pd.DataFrame({'Bandwidth': bw_caps, 'Blocking_Rate': results})
    df.to_csv(H.OUTPUT_DIR / 'exp1_results.csv', index=False)


def experiment_2_node_resource_sensitivity():
    """
    实验 2: 验证节点资源对阻塞率的影响 (复现论文图 3.10)
    固定带宽资源充裕 (例如 100)，变化节点资源 (90-290)
    """
    logger.info("\n=== 开始实验 2: 节点资源敏感性分析 ===")

    # 实验参数
    bw_cap_fixed = 100  # 充足的带宽资源
    node_caps = [90, 130, 170, 210, 250, 290]
    results = []

    for node_cap in node_caps:
        br = run_simulation_with_capacity(bw_cap_fixed, node_cap, "Exp2")
        results.append(br)

    # 绘图
    plt.figure(figsize=(8, 5))
    plt.plot(node_caps, results, 'y-o', linewidth=2, markersize=8, label='MSFC-CE')
    plt.title('业务请求阻塞率随节点资源容量的变化 (Exp 2)')
    plt.xlabel('网络节点资源容量')
    plt.ylabel('业务请求阻塞率')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(H.OUTPUT_DIR / 'exp2_node_sensitivity.png')
    logger.info(f"实验 2 完成，图表已保存至 {H.OUTPUT_DIR / 'exp2_node_sensitivity.png'}")

    # 保存数据
    df = pd.DataFrame({'Node_Cap': node_caps, 'Blocking_Rate': results})
    df.to_csv(H.OUTPUT_DIR / 'exp2_results.csv', index=False)


if __name__ == "__main__":
    # 确保输出目录存在
    if not H.OUTPUT_DIR.exists():
        H.OUTPUT_DIR.mkdir(parents=True)

    print(f"实验结果将保存至: {H.OUTPUT_DIR}")

    # 运行实验
    experiment_1_bandwidth_sensitivity()
    experiment_2_node_resource_sensitivity()

    print("\n所有实验结束。")