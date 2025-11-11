#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : train_hirl_sfc.py

import numpy as np
from collections import namedtuple
import os

# ---
# ✅ 步骤 1: 在文件顶部添加导入
# ---
import matplotlib
matplotlib.use('Agg') # <-- ✅ 修复: 在导入 pyplot 之前设置后端
import matplotlib.pyplot as plt
import pandas as pd

# 导入 Keras/TF 和辅助工具
from tensorflow.keras.utils import to_categorical

# 导入所有自定义模块
import hyperparameters as H
from hirl_sfc_env import SFC_HIRL_Env
from hirl_sfc_models import MetaControllerNN, Hdqn_SFC
from hirl_sfc_agent import Agent_SFC

# (来自 atari)
ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])


def main():
    """主训练循环"""

    # --- 1. 配置 ---
    print("--- 1. 配置 HIRL-SFC 实验 ---")
    if not H.OUTPUT_DIR.exists():
        H.OUTPUT_DIR.mkdir(parents=True)

    # --- 3. 初始化 ---
    print("--- 2. 初始化环境、专家和代理 ---")
    env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, H.CAPACITIES)

    # 获取SFC环境的特定维度
    STATE_SHAPE = env.observation_space.shape
    NB_GOALS = env.NB_HIGH_LEVEL_GOALS
    NB_ACTIONS = env.NB_LOW_LEVEL_ACTIONS

    # 高层
    metacontroller = MetaControllerNN(state_shape=STATE_SHAPE, n_goals=NB_GOALS, lr=H.LR)

    # 低层 (H-DQN 架构)
    hdqn_net = Hdqn_SFC(state_shape=STATE_SHAPE, n_goals=NB_GOALS, n_actions=NB_ACTIONS, lr=H.LR)
    low_level_agent = Agent_SFC(
        net=hdqn_net,
        n_actions=NB_ACTIONS,
        mem_cap=H.EXP_MEMORY,
        exploration_steps=H.EXPLORATION_STEPS,
        train_freq=H.TRAIN_FREQ,
        hard_update=H.HARD_UPDATE_FREQUENCY,
        n_samples=H.BATCH_SIZE,
        gamma=H.GAMMA
    )
    low_level_agent.compile()  # 构建 Keras 训练模型

    print(f"状态向量大小: {STATE_SHAPE}")
    print(f"高层目标(子任务)数量: {NB_GOALS}")
    print(f"低层动作数量: {NB_ACTIONS}")

    # --- 4. 阶段 1: 模仿学习 (预训练) ---
    print(f"--- 3. 阶段 1: 模仿学习预训练 ( {H.PRE_TRAIN_STEPS} 步) ---")
    stepCount = 0
    current_request, high_level_state = env.reset_request()

    t = 0
    while t < H.PRE_TRAIN_STEPS:
        if current_request is None:
            break  # 仿真结束

        # 获取专家完整轨迹
        _, expert_traj = env.expert.solve_request_for_expert(
            current_request, env._get_network_state_dict()
        )
        if not expert_traj:
            current_request, high_level_state = env.reset_request()
            continue

        # 遍历专家轨迹的所有步骤
        for step_idx, (exp_goal, exp_action_tuple, exp_cost) in enumerate(expert_traj):
            if exp_goal not in env.unadded_dest_indices:
                continue

            exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

            # 在环境中执行
            next_high_level_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

            # 存储
            reward = low_level_agent.criticize(sub_task_completed=True, cost=cost, request_failed=False)
            goal_one_hot = to_categorical(exp_goal, num_classes=NB_GOALS)
            exp = ActorExperience(high_level_state, goal_one_hot, exp_action, reward, next_high_level_state,
                                  sub_task_done)
            low_level_agent.store(exp)

            # 训练
            if t % low_level_agent.trainFreq == 0:
                low_level_agent.update(t)

            metacontroller.collect(high_level_state, exp_goal)
            if t % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                metacontroller.train()

            high_level_state = next_high_level_state
            t += 1

            if t % 1000 == 0:
                print(f"预训练... {t}/{H.PRE_TRAIN_STEPS}")

            if request_done:
                break

        current_request, high_level_state = env.reset_request()

    print("--- 4. 阶段 2: 混合 IL/RL 训练 ---")
    low_level_agent.randomPlay = False
    stepCount = 0
    episodeCount = 0

    # ---
    # ✅ 步骤 2: 在主循环前初始化跟踪列表
    # ---
    tracking_data = {
        'episode': [],
        'reward': [],
        'acceptance_rate': [],
        'avg_cpu_util': [],
        'avg_mem_util': [],
        'avg_bw_util': []
    }
    total_requests_arrived = 0
    total_requests_served = 0

    while episodeCount < H.EPISODE_LIMIT and stepCount < H.STEPS_LIMIT and env.t < env.T:

        current_request, high_level_state = env.reset_request()
        if current_request is None:
            break

        total_requests_arrived += 1  # 跟踪到达的请求
        request_done = False
        episode_reward = 0
        episode_steps = 0

        while not request_done:
            # --- A. 高层决策 (元控制器) ---
            high_level_state_v = np.reshape(high_level_state, (1, -1))
            goal_probs = metacontroller.predict(high_level_state_v)
            goal = metacontroller.sample(goal_probs)

            true_goal = env.get_expert_high_level_goal(high_level_state_v)
            metacontroller.collect(high_level_state, true_goal)

            if goal != true_goal:
                print(f"⚠️ 高层错误 (代理选择 {goal}, 专家选择 {true_goal}), 丢弃轨迹")
                request_done = True
                continue

            if goal not in env.unadded_dest_indices:
                goal = true_goal
                if goal >= NB_GOALS or goal not in env.unadded_dest_indices:
                    request_done = True
                    continue

            goal_one_hot = np.reshape(to_categorical(goal, num_classes=NB_GOALS), (1, -1))

            # --- B. 低层执行 (代理) ---
            sub_task_done = False
            low_level_state = high_level_state

            while not sub_task_done:
                low_level_state_v = np.reshape(low_level_state, (1, -1))

                valid_actions = env.get_valid_low_level_actions()
                if not valid_actions:
                    action = 0
                else:
                    action = low_level_agent.selectMove(low_level_state_v, goal_one_hot, valid_actions)

                # ✅ 修复: 调用调试工具
                validate_action_space(env, action)

                # B2. 环境执行
                next_low_level_state, cost, sub_task_done, request_done = env.step_low_level(goal, action)

                # B3. 计算内部奖励 (RL)
                reward = low_level_agent.criticize(sub_task_done, cost, request_failed=(not sub_task_done))
                episode_reward += reward

                # B4. 存储经验
                exp = ActorExperience(high_level_state, goal_one_hot.flatten(), action, reward, next_low_level_state,
                                      sub_task_done)
                low_level_agent.store(exp)

                # B5. 训练
                if stepCount % low_level_agent.trainFreq == 0:
                    loss, avgQ, avgTD = low_level_agent.update(stepCount)
                    if stepCount % 1000 == 0 and (avgQ != 0 or loss != 0):
                        print(f"Step {stepCount} | Q: {avgQ:.3f}, TD: {avgTD:.3f}, Loss: {loss:.3f}")

                if stepCount % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                    metacontroller.train()

                low_level_agent.annealControllerEpsilon(stepCount)

                low_level_state = next_low_level_state
                stepCount += 1
                episode_steps += 1

                if request_done:
                    break

            high_level_state = low_level_state

        # ---
        # ✅ 步骤 3: 在回合结束时 (print 之前) 收集数据
        # ---

        # 1. 检查请求是否成功 (即所有目的地都已连接)
        # (注意: 如果高层出错, request_done=True 但 unadded_dest_indices 可能还有)
        # 我们只统计真正 *完成* 的请求
        if not env.unadded_dest_indices:
            total_requests_served += 1

        # 2. 计算当前统计数据
        # 业务请求接受率
        current_acceptance_rate = (total_requests_served / total_requests_arrived) * 100.0
        # 节点资源消耗 (利用率)
        avg_cpu_util = (1.0 - np.mean(env.C) / env.C_cap) * 100.0
        avg_mem_util = (1.0 - np.mean(env.M) / env.M_cap) * 100.0
        # 带宽消耗 (利用率)
        avg_bw_util = (1.0 - np.mean(env.B) / env.B_cap) * 100.0

        # 3. 存储所有数据
        tracking_data['episode'].append(episodeCount)
        tracking_data['reward'].append(episode_reward)  # 奖励趋势
        tracking_data['acceptance_rate'].append(current_acceptance_rate)
        tracking_data['avg_cpu_util'].append(avg_cpu_util)
        tracking_data['avg_mem_util'].append(avg_mem_util)
        tracking_data['avg_bw_util'].append(avg_bw_util)

        print(f"--- 回合 {episodeCount} (T={env.t}) ---")
        print(f"总步数: {stepCount}, Epsilon: {low_level_agent.controllerEpsilon:.4f}")
        print(f"回合奖励: {episode_reward:.3f}, 回合步数: {episode_steps}")
        print(f"当前接受率: {current_acceptance_rate:.2f}% ({total_requests_served}/{total_requests_arrived})")  # 实时打印
        print(f"当前资源利用率 CPU: {avg_cpu_util:.2f}%, MEM: {avg_mem_util:.2f}%, BW: {avg_bw_util:.2f}%")  # 实时打印

        # ✅ 修复: 调用调试工具
        # visualize_state(env) # 我们将其注释掉，因为它在每个回合都打印，信息过多

        episodeCount += 1

        if episodeCount % 50 == 0:
            print("保存模型...")
            model_path = H.OUTPUT_DIR / f"sfc_hirl_model_ep{episodeCount}"
            hdqn_net.saveWeight(str(model_path))

    print("--- 5. 训练完成 ---")
    hdqn_net.saveWeight(str(H.OUTPUT_DIR / "sfc_hirl_model_final"))
    print("最终模型已保存。")

    # ---
    # ✅ 步骤 4: 在 main() 函数末尾添加绘图和保存
    # ---
    print("--- 6. 生成分析图表 ---")

    # 1. 保存原始数据到 CSV
    df = pd.DataFrame(tracking_data)
    df_path = H.OUTPUT_DIR / "training_metrics.csv"
    try:
        df.to_csv(df_path, index=False)
        print(f"训练指标已保存到 {df_path}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")

    # 2. 创建平滑函数 (用于更清晰的图表)
    def smooth_rewards(values, window_size=100):
        if len(values) == 0:
            return []
        if window_size > len(values):
            window_size = len(values)
        if window_size == 0:
            return values
        return pd.Series(values).rolling(window_size, min_periods=1).mean()

    # 3. 绘制图表
    try:
        # 绘制奖励趋势
        plt.figure(figsize=(12, 8))
        plt.plot(df['episode'], df['reward'], label='原始每回合奖励', alpha=0.3)
        plt.plot(df['episode'], smooth_rewards(df['reward'], 100), label='平滑奖励 (窗口=100)', color='red')
        plt.title('奖励趋势 (Reward Trend)')
        plt.xlabel('回合 (Episode)')
        plt.ylabel('奖励 (Reward)')
        plt.legend()
        plt.grid(True)
        reward_path = H.OUTPUT_DIR / "reward_trend.png"
        plt.savefig(reward_path)
        print(f"奖励趋势图已保存到 {reward_path}")
        plt.close()

        # 绘制业务请求接受率
        plt.figure(figsize=(12, 8))
        plt.plot(df['episode'], df['acceptance_rate'])
        plt.title('请求接受率 (Acceptance Rate)')
        plt.xlabel('回合 (Episode)')
        plt.ylabel('接受率 (%)')
        plt.ylim(0, 105)  # 锁定 Y 轴在 0-100%
        plt.grid(True)
        acceptance_path = H.OUTPUT_DIR / "acceptance_rate.png"
        plt.savefig(acceptance_path)
        print(f"接受率图已保存到 {acceptance_path}")
        plt.close()

        # 绘制资源消耗
        plt.figure(figsize=(12, 8))
        plt.plot(df['episode'], smooth_rewards(df['avg_cpu_util'], 100), label='平均CPU利用率')
        plt.plot(df['episode'], smooth_rewards(df['avg_mem_util'], 100), label='平均内存利用率')
        plt.plot(df['episode'], smooth_rewards(df['avg_bw_util'], 100), label='平均带宽利用率')
        plt.title('资源利用率 (平滑窗口=100)')
        plt.xlabel('回合 (Episode)')
        plt.ylabel('利用率 (%)')
        plt.legend()
        plt.grid(True)
        resource_path = H.OUTPUT_DIR / "resource_utilization.png"
        plt.savefig(resource_path)
        print(f"资源利用率图已保存到 {resource_path}")
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")
        print("请确保已安装 matplotlib 和 pandas: pip install matplotlib pandas")

    print("--- 分析完成 ---")


# ============================================
# ✅ 修复: 添加调试和可视化工具
# ============================================

def visualize_state(env):
    """可视化当前环境状态（调试用）"""
    print(f"\n=== 当前状态 (T={env.t}) ===")
    if env.current_request:
        print(f"请求 ID: {env.current_request['id']}")
        print(f"源: {env.current_request['source']}, 目的: {env.current_request['dest']}")
        print(f"未完成目的: {env.unadded_dest_indices}")
        print(f"树上节点数: {len(env.nodes_on_tree)}")
        print(f"树上路径数: {len(env.current_tree['paths_map']) if env.current_tree else 0}")
    print(f"平均CPU利用率: {1 - np.mean(env.C / env.C_cap):.2%}")
    print(f"平均带宽利用率: {1 - np.mean(env.B / env.B_cap):.2%}")
    print(f"已服务请求数: {len(env.served_requests)}")
    print("=" * 40)


def validate_action_space(env, action):
    """验证动作是否有效（调试用）"""
    valid_actions = env.get_valid_low_level_actions()
    if action not in valid_actions:
        print(f"⚠️ 警告: 动作 {action} 不在有效动作集 {valid_actions} 中")
        return False
    return True


if __name__ == "__main__":
    main()