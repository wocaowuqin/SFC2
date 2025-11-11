#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : train_hirl_sfc.py

import numpy as np
from collections import namedtuple
import os

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

    # ----------------------------------------------------
    # ✅ 修复 #7 和 #4: 替换为您的新预训练循环
    # ----------------------------------------------------
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

        # ✅ 遍历专家轨迹的所有步骤
        for step_idx, (exp_goal, exp_action_tuple, exp_cost) in enumerate(expert_traj):
            if exp_goal not in env.unadded_dest_indices:
                continue  # 跳过已完成的目标

            exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

            # 在环境中执行
            next_high_level_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

            # 存储
            reward = low_level_agent.criticize(sub_task_completed=True, cost=cost, request_failed=False)
            # ✅ [修复 #4] 存储 one-hot 编码的 goal
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
            t += 1  # 增加步数

            if t % 1000 == 0:
                print(f"预训练... {t}/{H.PRE_TRAIN_STEPS}")

            if request_done:
                break  # 退出内部轨迹循环

        # 请求完成,重置
        current_request, high_level_state = env.reset_request()

    print("--- 4. 阶段 2: 混合 IL/RL 训练 ---")
    low_level_agent.randomPlay = False  # 结束随机播放
    stepCount = 0  # 重置步数
    episodeCount = 0

    while episodeCount < H.EPISODE_LIMIT and stepCount < H.STEPS_LIMIT and env.t < env.T:

        current_request, high_level_state = env.reset_request()
        if current_request is None:
            break  # 仿真结束

        request_done = False
        episode_reward = 0
        episode_steps = 0

        while not request_done:
            # --- A. 高层决策 (元控制器) ---
            high_level_state_v = np.reshape(high_level_state, (1, -1))
            goal_probs = metacontroller.predict(high_level_state_v)
            goal = metacontroller.sample(goal_probs)  # 采样一个子任务

            # (DAgger) 查询专家
            true_goal = env.get_expert_high_level_goal(high_level_state_v)
            metacontroller.collect(high_level_state, true_goal)

            # ----------------------------------------------------
            # ✅ 修复 #9: 添加分层指导 (HIRL 核心)
            # ----------------------------------------------------
            if goal != true_goal:
                print(f"⚠️ 高层错误 (代理选择 {goal}, 专家选择 {true_goal}), 丢弃轨迹")
                request_done = True  # 终止当前请求
                continue  # 不执行低层

            # (如果子任务无效，则跳过)
            if goal not in env.unadded_dest_indices:
                goal = true_goal  # 强制使用专家
                if goal >= NB_GOALS or goal not in env.unadded_dest_indices:
                    request_done = True
                    continue

            goal_one_hot = np.reshape(to_categorical(goal, num_classes=NB_GOALS), (1, -1))

            # --- B. 低层执行 (代理) ---
            sub_task_done = False
            low_level_state = high_level_state

            while not sub_task_done:
                low_level_state_v = np.reshape(low_level_state, (1, -1))

                # ----------------------------------------------------
                # ✅ 修复 #6: 获取并使用动作掩码
                # ----------------------------------------------------
                valid_actions = env.get_valid_low_level_actions()
                if not valid_actions:
                    # 如果掩码为空（不应发生），则强制失败
                    action = 0
                else:
                    action = low_level_agent.selectMove(low_level_state_v, goal_one_hot, valid_actions)

                # B2. 环境执行
                next_low_level_state, cost, sub_task_done, request_done = env.step_low_level(goal, action)

                # B3. 计算内部奖励 (RL)
                reward = low_level_agent.criticize(sub_task_done, cost, request_failed=(not sub_task_done))
                episode_reward += reward

                # B4. 存储经验
                # ----------------------------------------------------
                # ✅ 修复 #4: 存储 one-hot 编码的 goal
                # ----------------------------------------------------
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
                    break  # 整个请求完成了

            high_level_state = low_level_state

        print(f"--- 回合 {episodeCount} (T={env.t}) ---")
        print(f"总步数: {stepCount}, Epsilon: {low_level_agent.controllerEpsilon:.4f}")
        print(f"回合奖励: {episode_reward:.3f}, 回合步数: {episode_steps}")
        episodeCount += 1

        if episodeCount % 50 == 0:
            print("保存模型...")
            model_path = H.OUTPUT_DIR / f"sfc_hirl_model_ep{episodeCount}"
            hdqn_net.saveWeight(str(model_path))  # 确保路径是字符串

    print("--- 5. 训练完成 ---")
    hdqn_net.saveWeight(str(H.OUTPUT_DIR / "sfc_hirl_model_final"))
    print("最终模型已保存。")


if __name__ == "__main__":
    main()