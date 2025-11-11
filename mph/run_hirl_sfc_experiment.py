# (文件: run_hirl_sfc_experiment.py)

import numpy as np
from msfce_simulator import DynamicSimulator
from expert_msfce import ExpertMSFCE
from hirl_sfc_models import create_meta_controller, Hdqn_SFC
from hirl_sfc_agent import Agent_SFC
from collections import namedtuple

ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])


def main():
    # 1. 初始化
    env = DynamicSimulator(...)
    expert = ExpertMSFCE(...)

    # 高层
    metacontroller = create_meta_controller()

    # 低层 (每个子任务一个代理)
    agent_list = []
    for i in range(NB_SUBTASKS):
        net = Hdqn_SFC()
        agent = Agent_SFC(net, ...)
        agent_list.append(agent)

    # 2. 训练阶段 1: 模仿学习 (Behavioral Cloning)
    # 我们首先用专家的轨迹"预热"低层代理
    print("--- 阶段 1: 模仿学习 (预热) ---")
    for t in range(PRE_TRAIN_STEPS):
        req = env.get_next_request()  # 假设
        network_state = env.get_state(t)

        # 运行专家并获取轨迹
        trajectory = expert.solve_and_log(req, network_state)

        for (state, goal, action, next_state, done) in trajectory:
            # 将专家数据填充到对应的低层代理中
            agent_list[goal].store_imitation_data([(state, goal, action, next_state, done)])

            # (也可以在这里训练高层元控制器)
            # metacontroller.train_on_batch(state, goal_one_hot)

        # 预训练低层
        for agent in agent_list:
            if len(agent.memory) > BATCH_SIZE:
                agent.update(t)

    print("--- 阶段 2: 混合 模仿-强化 训练 ---")

    # 3. 训练阶段 2: 混合训练 (主循环)
    for episode in range(EPISODE_LIMIT):
        # ... (重置环境)

        while not env.isTerminal():

            # A. 高层决策 (元控制器选择子任务)
            flat_state = env.get_flat_state()
            goal = metacontroller.predict(flat_state).argmax()  # 选择子任务
            true_goal = expert.get_high_level_choice(flat_state)  # DAgger: 向专家查询

            # (DAgger: 收集高层数据)
            metacontroller.collect(flat_state, true_goal)

            # B. 低层执行
            while not env.subTaskCompleted(goal):
                low_level_state = env.get_flat_state()
                goal_one_hot = ...

                # B1. 低层选择动作
                action = agent_list[goal].selectMove(low_level_state, goal_one_hot)

                # B2. 环境执行动作
                # 这里需要修改 env, 使其能接收低层动作 (i_idx, k)
                next_state, sub_task_completed, cost = env.step_low_level(action)

                # B3. 计算内部奖励 (RL)
                reward = agent_list[goal].criticize(sub_task_completed, cost)

                # B4. 存储经验 (用于 RL)
                exp = ActorExperience(low_level_state, goal, action, reward, next_state, sub_task_completed)
                agent_list[goal].store(exp)

                # B5. 训练低层 (RL)
                agent_list[goal].update(stepCount)

            # C. 训练高层 (IL)
            if metacontroller.check_training_clock():
                metacontroller.train()  # DAgger 训练