#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : DRL.py
# @Date    : 2025-11-06
# @Author  : gxr
# @Desc    : 深度强化学习高层调度器 (HRL)
#            - SFCEnv: Gym 环境封装
#            - train_agent: DRL 训练脚本
# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from mph1 import DynamicSimulator, parse_mat_request, topology_link
from pathlib import Path


class SFCEnv(gym.Env):
    """
    高层 DRL 环境：学习部署顺序
    这是一个 gym.Env 包装器，用于包装 DynamicSimulator
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, simulator: DynamicSimulator):
        super(SFCEnv, self).__init__()
        self.sim = simulator
        self.t = 0  # 当前时间步
        self.pending_requests = []  # 当前时间步待处理的请求
        self.done_at_t = False  # 标记当前时间步是否已无请求
        self.max_pending_requests = 10  # 假设一个时间步最多10个请求

        # 动作空间：选择待办列表中的一个请求 (的索引)
        # 0, 1, ..., 9
        self.action_space = spaces.Discrete(self.max_pending_requests)

        # 观测空间
        n_nodes = self.sim.solver.node_num
        n_links = self.sim.solver.link_num
        n_vnf_types = self.sim.solver.type_num

        # [CPU_nodes, MEM_nodes, BW_links, HVT_nodes*types, Pending_req_slots]
        # (N,) + (N,) + (L,) + (N*K,) + (10,)
        obs_shape = (n_nodes + n_nodes + n_links + (n_nodes * n_vnf_types) + self.max_pending_requests,)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,  # 我们将对所有状态进行归一化
            shape=obs_shape, dtype=np.float32
        )

    def _get_state(self):
        """将当前的网络状态和待办列表编码为 DRL 智能体可以理解的向量"""

        # 1. 获取当前 t 时刻的资源状态
        state = self.sim.get_state(self.t)

        # 2. 归一化资源
        cpu_norm = state['cpu'] / self.sim.solver.cpu_capacity
        mem_norm = state['mem'] / self.sim.solver.memory_capacity
        bw_norm = state['bw'] / self.sim.solver.bandwidth_capacity

        # 3. 归一化 hvt_all (使用引用计数，假设最大共享为 10)
        hvt_norm = state['hvt'].flatten() / 10.0

        # 4. 编码待办列表
        pending_vec = np.zeros(self.max_pending_requests, dtype=np.float32)
        # 用 1 标记有请求的位置
        pending_vec[:len(self.pending_requests)] = 1.0

        # 5. 组合所有特征
        combined_state = np.concatenate([
            cpu_norm,
            mem_norm,
            bw_norm,
            hvt_norm,
            pending_vec
        ]).astype(np.float32)

        return combined_state

    def reset(self):
        """重置仿真器以开始一个新 Episode"""
        print("--- 仿真 Episode 重置 ---")
        self.t = 0

        # 重置所有资源
        self.sim.B[:] = self.sim.solver.bandwidth_capacity
        self.sim.C[:] = self.sim.solver.cpu_capacity
        self.sim.M[:] = self.sim.solver.memory_capacity
        self.sim.hvt_all[:] = 0

        # 重置状态
        self.sim.served_requests = []
        self.sim.served_trees = []
        self.sim.block = 0
        self.sim.arrived = 0

        # 加载第一个时间步的请求
        self.pending_requests = list(self.sim.events[self.t]['arrive'])
        self.sim.arrived += len(self.pending_requests)
        self.done_at_t = (len(self.pending_requests) == 0)

        return self._get_state()

    def step(self, action: int):
        """
        DRL 智能体执行一个动作
        action: 待办列表中请求的 *索引* (0 to 9)
        """

        # --- 1. 检查是否需要推进时间 ---
        # 如果上一步已经处理完所有请求
        if self.done_at_t:
            self.t += 1  # 推进时间

            # 检查整个 Episode 是否结束
            if self.t >= self.sim.T:
                final_blocking_rate = self.sim.block / self.sim.arrived if self.sim.arrived > 0 else 0
                print(f"--- Episode 结束 --- 阻塞率: {final_blocking_rate:.4f}")
                return self._get_state(), 0.0, True, {"episode_blocked": self.sim.block}

            # --- 2. 处理新时间步的离开事件 ---
            leave_ids = self.sim.events[self.t]['leave']
            if leave_ids is not None and leave_ids.size > 0:
                # 注意：必须传入 self.t, 因为我们要修改 B[t], C[t], M[t]
                self.sim._handle_leave_event(self.t, leave_ids)

            # --- 3. 加载新时间步的到达事件 ---
            self.pending_requests = list(self.sim.events[self.t]['arrive'])
            self.sim.arrived += len(self.pending_requests)

            # 如果新时间步没有请求，则自动标记为 "done_at_t"
            if len(self.pending_requests) == 0:
                self.done_at_t = True
                # 返回 0 奖励，并告诉智能体这一步 "done" (以便它再次调用 step)
                return self._get_state(), 0.0, True, {}
            else:
                self.done_at_t = False

        # --- 4. 处理智能体的动作 (部署一个请求) ---

        # 检查非法动作 (例如待办列表只有3个，智能体选了 5)
        if action >= len(self.pending_requests):
            reward = -1000.0  # 对非法动作的巨大惩罚
            done = False  # 这一步没完成，但智能体犯错了
            return self._get_state(), reward, done, {"error": "Invalid action"}

        # 弹出被选中的请求
        req_id = self.pending_requests.pop(action)

        if req_id not in self.sim.req_map:
            reward = -1000.0  # 惩罚
            return self._get_state(), reward, False, {"error": "Request ID not found"}

        req = self.sim.req_map[req_id]

        # --- 5. 调用低层执行器 (MSFCE_Solver) ---
        plan = self.sim.solver.solve_request(req, self.sim.get_state(self.t))

        # --- 6. 计算奖励 ---
        if plan is None:
            # 阻塞
            reward = -500.0  #
            self.sim.block += 1
        else:
            # 成功：应用部署并计算奖励
            self.sim._apply_deployment(self.t, req, plan)
            self.sim.served_requests.append(req)
            self.sim.served_trees.append(plan)

            #
            cost = plan['cost_bw'] + plan['cost_cpu'] + plan['cost_mem']
            reward = -cost * 0.01  # 将成本 (例如 5000) 缩放到合理的奖励范围 (例如 -50)

        # --- 7. 检查当前时间步是否完成 ---
        self.done_at_t = (len(self.pending_requests) == 0)

        # 'done' 标志在这里有两种含义:
        # 1. done=True: 当前 *时间步 t* 的所有请求都处理完了。
        # 2. done=True 且 t >= T: 整个 *Episode* 结束了。
        # DRL 智能体 (如 PPO) 会将 done=True 视为一个 mini-episode 的结束。
        return self._get_state(), reward, self.done_at_t, {}

    def render(self, mode='human'):
        print(
            f"  [T={self.t}] 待办: {len(self.pending_requests)}, 已服务: {len(self.sim.served_requests)}, 阻塞: {self.sim.block}")

    def close(self):
        print("环境关闭")


# ==============================================================
# ✅ DRL 训练脚本
# ==============================================================
def train_agent():
    """
    主训练函数
    """

    # --- 1. 配置 (来自您的 train_agent.py) ---
    INPUT_DIR = Path('E:/pycharmworkspace/SFC-master/mph/mat')
    OUTPUT_DIR = Path('E:/pycharmworkspace/SFC-master/mph/out')

    # (来自您的 MPH1.py 脚本)
    DC_NODES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 27, 28]

    # (来自您的 MPH1.py 脚本,)
    CAPACITIES = {
        'cpu': 2000.0,
        'memory': 1100.0,
        'bandwidth': 500.0
    }

    # (来自您的 mph.py 脚本,)
    TOPOLOGY_MATRIX = np.array([
        [np.inf, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [1, np.inf, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [1, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, 1, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, 1, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf,
         1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, 1,
         np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf,
         np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf,
         np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1,
         np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf,
         np.inf, np.inf, 1, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, 1, np.inf, 1, np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, 1, 1, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1,
         np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, np.inf, np.inf, 1, np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, np.inf, 1, np.inf, 1],
        [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
         np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1, 1, np.inf, np.inf, 1, np.inf]
    ])

    # --- 2. 初始化环境 ---
    try:
        sim = DynamicSimulator(
            input_dir=INPUT_DIR,
            output_dir=OUTPUT_DIR,
            topo=TOPOLOGY_MATRIX,
            dc_nodes=DC_NODES,
            capacities=CAPACITIES
        )

        # 将仿真器包装在 Gym 环境中
        env = SFCEnv(sim)

        # 包装在 DummyVecEnv 中以供 SB3 使用
        vec_env = DummyVecEnv([lambda: env])

        # --- 3. 初始化 PPO 智能体 ---
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log="./ppo_sfc_logs/")

        # --- 4. 训练 ---
        print("\n--- 开始 DRL 训练 ---")
        model.learn(total_timesteps=50000)  #
        print("--- 训练完成 ---")

        # --- 5. 保存模型 ---
        model.save("ppo_sfc_scheduler")
        print(f"模型已保存到 ppo_sfc_scheduler.zip")

        # --- 6. 评估 ---
        print("\n--- 开始评估已训练的模型 ---")
        obs = vec_env.reset()
        episode_rewards = []
        current_episode_reward = 0

        # 运行一个完整的 Episode (T=400 步)
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)

            current_episode_reward += reward

            if done:
                # 检查是 episode 结束还是只是 t 结束
                if env.envs[0].done_at_t:
                    if env.envs[0].t >= env.envs[0].sim.T:
                        # 真正的 Episode 结束
                        print("评估 Episode 结束。")
                        episode_rewards.append(current_episode_reward)
                        break
                    else:
                        # 只是一个时间步结束，继续
                        pass

        print(f"评估完成。总奖励: {episode_rewards[0]}")
        print(f"最终阻塞率: {env.envs[0].sim.block / env.envs[0].sim.arrived:.4f}")

    except FileNotFoundError as e:
        print(f"❌ 关键文件未找到: {e}")
        print(f"请确保 'US_Backbone_path.mat', 'sorted_requests.mat', 和 'event_list.mat' 存在于: {INPUT_DIR}")
    except Exception as e:
        print(f"❌ 仿真时发生未知错误: {e}")


if __name__ == '__main__':
    train_agent()
