#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_agent.py
"""
DAgger + Low Epsilon 修改版 Agent (TensorFlow/Keras)
改进说明：
- 集成 DAggerSchedule：随着训练进行，逐渐从模仿专家过渡到自主决策。
- 低 Epsilon 探索：初始 epsilon 降至 0.2，避免盲目随机，依赖专家进行探索。
- 动作掩码与安全预测保留。
- 增加了 Expert 交互接口。
"""

import numpy as np
import random
from collections import namedtuple
from typing import List, Tuple
import logging
import gc

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers, Model
from tensorflow.keras.layers import Input

# project utils (assumed present)
from hirl_utils import PrioritizedReplayBuffer, LinearSchedule

# model
from hirl_sfc_models import Hdqn_SFC, huber_loss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Experience = namedtuple('Experience', ['state', 'goal', 'action', 'reward', 'next_state', 'done'])


# ====================================================================
# 新增：DAgger 调度器
# ====================================================================
class DAggerSchedule:
    """
    DAgger 的 Beta 调度器
    Beta = 采纳专家建议的概率
    """

    def __init__(self, initial_beta=0.9, final_beta=0.1, decay_steps=1000000):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.decay_steps = decay_steps

    def value(self, step):
        if step >= self.decay_steps:
            return self.final_beta
        # 线性衰减
        fraction = step / float(self.decay_steps)
        beta = self.initial_beta - fraction * (self.initial_beta - self.final_beta)
        return beta


class Agent_SFC:
    def __init__(self, net: Hdqn_SFC, n_actions: int, mem_cap: int, exploration_steps: int,
                 train_freq: int, hard_update: int, n_samples: int, gamma: float,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4):
        """
        Args:
            net: Hdqn_SFC model wrapper
            n_actions: total low-level discrete actions
            mem_cap: PER capacity
            exploration_steps: timesteps schedule for epsilon decay
            train_freq: how often to call update
            hard_update: steps interval to sync target net
            n_samples: batch size
            gamma: discount factor
            per_alpha: PER exponent
            per_beta_start: initial beta for importance sampling
        """
        self.net = net
        self.n_actions = n_actions
        self.memory = PrioritizedReplayBuffer(mem_cap, alpha=per_alpha)

        # 🔥 修改：降低初始 Epsilon (0.3/1.0 -> 0.2)
        # 因为有了 Expert 指导，不需要一开始进行高概率的随机探索
        self.exploration = LinearSchedule(schedule_timesteps=exploration_steps, initial_p=0.2, final_p=0.02)

        # 🔥 新增：DAgger 调度器 (默认 100万步衰减)
        self.dagger_schedule = DAggerSchedule(initial_beta=0.9, final_beta=0.1, decay_steps=1000000)

        # 🔥 新增：Expert 接口与统计
        self.expert = None
        self.action_stats = {
            'expert': 0,
            'agent_exploit': 0,
            'agent_explore': 0,
            'total': 0
        }

        self.trainFreq = train_freq
        self.hard_update = hard_update
        self.nSamples = n_samples
        self.gamma = gamma
        self.update_count = 0
        self.beta_schedule = LinearSchedule(exploration_steps, initial_p=per_beta_start, final_p=1.0)
        self.controllerEpsilon = 1.0
        self.randomPlay = True
        self.compiled = False

        # optimizer
        self._optimizer = optimizers.RMSprop(learning_rate=self.net.lr, rho=0.95, epsilon=1e-08, clipnorm=10.0)

        self.trainable_model = None

    # ---------------------------
    # 新增：Expert 设置与交互
    # ---------------------------
    def set_expert(self, expert_instance):
        """设置外部专家实例 (通常在 env 初始化后调用: agent.set_expert(env.expert))"""
        self.expert = expert_instance
        logger.info(f"[Agent] Expert set: {expert_instance}")

    def _node_to_action(self, next_node):
        """
        将节点ID转换为动作ID。
        假设动作空间 0~27 对应节点 1~28。如果不一致请在此修改。
        """
        # 假设动作索引 = 节点ID - 1
        if 1 <= next_node <= self.n_actions:
            return next_node - 1
        return None

    def _get_expert_action(self, current_node, goal_node, valid_actions_idx):
        """尝试获取专家建议动作"""
        if self.expert is None or current_node is None or goal_node is None:
            return None

        try:
            # 适配常见的 Expert 接口
            # 1. 优先尝试 get_next_hop
            if hasattr(self.expert, 'get_next_hop'):
                next_node = self.expert.get_next_hop(current_node, goal_node)
                action = self._node_to_action(next_node)
                if action is not None and action in valid_actions_idx:
                    return action

            # 2. 尝试 get_shortest_path
            elif hasattr(self.expert, 'get_shortest_path'):
                path = self.expert.get_shortest_path(current_node, goal_node)
                if path and len(path) > 1:
                    next_node = path[1]  # 路径的下一个跳
                    action = self._node_to_action(next_node)
                    if action is not None and action in valid_actions_idx:
                        return action

            # 3. 尝试 _get_path_info (PathDB)
            elif hasattr(self.expert, '_get_path_info'):
                nodes, _, _ = self.expert._get_path_info(current_node, goal_node, k=1)
                if nodes and len(nodes) > 1:
                    next_node = nodes[1]
                    action = self._node_to_action(next_node)
                    if action is not None and action in valid_actions_idx:
                        return action
        except Exception:
            # 专家获取失败，静默回退到 Agent
            pass
        return None

    # ---------------------------
    # Utility: ensure batch dims & dtype
    # ---------------------------
    @staticmethod
    def _to_batch(arr):
        a = np.atleast_2d(np.asarray(arr, dtype=np.float32))
        return a

    # ---------------------------
    # Safe predict wrappers
    # ---------------------------
    def _safe_predict_q(self, model: Model, state_batch: np.ndarray, goal_batch: np.ndarray) -> np.ndarray:
        state_np = self._to_batch(state_batch)
        goal_np = self._to_batch(goal_batch)
        return model.predict([state_np, goal_np], verbose=0)

    # ---------------------------
    # 🔥 修改：Action selection (DAgger + Mask)
    # ---------------------------
    def selectMove(self, state: np.ndarray, goal_one_hot: np.ndarray, valid_actions: list,
                   current_node=None, goal_node=None):
        """
        选择动作：DAgger 增强版
        Args:
            state, goal_one_hot: 神经网络输入
            valid_actions: 合法动作列表 (indices)
            current_node: 当前拓扑节点 ID (用于询问专家)
            goal_node: 目标拓扑节点 ID (用于询问专家)
        """
        # 1. 基本检查
        if valid_actions is None or len(valid_actions) == 0:
            logger.warning("[selectMove] No valid_actions provided, using action 0")
            return 0

        # 2. 获取 Agent Q 值
        state_b = self._to_batch(state)
        goal_b = self._to_batch(goal_one_hot)
        q_vals = self._safe_predict_q(self.net.controllerNet, state_b, goal_b)[0]

        # 过滤有效动作索引
        valid_idx = [a for a in valid_actions if 0 <= a < len(q_vals)]
        if not valid_idx:
            return 0

        # 3. 🔥 DAgger 逻辑：决定是否听从专家
        beta = self.dagger_schedule.value(self.update_count)

        # 尝试获取专家动作
        # 只有当随机数 < beta 时，我们才真正去调用专家 (节省计算资源)
        # 或者为了收集数据，也可以每次都调，但只按概率采纳
        if np.random.random() < beta:
            expert_act = self._get_expert_action(current_node, goal_node, valid_idx)
            if expert_act is not None:
                self.action_stats['expert'] += 1
                self.action_stats['total'] += 1
                return expert_act

        # 4. Agent 决策 (回退逻辑)
        # 准备 Masked Q
        mask = np.full_like(q_vals, -1e9, dtype=np.float32)
        mask[valid_idx] = 0.0
        masked_q = q_vals + mask
        best_agent_action = int(np.argmax(masked_q))

        # Epsilon-Greedy (使用较低的 epsilon)
        eps = float(self.exploration.value(self.update_count))
        self.controllerEpsilon = eps

        if random.random() > eps:
            # Exploit
            action = best_agent_action
            if action not in valid_idx: action = valid_idx[0]
            self.action_stats['agent_exploit'] += 1
        else:
            # Explore
            action = int(random.choice(valid_idx))
            self.action_stats['agent_explore'] += 1

        self.action_stats['total'] += 1
        return action

    # ---------------------------
    # 统计打印方法
    # ---------------------------
    def print_action_stats(self):
        """在 Episode 结束时调用此方法查看决策分布"""
        total = self.action_stats['total']
        if total == 0: return

        print(f"\n🎯 [Agent Stats Step {self.update_count}]")
        print(f"  Expert Advice: {self.action_stats['expert'] / total:6.1%} ({self.action_stats['expert']})")
        print(
            f"  Agent Exploit: {self.action_stats['agent_exploit'] / total:6.1%} ({self.action_stats['agent_exploit']})")
        print(
            f"  Agent Explore: {self.action_stats['agent_explore'] / total:6.1%} ({self.action_stats['agent_explore']})")
        print(f"  Current Beta : {self.dagger_schedule.value(self.update_count):.4f}")
        print(f"  Current Eps  : {self.exploration.value(self.update_count):.4f}")

    def reset_action_stats(self):
        self.action_stats = {k: 0 for k in self.action_stats}

    # ---------------------------
    # Store experience
    # ---------------------------
    def store(self, experience: Experience):
        self.memory.add(experience.state.astype(np.float32),
                        experience.goal.astype(np.float32),
                        int(experience.action),
                        float(experience.reward),
                        experience.next_state.astype(np.float32),
                        bool(experience.done))

    def update(self, t=None):
        try:
            return self.train_from_memory()
        except Exception as e:
            logger.error(f"[Agent] update() failed: {e}")
            return None

    # ---------------------------
    # Build & Compile (Gradient Clipping)
    # ---------------------------
    def compile(self):
        if self.compiled:
            return

        inputs = self.net.controllerNet.input
        y_true = Input(name='y_true', shape=(self.n_actions,), dtype='float32')
        mask = Input(name='mask', shape=(self.n_actions,), dtype='float32')
        y_pred = self.net.controllerNet.output

        def masked_loss(args):
            y_t, y_p, m = args
            loss_elem = huber_loss(y_t, y_p, clip_value=1.0)
            loss_elem = loss_elem * m
            return K.mean(K.sum(loss_elem, axis=-1))

        loss_out = tf.keras.layers.Lambda(masked_loss, name='loss')([y_true, y_pred, mask])
        trainable_model = Model(inputs=inputs + [y_true, mask], outputs=loss_out)

        optimizer = optimizers.RMSprop(
            learning_rate=self.net.lr,
            rho=0.95,
            epsilon=1e-08,
            clipnorm=10.0  # 梯度裁剪防止爆炸
        )

        trainable_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
        self.trainable_model = trainable_model
        self.compiled = True
        logger.info("[Agent] Compiled with DAgger support")

    # ---------------------------
    # Train Loop (Double DQN + PER)
    # ---------------------------
    def train_from_memory(self):
        if not self.compiled:
            self.compile()
        if self.memory.size() < max(32, self.nSamples):
            return 0.0

        # sample
        beta = float(self.beta_schedule.value(self.update_count))
        samples = self.memory.sample(self.nSamples, beta=beta)

        states = np.asarray(samples['states'], dtype=np.float32)
        goals = np.asarray(samples['goals'], dtype=np.float32)
        actions = np.asarray(samples['actions'], dtype=np.int32)
        rewards = np.asarray(samples['rewards'], dtype=np.float32)
        next_states = np.asarray(samples['next_states'], dtype=np.float32)
        dones = np.asarray(samples['dones'], dtype=np.float32)
        is_weights = np.asarray(samples['weights'], dtype=np.float32)
        idxes = samples['idxes']

        # predict
        q_values = self._safe_predict_q(self.net.controllerNet, states, goals)
        q_next_online = self._safe_predict_q(self.net.controllerNet, next_states, goals)
        q_next_target = self._safe_predict_q(self.net.targetControllerNet, next_states, goals)

        # Double DQN target
        next_actions = np.argmax(q_next_online, axis=1)
        q_next_selected = q_next_target[np.arange(self.nSamples), next_actions]
        targets = rewards + (1.0 - dones) * self.gamma * q_next_selected

        # TD Error
        td_errors = np.abs(q_values[np.arange(self.nSamples), actions] - targets)

        # Prepare inputs
        y_true = q_values.copy()
        y_true[np.arange(self.nSamples), actions] = targets

        mask = np.zeros_like(y_true, dtype=np.float32)
        mask[np.arange(self.nSamples), actions] = 1.0
        weighted_mask = mask * is_weights.reshape(-1, 1)

        # Train
        loss = self.trainable_model.train_on_batch(
            [states, goals, y_true, weighted_mask],
            np.zeros((len(states),))
        )

        # Update PER
        self.memory.update_priorities(idxes, td_errors + 1e-6)

        # Update Step & Target Net
        self.update_count += 1
        if self.update_count % self.hard_update == 0:
            self.net.targetControllerNet.set_weights(self.net.controllerNet.get_weights())

        gc.collect()
        return float(loss)

    def save(self, prefix: str):
        self.net.controllerNet.save_weights(f"{prefix}_controller.weights.h5")
        self.net.targetControllerNet.save_weights(f"{prefix}_target_controller.weights.h5")
        logger.info("Agent weights saved to %s_*", prefix)

    def load(self, prefix: str):
        self.net.controllerNet.load_weights(f"{prefix}_controller.weights.h5")
        self.net.targetControllerNet.load_weights(f"{prefix}_target_controller.weights.h5")
        logger.info("Agent weights loaded from %s_*", prefix)

    def q_values(self, state: np.ndarray, goal_one_hot: np.ndarray) -> np.ndarray:
        state_b = self._to_batch(state)
        goal_b = self._to_batch(goal_one_hot)
        return self._safe_predict_q(self.net.controllerNet, state_b, goal_b)[0]