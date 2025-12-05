#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_agent.py
"""
改进说明（已实现于本文件）：
- 动作掩码：selectMove 使用掩码方式将非法动作设为 -inf，避免错误选择。
- Safe predict：所有模型 predict 都通过 np.atleast_2d 包装，保证 batch 维度。
- 减少重复模型调用：update 时尽量复用一次 forward，避免双重 predict 开销。
- PER 优先级计算采用 (|td| + eps)^alpha，并支持 beta 权重。
- Double DQN：采用 online-network 选择动作，target-network 计算 Q_target。
- 梯度裁剪：在 optimizer 中使用 clipnorm 保护梯度爆炸。
- 日志替换 print：使用 logging，默认级别 INFO，可调整为 DEBUG。
- 统一数据类型：输入强制 float32。
- 提供 save/load 接口，同步 target 网络权重。
- 简化/明确 API：predict/safe functions, step_update, train_from_memory
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
logger.setLevel(logging.INFO)  # change to DEBUG for verbose logs

Experience = namedtuple('Experience', ['state', 'goal', 'action', 'reward', 'next_state', 'done'])

class Agent_SFC:
    def __init__(self, net: Hdqn_SFC, n_actions: int, mem_cap: int, exploration_steps: int,
                 train_freq: int, hard_update: int, n_samples: int, gamma: float,
                 per_alpha: float = 0.6, per_beta_start: float = 0.4):
        """
        Args:
            net: Hdqn_SFC model wrapper (with controllerNet & targetControllerNet)
            n_actions: total low-level discrete actions
            mem_cap: PER capacity
            exploration_steps: timesteps schedule for epsilon decay
            train_freq: how often to call update (in environment steps)
            hard_update: steps interval to sync target net
            n_samples: batch size
            gamma: discount factor
            per_alpha: PER exponent
            per_beta_start: initial beta for importance sampling
        """
        self.net = net
        self.n_actions = n_actions
        self.memory = PrioritizedReplayBuffer(mem_cap, alpha=per_alpha)
        self.exploration = LinearSchedule(schedule_timesteps=exploration_steps, initial_p=1.0, final_p=0.02)
        self.trainFreq = train_freq
        self.hard_update = hard_update
        self.nSamples = n_samples
        self.gamma = gamma
        self.update_count = 0
        self.beta_schedule = LinearSchedule(exploration_steps, initial_p=per_beta_start, final_p=1.0)
        self.controllerEpsilon = 1.0
        self.randomPlay = True
        self.compiled = False

        # optimizer for trainable_model created in compile()
        self._optimizer = optimizers.RMSprop(learning_rate=self.net.lr, rho=0.95, epsilon=1e-08, clipnorm=10.0)

        # placeholders
        self.trainable_model = None

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
        """Predict Q-values with guaranteed batch dims and dtype."""
        state_np = self._to_batch(state_batch)
        goal_np = self._to_batch(goal_batch)
        # model expects [state, goal]
        return model.predict([state_np, goal_np], verbose=0)

    # ---------------------------
    # Action selection with mask
    # ---------------------------
    def selectMove(self, state: np.ndarray, goal_one_hot: np.ndarray, valid_actions: list):
        """
        修复版 selectMove

        修复点:
        1. 更健壮的 valid_actions 处理
        2. 避免返回无效动作
        3. 添加更多日志
        """
        state_b = self._to_batch(state)
        goal_b = self._to_batch(goal_one_hot)

        # exploration prob
        eps = float(self.exploration.value(self.update_count))
        self.controllerEpsilon = eps

        # 获取 Q 值
        q_vals = self._safe_predict_q(self.net.controllerNet, state_b, goal_b)[0]

        # 🔧 修复: 更健壮的 valid_actions 处理
        if valid_actions is None or len(valid_actions) == 0:
            logger.warning("[selectMove] No valid_actions provided, using action 0")
            return 0

        # 过滤有效动作（在Q值范围内）
        valid_idx = [a for a in valid_actions if 0 <= a < len(q_vals)]

        if len(valid_idx) == 0:
            # 🔧 修复: 如果没有有效动作，记录警告并返回0
            logger.warning(
                f"[selectMove] No valid actions in range [0, {len(q_vals)}). "
                f"Provided: {valid_actions[:10]}... Returning 0"
            )
            return 0

        # 创建掩码
        mask = np.full_like(q_vals, -1e9, dtype=np.float32)
        mask[valid_idx] = 0.0

        if random.random() > eps:
            # exploit: apply mask then argmax
            masked = q_vals + mask
            action = int(np.argmax(masked))

            # 🔧 额外检查: 确保选择的动作确实有效
            if action not in valid_idx:
                logger.warning(f"[selectMove] Argmax action {action} not in valid_idx, using first valid")
                action = valid_idx[0]

            logger.debug(f"[selectMove] Exploit: action={action}, eps={eps:.4f}")
            return action
        else:
            # explore uniformly among valid
            action = int(random.choice(valid_idx))
            logger.debug(f"[selectMove] Explore: action={action}, eps={eps:.4f}")
            return action

    # ---------------------------
    # Store experience into PER
    # ---------------------------
    def store(self, experience: Experience):
        """
        experience: namedtuple (state, goal, action, reward, next_state, done)
        goal assumed one-hot
        """
        self.memory.add(experience.state.astype(np.float32),
                        experience.goal.astype(np.float32),
                        int(experience.action),
                        float(experience.reward),
                        experience.next_state.astype(np.float32),
                        bool(experience.done))

    def update(self, t=None):
        """
        为了兼容训练脚本的 update() 调用，
        这里直接调用 train_from_memory()。
        """
        try:
            return self.train_from_memory()
        except Exception as e:
            logger.error(f"[Agent] update() 调用失败: {e}")
            return None

    # ---------------------------
    # Build trainable_model (custom loss) and compile
    # ---------------------------
    # ============================================
    # 修复3: compile - Lambda 层 loss 设计
    # ============================================
    def compile(self):
        """
        修复版 compile

        修复点:
        1. 修正 masked_loss 中的 axis 参数
        2. 添加梯度裁剪
        """
        if self.compiled:
            return

        from hirl_sfc_models import huber_loss

        inputs = self.net.controllerNet.input
        y_true = Input(name='y_true', shape=(self.n_actions,), dtype='float32')
        mask = Input(name='mask', shape=(self.n_actions,), dtype='float32')

        y_pred = self.net.controllerNet.output

        def masked_loss(args):
            y_t, y_p, m = args
            # huber per-element
            loss_elem = huber_loss(y_t, y_p, clip_value=1.0)
            # apply mask (mask already includes IS weights if using weighted version)
            loss_elem = loss_elem * m
            # 🔧 修复: 正确的 mean 计算
            # 先在 action 维度求和，然后在 batch 维度求平均
            return K.mean(K.sum(loss_elem, axis=-1))  # 移除 axis=0

        loss_out = tf.keras.layers.Lambda(masked_loss, name='loss')([y_true, y_pred, mask])

        trainable_model = Model(inputs=inputs + [y_true, mask], outputs=loss_out)

        # 🔧 修复: 添加梯度裁剪
        optimizer = optimizers.RMSprop(
            learning_rate=self.net.lr,
            rho=0.95,
            epsilon=1e-08,
            clipnorm=10.0  # 梯度裁剪
        )

        trainable_model.compile(optimizer=optimizer, loss=lambda y_true, y_pred: y_pred)
        self.trainable_model = trainable_model
        self.compiled = True
        logger.info("[Agent] Trainable model compiled with gradient clipping")

    # ============================================
    # 修复2: train_from_memory - TD-error 计算时机
    # ============================================
    def train_from_memory(self):
        """
        修复版 train_from_memory

        修复点:
        1. 在训练前计算 TD-error（而不是训练后）
        2. 减少不必要的前向传播
        """
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

        # predict Q(s,a) and Q(next)
        q_values = self._safe_predict_q(self.net.controllerNet, states, goals)

        # Double DQN
        q_next_online = self._safe_predict_q(self.net.controllerNet, next_states, goals)
        q_next_target = self._safe_predict_q(self.net.targetControllerNet, next_states, goals)

        # compute target
        next_actions = np.argmax(q_next_online, axis=1)
        q_next_selected = q_next_target[np.arange(self.nSamples), next_actions]
        targets = rewards + (1.0 - dones) * self.gamma * q_next_selected

        # 🔧 修复: 在训练前计算 TD-error
        td_errors_before = np.abs(q_values[np.arange(self.nSamples), actions] - targets)

        # create y_true
        y_true = q_values.copy()
        y_true[np.arange(self.nSamples), actions] = targets

        # create mask
        mask = np.zeros_like(y_true, dtype=np.float32)
        mask[np.arange(self.nSamples), actions] = 1.0

        # 🔧 修复: 应用 importance sampling weights
        # 将权重应用到 mask 上
        weighted_mask = mask * is_weights.reshape(-1, 1)

        # train
        loss = self.trainable_model.train_on_batch(
            [states, goals, y_true, weighted_mask],
            np.zeros((len(states),))
        )

        # 🔧 修复: 使用训练前的 TD-error 更新优先级
        self.memory.update_priorities(idxes, td_errors_before + 1e-6)

        # sync target network
        self.update_count += 1
        if self.update_count % self.hard_update == 0:
            self.net.targetControllerNet.set_weights(self.net.controllerNet.get_weights())
            logger.debug(f"[Agent] Target network synced at step {self.update_count}")

        gc.collect()
        return float(loss)
    # ---------------------------
    # Save / Load
    # ---------------------------
    def save(self, prefix: str):
        self.net.controllerNet.save_weights(f"{prefix}_controller.weights.h5")
        self.net.targetControllerNet.save_weights(f"{prefix}_target_controller.weights.h5")
        logger.info("Agent weights saved to %s_*", prefix)

    def load(self, prefix: str):
        self.net.controllerNet.load_weights(f"{prefix}_controller.weights.h5")
        self.net.targetControllerNet.load_weights(f"{prefix}_target_controller.weights.h5")
        logger.info("Agent weights loaded from %s_*", prefix)

    # ---------------------------
    # Utility: get q-values (for evaluation)
    # ---------------------------
    def q_values(self, state: np.ndarray, goal_one_hot: np.ndarray) -> np.ndarray:
        state_b = self._to_batch(state)
        goal_b = self._to_batch(goal_one_hot)
        return self._safe_predict_q(self.net.controllerNet, state_b, goal_b)[0]

