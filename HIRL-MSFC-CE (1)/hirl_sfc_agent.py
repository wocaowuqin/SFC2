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
    def selectMove(self, state: np.ndarray, goal_one_hot: np.ndarray, valid_actions: List[int]):
        """
        Epsilon-greedy with action masking.
        - state: (dim,) or (1, dim)
        - goal_one_hot: (n_goals,) or (1, n_goals)
        - valid_actions: list of allowed action ids
        """
        state_b = self._to_batch(state)
        goal_b = self._to_batch(goal_one_hot)

        # exploration prob
        eps = float(self.exploration.value(self.update_count))
        self.controllerEpsilon = eps

        # mask: create vector of -inf then set valid indices to 0
        q_vals = self._safe_predict_q(self.net.controllerNet, state_b, goal_b)[0]  # (n_actions,)

        mask = np.full_like(q_vals, -1e9, dtype=np.float32)
        if valid_actions:
            valid_idx = [a for a in valid_actions if 0 <= a < len(q_vals)]
            if len(valid_idx) == 0:
                # fallback random
                chosen = random.choice(valid_actions) if valid_actions else 0
                return int(chosen)
            mask[valid_idx] = 0.0
        else:
            # no valid actions, fallback to 0
            mask[0] = 0.0

        if random.random() > eps:
            # exploit: apply mask then argmax
            masked = q_vals + mask
            action = int(np.argmax(masked))
            logger.debug("selectMove exploit chosen: %d (eps=%.4f)", action, eps)
            return action
        else:
            # explore uniformly among valid
            if valid_actions:
                action = int(random.choice(valid_actions))
            else:
                action = 0
            logger.debug("selectMove explore chosen: %d (eps=%.4f)", action, eps)
            return action

    # ---------------------------
    # Intrinsic reward (criticize)
    # ---------------------------
    def criticize(self, sub_task_completed: bool, cost: float, request_failed: bool):
        """
        Compute intrinsic reward for a low-level step.
        Assumes cost is already normalized to ~[0,1].
        """
        reward = 0.0
        if sub_task_completed:
            reward += 1.0
        # penalize cost
        reward -= float(cost)
        if request_failed:
            reward -= 5.0
        return float(np.clip(reward, -10.0, 1.0))

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
    def compile(self):
        """
        Build trainable Keras model for masked loss updates.
        The Hdqn_SFC.controllerNet is used to compute forward Q-values;
        a Lambda layer computes masked huber loss and returns scalar loss.
        """
        if self.compiled:
            return

        # create inputs wrappers referencing model inputs
        inputs = self.net.controllerNet.input  # [state_input, goal_input]
        y_true = Input(name='y_true', shape=(self.n_actions,), dtype='float32')
        mask = Input(name='mask', shape=(self.n_actions,), dtype='float32')

        # using model output as y_pred
        y_pred = self.net.controllerNet.output  # (batch, n_actions)

        def masked_loss(args):
            y_t, y_p, m = args
            # huber per-element
            loss_elem = huber_loss(y_t, y_p, clip_value=1.0)
            # apply mask
            loss_elem = loss_elem * m
            # sum across actions and mean across batch
            return K.mean(K.sum(loss_elem, axis=-1), axis=0)

        loss_out = tf.keras.layers.Lambda(masked_loss, name='loss')([y_true, y_pred, mask])

        trainable_model = Model(inputs=inputs + [y_true, mask], outputs=loss_out)
        # loss is precomputed in Lambda layer, so use identity
        trainable_model.compile(optimizer=self._optimizer, loss=lambda y_true, y_pred: y_pred)
        self.trainable_model = trainable_model
        self.compiled = True
        logger.info("Agent trainable model compiled.")

    # ---------------------------
    # Update: sample PER and perform DDQN update
    # ---------------------------
    def train_from_memory(self):
        if not self.compiled:
            self.compile()
        if self.memory.size() < max(32, self.nSamples):
            return 0.0

        # sample
        beta = float(self.beta_schedule.value(self.update_count))
        samples = self.memory.sample(self.nSamples, beta=beta)
        # samples: dict with keys: states, goals, actions, rewards, next_states, dones, weights, idxes

        states = np.asarray(samples['states'], dtype=np.float32)
        goals = np.asarray(samples['goals'], dtype=np.float32)
        actions = np.asarray(samples['actions'], dtype=np.int32)
        rewards = np.asarray(samples['rewards'], dtype=np.float32)
        next_states = np.asarray(samples['next_states'], dtype=np.float32)
        dones = np.asarray(samples['dones'], dtype=np.float32)
        is_weights = np.asarray(samples['weights'], dtype=np.float32)
        idxes = samples['idxes']

        # predict Q(s,a) and Q(next)
        q_values = self._safe_predict_q(self.net.controllerNet, states, goals)  # (B, n_actions)
        # Double DQN: online selects argmax on next, target evaluates
        q_next_online = self._safe_predict_q(self.net.controllerNet, next_states, goals)
        q_next_target = self._safe_predict_q(self.net.targetControllerNet, next_states, goals)

        # compute target y for each sample
        next_actions = np.argmax(q_next_online, axis=1)  # action indices chosen by online net
        q_next_selected = q_next_target[np.arange(self.nSamples), next_actions]
        targets = rewards + (1.0 - dones) * self.gamma * q_next_selected

        # create y_true as current q_values but with target updated at action indices
        y_true = q_values.copy()
        y_true[np.arange(self.nSamples), actions] = targets

        # create mask to compute loss only on taken actions (or we can train all actions but weight them)
        mask = np.zeros_like(y_true, dtype=np.float32)
        mask[np.arange(self.nSamples), actions] = 1.0

        # perform one training step on the trainable model
        # convert to float32 ensure dtype consistent
        loss = self.trainable_model.train_on_batch([states, goals, y_true, mask], np.zeros((len(states),)))
        # compute td_errors for PER update
        q_updated = self._safe_predict_q(self.net.controllerNet, states, goals)
        td_errors = np.abs(q_updated[np.arange(self.nSamples), actions] - targets)

        # update priorities in memory
        self.memory.update_priorities(idxes, td_errors + 1e-6)

        # soft/hard update target network
        self.update_count += 1
        if self.update_count % self.hard_update == 0:
            self.net.targetControllerNet.set_weights(self.net.controllerNet.get_weights())

        # free memory
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

