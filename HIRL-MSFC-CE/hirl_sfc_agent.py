#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_agent.py

import numpy as np
import random
from collections import namedtuple
from typing import List

# 导入 Keras/TF 和辅助工具
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K

# 导入自定义模块
from hirl_utils import PrioritizedReplayBuffer, LinearSchedule
from hirl_sfc_models import Hdqn_SFC, huber_loss


class Agent_SFC:
    """
    低层策略代理 (SFC MLP 版本)
    (基于: hybrid_rl_il_agent_atari.py)
    """

    def __init__(self, net: Hdqn_SFC, n_actions: int, mem_cap: int, exploration_steps: int,
                 train_freq: int, hard_update: int, n_samples: int, gamma: float):

        self.net = net
        self.action_set = range(n_actions)
        self.n_actions = n_actions
        self.nSamples = n_samples
        self.gamma = gamma
        self.memory = PrioritizedReplayBuffer(mem_cap, alpha=0.6)
        self.exploration = LinearSchedule(schedule_timesteps=exploration_steps, initial_p=1.0, final_p=0.02)
        self.trainFreq = train_freq
        self.hard_update = hard_update
        self.beta_schedule = LinearSchedule(exploration_steps, initial_p=0.4, final_p=1.0)  # PER beta

        self.controllerEpsilon = 1.0
        self.randomPlay = True  # 训练初期标志
        self.trainable_model = None
        self.compiled = False

    # ----------------------------------------------------
    # ✅ 修复 #6: 更改签名以接受 valid_actions
    # ----------------------------------------------------
    def selectMove(self, state, goal_one_hot, valid_actions: List[int]):
        """低层动作选择 (Epsilon-Greedy + 动作掩码)"""
        if self.controllerEpsilon < random.random():
            # (利用)
            q_values = self.net.controllerNet.predict([state, goal_one_hot], verbose=0)[0]

            # 仅在有效动作中选择
            valid_q = {a: q_values[a] for a in valid_actions if a < len(q_values)}
            if not valid_q:
                return random.choice(valid_actions)  # 如果掩码出问题，随机选
            return max(valid_q, key=valid_q.get)

        # (探索)
        return random.choice(valid_actions)

    def criticize(self, sub_task_completed: bool, cost: float, request_failed: bool):
        """
        计算内部奖励 (Intrinsic Reward)
        我们希望“低成本”完成“高层指定”的子任务
        """
        reward = 0.0
        if sub_task_completed:
            reward += 1.0  # 完成子任务

        # 奖励与成本负相关 (cost 已经归一化)
        reward -= cost

        if request_failed:  # 如果动作导致整个请求失败
            reward -= 5.0  # 巨大惩罚

        return np.clip(reward, -5.0, 1.0)

    def store(self, experience: namedtuple):
        """
        存储 (s, g, a, r, s', done)
        ✅ 修复 #4 (相关): 假设 experience.goal 已经是 one-hot 编码
        """
        self.memory.add(
            experience.state, experience.goal, experience.action,
            experience.reward, experience.next_state, experience.done
        )

    def compile(self):
        """(来自 atari) 构建 Keras 的自定义 loss 训练模型"""

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, clip_value=1.0)
            loss *= mask  # 按元素应用掩码
            return K.sum(loss, axis=-1)

        y_pred = self.net.controllerNet.output
        y_true = Input(name='y_true', shape=(self.n_actions,))
        mask = Input(name='mask', shape=(self.n_actions,))

        loss_out = Lambda(clipped_masked_error, output_shape=(1,), name='loss')([y_pred, y_true, mask])

        ins = self.net.controllerNet.input  # [state_input, goal_input]
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])

        losses = [
            lambda y_true, y_pred: y_pred,  # loss 在 Lambda 层计算
            lambda y_true, y_pred: K.zeros_like(y_pred),
        ]
        rmsProp = optimizers.RMSprop(learning_rate=self.net.lr, rho=0.95, epsilon=1e-08)
        trainable_model.compile(optimizer=rmsProp, loss=losses)

        self.trainable_model = trainable_model
        self.compiled = True

    def _update(self, stepCount):
        """(来自 atari) 执行一次 DQN (PER + Double DQN) 更新"""

        # 1. 采样
        batches = self.memory.sample(self.nSamples, beta=self.beta_schedule.value(stepCount))
        (stateVector, goalVector, actionVector, rewardVector, nextStateVector, doneVector,
         importanceVector, idxVector) = batches

        # (goalVector 已经是 one-hot 编码)

        # 2. Double DQN
        # 使用 controllerNet 选动作
        q_values_next = self.net.controllerNet.predict([nextStateVector, goalVector], verbose=0)
        actions_next = np.argmax(q_values_next, axis=1)

        # 使用 targetControllerNet 评估 Q 值
        target_q_values_next = self.net.targetControllerNet.predict([nextStateVector, goalVector], verbose=0)
        q_batch = target_q_values_next[range(self.nSamples), actions_next]

        # 3. 计算 Bellman 目标
        targets = np.zeros((self.nSamples, self.n_actions))
        dummy_targets = np.zeros((self.nSamples,))
        masks = np.zeros((self.nSamples, self.n_actions))

        terminal_batch = np.array([1.0 - float(d) for d in doneVector])
        discounted_reward_batch = self.gamma * q_batch * terminal_batch
        Rs = rewardVector + discounted_reward_batch

        q_values_current = self.net.controllerNet.predict([stateVector, goalVector], verbose=0)
        td_errors = np.zeros(self.nSamples)

        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, actionVector)):
            target[action] = R
            dummy_targets[idx] = R
            mask[action] = 1.
            td_errors[idx] = R - q_values_current[idx, action]

        # 4. 更新 PER 优先级
        new_priorities = np.abs(td_errors) + 1e-6
        self.memory.update_priorities(idxVector, new_priorities)

        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        # 5. 训练
        ins = [stateVector, goalVector]
        loss = self.trainable_model.train_on_batch(
            ins + [targets, masks],
            [dummy_targets, targets],
            sample_weight=[np.array(importanceVector), np.ones(self.nSamples)]
        )

        # 6. 硬更新
        if stepCount > 0 and stepCount % self.hard_update == 0:
            self.net.targetControllerNet.set_weights(self.net.controllerNet.get_weights())

        return loss[1], np.mean(q_values_current), np.mean(np.abs(td_errors))

    def update(self, stepCount):
        if not self.compiled:
            print("错误: 代理未编译 (agent.compile())")
            return 0, 0, 0
        if len(self.memory) < self.nSamples:
            return 0, 0, 0

        return self._update(stepCount)

    def annealControllerEpsilon(self, stepCount, option_learned=False):
        """(来自 atari) Epsilon 退火"""
        if not self.randomPlay:
            if option_learned:
                self.controllerEpsilon = 0.0
            else:
                self.controllerEpsilon = self.exploration.value(stepCount)