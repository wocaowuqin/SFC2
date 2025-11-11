#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_models.py

import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


def huber_loss(y_true, y_pred, clip_value=1.0):
    """Huber loss, 确保 clip_value > 0."""
    x = y_true - y_pred
    condition = K.abs(x) < clip_value
    squared_loss = 0.5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - 0.5 * clip_value)
    if K.backend() == 'tensorflow':
        return tf.where(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))


class MetaControllerNN:
    """
    高层元控制器 (SFC MLP 版本)
    (基于: meta_net_il.py)
    """

    def __init__(self, state_shape, n_goals, lr=0.00025):
        self.state_shape = state_shape
        self.n_goals = n_goals
        self.replay_hist = [None] * 1000  # DAgger 聚合缓冲区
        self.ind = 0
        self.count = 0

        rmsProp = optimizers.RMSprop(learning_rate=lr, rho=0.95, epsilon=1e-08)
        self.meta_controller = self._build_model()
        self.meta_controller.compile(loss='categorical_crossentropy', optimizer=rmsProp)

    # def _build_model(self):
    #     """H-DQN 架构: 输入 = (State, Goal), 输出 = Q(State, Goal, Action)"""
    #     state_input = Input(shape=self.state_shape, name='state_input')
    #     goal_input = Input(shape=(self.n_goals,), name='goal_input')
    #
    #     merged_input = concatenate([state_input, goal_input])
    #
    #     x = Dense(256, activation='relu')(merged_input)
    #     x = Dense(256, activation='relu')(x)
    #     output = Dense(self.n_actions, activation='linear', name='q_values')(x)
    #
    #     model = Model(inputs=[state_input, goal_input], outputs=output)
    #
    #     # ✅ 修复：在这里定义 optimizer
    #     rmsProp = optimizers.RMSprop(learning_rate=self.lr, rho=0.95, epsilon=1e-08)
    #
    #     model.compile(loss='mse', optimizer=rmsProp)
    #     return model
    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=self.state_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_goals, activation='softmax'))  # ✅ 正确: 输出 n_goals
        return model
    def check_training_clock(self):
        return (self.count >= 100)  # 每100次收集就训练一次

    def collect(self, state, expert_goal):
        """收集 (状态, 专家高层目标)"""
        expert_goal_one_hot = to_categorical(expert_goal, num_classes=self.n_goals)
        self.replay_hist[self.ind] = (state.astype(np.float32), expert_goal_one_hot.astype(np.float32))
        self.ind = (self.ind + 1) % len(self.replay_hist)
        self.count += 1

    def train(self):
        """训练 DAgger 策略"""
        num_valid = self.ind if self.replay_hist[-1] is None else len(self.replay_hist)
        if num_valid < 32: return  # 样本不足

        samples = random.sample(range(num_valid), 32)
        data = [self.replay_hist[i] for i in samples]

        train_x = np.array([d[0] for d in data])
        train_y = np.array([d[1] for d in data])

        self.meta_controller.fit(train_x, train_y, batch_size=32, epochs=1, verbose=0)
        self.count = 0  # 重置计数器

    def predict(self, x):
        return self.meta_controller.predict(x, verbose=0)[0]

    def sample(self, prob_vec, temperature=0.1):
        """(来自: meta_net_il.py)"""
        prob_pred = np.log(prob_vec) / temperature
        dist = np.exp(prob_pred) / np.sum(np.exp(prob_pred))
        choices = range(len(prob_pred))
        return np.random.choice(choices, p=dist)


class Hdqn_SFC:
    """
    低层控制器 Q-Network (SFC MLP 版本)
    (基于: hybrid_model_atari.py)
    """

    def __init__(self, state_shape, n_goals, n_actions, lr=0.00025):
        self.state_shape = state_shape
        self.n_goals = n_goals
        self.n_actions = n_actions
        self.lr = lr

        self.controllerNet = self._build_model()
        self.targetControllerNet = self._build_model()
        self.targetControllerNet.set_weights(self.controllerNet.get_weights())

    def _build_model(self):
        """H-DQN 架构: 输入 = (State, Goal), 输出 = Q(State, Goal, Action)"""
        state_input = Input(shape=self.state_shape, name='state_input')
        goal_input = Input(shape=(self.n_goals,), name='goal_input')

        merged_input = concatenate([state_input, goal_input])

        x = Dense(256, activation='relu')(merged_input)
        x = Dense(256, activation='relu')(x)
        output = Dense(self.n_actions, activation='linear', name='q_values')(x)

        model = Model(inputs=[state_input, goal_input], outputs=output)
        rmsProp = optimizers.RMSprop(learning_rate=self.lr, rho=0.95, epsilon=1e-08)
        model.compile(loss='mse', optimizer=rmsProp)  # 'mse' 只是占位符，实际 loss 在 Agent 中计算
        return model

    def saveWeight(self, file_prefix):
        self.controllerNet.save_weights(f"{file_prefix}_controller.weights.h5")

    def loadWeight(self, file_prefix):
        self.controllerNet.load_weights(f"{file_prefix}_controller.weights.h5")
        self.controllerNet.reset_states()