#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_utils.py

import random
import numpy as np


# ================================================================
# Segment Tree for PER
# ================================================================
class SegmentTree(object):
    """
    Segment Tree: 优先经验回放 (PER) 的底层数据结构。
    """

    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, \
            "capacity must be positive and a power of 2."

        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]

        mid = (node_start + node_end) // 2

        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        elif start > mid:
            return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
        else:
            return self._operation(
                self._reduce_helper(start, mid, 2 * node, node_start, mid),
                self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
            )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2

        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx], self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        return self._value[idx + self._capacity]

    def find_prefixsum_idx(self, prefixsum):
        idx = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


# ================================================================
# 线性退火调度器
# ================================================================
class LinearSchedule(object):
    """
    线性退火调度器 (用于 Epsilon-Greedy)
    """

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


# ================================================================
# 优先经验回放 PER
# ================================================================
class PrioritizedReplayBuffer(object):
    """
    优先经验回放 (PER) 缓冲区
    """

    def __init__(self, size, alpha):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SegmentTree(it_capacity, lambda x, y: x + y, 0)
        self._it_min = SegmentTree(it_capacity, min, float('inf'))
        self._max_priority = 1.0

    # ----------------------------------------------------
    # 🔧 兼容 Agent_SFC.train_from_memory() 所需
    # ----------------------------------------------------
    def size(self):
        return len(self._storage)

    def __len__(self):
        return len(self._storage)

    # ----------------------------------------------------
    # 添加经验
    # ----------------------------------------------------
    def add(self, *args):
        data = args   # (s, g, a, r, ns, done)

        idx = self._next_idx

        if idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[idx] = data

        self._next_idx = (idx + 1) % self._maxsize

        priority = self._max_priority ** self._alpha
        self._it_sum[idx] = priority
        self._it_min[idx] = priority

    # ----------------------------------------------------
    # 按比例采样
    # ----------------------------------------------------
    def _sample_proportional(self, batch_size):
        total = self._it_sum.reduce()
        res = []
        for _ in range(batch_size):
            mass = random.random() * total
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    # ----------------------------------------------------
    # 采样 batch
    # ----------------------------------------------------
    def sample(self, batch_size, beta):
        assert self.size() > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        total = self._it_sum.reduce()

        # min sampling prob
        p_min = self._it_min.reduce() / total
        max_weight = (p_min * self.size()) ** (-beta)

        states, goals = [], []
        actions, rewards = [], []
        next_states, dones = [], []

        for idx in idxes:
            s, g, a, r, ns, d = self._storage[idx]

            states.append(s)
            goals.append(g)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

            p_sample = self._it_sum[idx] / total
            w = (p_sample * self.size()) ** (-beta)
            weights.append(w / max_weight)

        return {
            "states": np.array(states, dtype=np.float32),
            "goals": np.array(goals, dtype=np.float32),
            "actions": np.array(actions, dtype=np.int32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones": np.array(dones, dtype=np.float32),
            "weights": np.array(weights, dtype=np.float32),
            "idxes": idxes
        }

    # ----------------------------------------------------
    # 更新优先级
    # ----------------------------------------------------
    def update_priorities(self, idxes, priorities):
        for idx, p in zip(idxes, priorities):
            p = max(float(p), 1e-6)
            self._it_sum[idx] = p ** self._alpha
            self._it_min[idx] = p ** self._alpha
            self._max_priority = max(self._max_priority, p)
