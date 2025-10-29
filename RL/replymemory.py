# -*- coding: utf-8 -*-
# @File    : replymemory.py
# @Date    : 2024-10-14 (适配 SFCEnv)
# @Author  : chenwei
# @From    :
import random
import operator
from collections import namedtuple
import torch
import numpy as np

# from config import Config # 移除对Config的直接依赖，避免循环引用或缺失属性

# 使用具名元组 快速建立一个类
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class ExperienceReplayMemory:
    def __init__(self, capacity, torch_type) -> None:
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.torch_type = torch_type

    def push(self, *args):
        """保存变换"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        s, a, r, s_ = args
        # 确保动作 a 和奖励 r 都是 (1, 1) 形状的 Tensor
        self.memory[self.position] = Transition(s,
                                                a.reshape(1, -1),
                                                torch.tensor(r, dtype=self.torch_type).reshape(1, -1),
                                                s_)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # 返回 transitions, indices, weights (None, None) 以兼容 PrioritizedReplayMemory 的签名
        return random.sample(self.memory, batch_size), None, None

    def clean_memory(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


class SegmentTree:
    """
    线段树基类，用于高效地进行区间操作 (reduce) 和单点更新 (set item)。
    """

    def __init__(self, capacity, operation, neutral_element):
        # capacity 必须是 2 的幂
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        # 存储线段树的值，大小为 2 * capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation  # 操作符 (e.g., operator.add, min)

    def _reduce_helper(self, query_start, query_end, node, node_start, node_end):
        if query_start == node_start and query_end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if query_end <= mid:
            return self._reduce_helper(query_start, query_end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= query_start:
                return self._reduce_helper(query_start, query_end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(query_start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, query_end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """对 [start, end) 范围进行操作"""
        if end is None:
            end = self._capacity
        if end <= 0:
            end += self._capacity
        end -= 1  # 转换为包含结束索引
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        """更新 idx 处的值"""
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    """求和线段树，用于计算总优先级和按比例采样"""

    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """
        查找第一个前缀和大于或等于 prefixsum 的索引
        """
        try:
            assert 0 <= prefixsum <= self.sum() + np.finfo(np.float32).eps
        except AssertionError:
            print(f"Prefix sum error: {prefixsum}")
            # 允许在调试中退出，但在生产环境中应使用更优雅的错误处理
            exit()

        idx = 1
        while idx < self._capacity:
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    """求最小值线段树，用于计算最小权重 (p_min)"""

    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        return super(MinSegmentTree, self).reduce(start, end)


class PrioritizedReplayMemory:
    """
    优先经验回放 (PER)
    """
    # 修正 beta_frames 的默认值，避免依赖 Config.PKL_NUM
    DEFAULT_BETA_FRAMES = 500 * 3000  # 假设 500 回合 * 3000 步/回合

    def __init__(self, torch_type, size, alpha=0.6, beta_start=0.4, beta_frames=DEFAULT_BETA_FRAMES):
        self.torch_type = torch_type
        # 兼容 rl.py，device 需在 rl.py 中设置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

        assert alpha >= 0
        self._alpha = alpha

        self.beta_start = beta_start
        self.beta_frames = beta_frames  # 使用修正后的默认值
        self.frame = 1  # 帧数/学习步数计数

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def __len__(self):
        return len(self._storage)

    def beta_by_frame(self, frame_idx):
        """线性退火 beta 值，从 beta_start 到 1.0"""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def push(self, *data):
        """
        存储经验，并以当前最大优先级初始化其优先级。
        """
        s, a, r, s_ = data
        data = Transition(s,
                          a.reshape(1, -1),
                          torch.tensor(r, dtype=self.torch_type).reshape(1, -1),
                          s_)

        idx = self._next_idx
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

        # 初始化新经验的优先级为当前最大优先级
        priority = self._max_priority ** self._alpha
        self._it_sum[idx] = priority
        self._it_min[idx] = priority

    def _encode_sample(self, idxes):
        return [self._storage[i] for i in idxes]

    def _sample_proportional(self, batch_size):
        """
        按比例采样 (基于优先级求和树)
        """
        res = []
        # sum(0, len-1) 是当前 memory 中所有优先级的总和
        total_sum = self._it_sum.sum(0, len(self._storage) - 1)
        for _ in range(batch_size):
            # 随机选取一个 [0, total_sum) 中的值
            mass = random.random() * total_sum
            # 找到前缀和大于 mass 的索引
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """
        采样经验、计算 IS 权重，并更新帧数。
        """
        idxes = self._sample_proportional(batch_size)
        weights = []

        # p_min / sum(p) 是最小优先级经验的采样概率
        p_min = self._it_min.min() / self._it_sum.sum()

        beta = self.beta_by_frame(self.frame)

        self.frame += 1  # 更新帧数 (用于 beta 退火)

        # 计算最大权重 (用于归一化)
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            # 当前经验的采样概率
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            # 计算 IS 权重: w_i = ( (1/N) * (1/P_i) )^beta = ( (1/P_i) * (1/N) )^beta
            weight = (p_sample * len(self._storage)) ** (-beta)
            # 归一化权重: w_i / max(w)
            weights.append(weight / max_weight)

        weights = torch.tensor(weights, device=self.device, dtype=self.torch_type)
        encoded_sample = self._encode_sample(idxes)

        # 返回: 经验列表, 索引列表, 权重列表
        return encoded_sample, idxes, weights

    def update_priorities(self, idxes, priorities):
        """
        更新经验池中指定索引的优先级。
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert 0 <= idx < len(self._storage)
            # 优先级 p = |TD_error| + epsilon
            priority_with_eps = priority + np.finfo(np.float32).eps

            # 更新线段树: 存储 p^alpha
            p_alpha = priority_with_eps ** self._alpha
            self._it_sum[idx] = p_alpha
            self._it_min[idx] = p_alpha

            # 更新最大优先级 (用于初始化新经验)
            self._max_priority = max(self._max_priority, priority_with_eps)