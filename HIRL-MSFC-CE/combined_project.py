#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# === 这是一个由7个文件合并而成的项目 ===
#
# 原始文件包括:
# 1. hyperparameters.py
# 2. hirl_utils.py
# 3. expert_msfce.py
# 4. hirl_sfc_models.py
# 5. hirl_sfc_env.py
# 6. hirl_sfc_agent.py
# 7. train_hirl_sfc.py
#
# =======================================

import os
import random
import numpy as np
import scipy.io as sio
import gym
from gym import spaces
from pathlib import Path
from collections import namedtuple
from typing import Dict, List, Tuple, Set, Optional, Any

# --- Matplotlib & Pandas (用于日志) ---
import matplotlib

matplotlib.use('Agg')  # 设置后端以用于非交互式保存
import matplotlib.pyplot as plt
import pandas as pd

# --- TensorFlow / Keras ---
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate, Lambda
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical


# =======================================
# === 文件 1: hyperparameters.py
# =======================================
# (封装在一个类中以保留 H.VARIABLE 语法)

class H:
    # --- 1. 路径和环境配置 ---
    INPUT_DIR = Path('E:/pycharmworkspace/SFC-master/HIRL-MSFC-CE/mat')
    OUTPUT_DIR = Path('E:/pycharmworkspace/SFC-master/HIRL-MSFC-CE/out_hirl')

    CAPACITIES = {
        'cpu': 2000.0,
        'memory': 1100.0,
        'bandwidth': 500.0
    }

    DC_NODES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 27, 28]

    # 拓扑矩阵
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

    # --- 2. 训练超参数 ---
    PRE_TRAIN_STEPS = 10000  # 阶段1: 纯模仿学习步数
    EPISODE_LIMIT = 5000  # 阶段2: 总训练回合数
    STEPS_LIMIT = 2000000  # 总步数限制
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 0.00025
    EXP_MEMORY = 100000
    EXPLORATION_STEPS = 500000
    TRAIN_FREQ = 4
    HARD_UPDATE_FREQUENCY = 1000
    META_TRAIN_FREQ = 100  # DAgger 训练频率


# =======================================
# === 文件 2: hirl_utils.py
# =======================================

class SegmentTree(object):
    """
    Segment Tree: 优先经验回放 (PER) 的底层数据结构。
    """

    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            # ----------------------------------------------------
            # ✅ 修复 #1: (self.value -> self._value)
            # ----------------------------------------------------
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
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
        end -= 1  # 您的建议：此处可读性差，但保持原逻辑
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        idx_in_leaf = idx + self._capacity
        self._value[idx_in_leaf] = val
        idx = idx_in_leaf // 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
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

    def add(self, *args):
        idx = self._next_idx
        # ----------------------------------------------------
        # ✅ 修复 (建议): data = (args) -> data = args
        # ----------------------------------------------------
        data = args
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.reduce()
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    # ----------------------------------------------------
    # ✅ 修复 #5: 替换为您的新 sample 方法
    # ----------------------------------------------------
    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.reduce() / self._it_sum.reduce()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        states, goals, actions, rewards, next_states, dones = [], [], [], [], [], []

        for idx in idxes:
            data = self._storage[idx]
            s, g, a, r, ns, d = data
            states.append(s)
            goals.append(g)  # g 已经是 one-hot 数组
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

            p_sample = self._it_sum[idx] / self._it_sum.reduce()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        # ✅ 确保goals是正确的形状和类型
        return (
            np.array(states, dtype=np.float32),
            np.array(goals, dtype=np.float32),  # (batch, n_goals)
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            np.array(weights, dtype=np.float32),
            idxes
        )

    def update_priorities(self, idxes, priorities):
        for idx, priority in zip(idxes, priorities):
            priority = max(priority, 1e-6)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def __len__(self):
        return len(self._storage)


# =======================================
# === 文件 3: expert_msfce.py
# =======================================

def parse_mat_request(req_obj) -> Dict:
    """
    将 MATLAB 请求结构 (来自 sorted_requests.mat) 解析为 Python 字典
    (代码来自: msfce_simulator_fixed.py)
    """
    req = req_obj
    try:
        parsed = {
            'id': int(req['id'][0, 0]),
            'source': int(req['source'][0, 0]),
            'dest': [int(d) for d in req['dest'].flatten()],
            'vnf': [int(v) for v in req['vnf'].flatten()],
            'bw_origin': float(req['bw_origin'][0, 0]),
            'cpu_origin': [float(c) for c in req['cpu_origin'].flatten()],
            'memory_origin': [float(m) for m in req['memory_origin'].flatten()],
            'arrival_time': int(req['arrival_time'][0, 0]),
            'leave_time': int(req['leave_time'][0, 0]),
        }
    except (IndexError, TypeError):
        parsed = {
            'id': int(req[0][0][0]),
            'source': int(req[0][1][0]),
            'dest': [int(x) for x in req[0][2].flatten()],
            'vnf': [int(x) for x in req[0][3].flatten()],
            'cpu_origin': [float(x) for x in req[0][4].flatten()],
            'memory_origin': [float(x) for x in req[0][5].flatten()],
            'bw_origin': float(req[0][6][0][0])
        }
    return parsed


class MSFCE_Solver:
    """
    MSFC-CE 启发式算法求解器 (无状态)
    (代码来自: msfce_simulator_fixed.py)
    这是我们的“专家预言机”，用于生成模仿学习的标签。
    """

    def __init__(self, path_db_file: Path, topology_matrix: np.ndarray,
                 dc_nodes: List[int], capacities: Dict):
        print("初始化 MSFC-CE 专家求解器...")
        try:
            self.path_db = sio.loadmat(path_db_file)['Paths']
        except FileNotFoundError:
            print(f"❌ 致命错误: 找不到 MAT 文件 {path_db_file}")
            raise
        except KeyError:
            print(f"❌ 致命错误: MAT 文件 {path_db_file} 中缺少 'Paths' 键")
            raise

        self.node_num = topology_matrix.shape[0]
        self.type_num = 8
        self.k_path_count = 5
        self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)
        self.cpu_capacity = capacities['cpu']
        self.memory_capacity = capacities['memory']
        self.bandwidth_capacity = capacities['bandwidth']
        self.link_num, self.link_map = self._create_link_map(topology_matrix)
        print(f"✅ 专家加载: {self.node_num}节点, {self.link_num}链路, {self.dc_num}DC")

    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        link_map, lid = {}, 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def _get_path_from_db(self, src: int, dst: int, k: int):
        if src < 1 or dst < 1 or src > self.node_num or dst > self.node_num:
            return [], 0, []
        try:
            p = self.path_db[src - 1, dst - 1]
            dist = int(p['pathsdistance'][k - 1][0])
            nodes = p['paths'][k - 1, :dist + 1].astype(int).tolist()
            links = [self.link_map[(nodes[i], nodes[i + 1])]
                     for i in range(len(nodes) - 1)
                     if (nodes[i], nodes[i + 1]) in self.link_map]
            return nodes, dist, links
        except Exception:
            return [], 0, []

    def _get_kth_path_max_distance(self, src: int, dst: int, kpath: int) -> int:
        try:
            return int(self.path_db[src - 1, dst - 1]['pathsdistance'][kpath - 1][0])
        except Exception:
            return 1

    def _calc_score(self, src: int, dst: int, dist: int, dc_count: int,
                    cpu_sum: float, mem_sum: float, bw_sum: float) -> float:
        """计算评分函数"""
        max_dist = self._get_kth_path_max_distance(src, dst, self.k_path_count) or 1
        score = (
                (1 - dist / max_dist) +
                dc_count / self.dc_num +
                cpu_sum / (self.cpu_capacity * self.dc_num) +
                mem_sum / (self.memory_capacity * self.dc_num) +
                bw_sum / (self.bandwidth_capacity * self.link_num)
        )
        return score

    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """评估 S->d 的第k条路径"""
        bw, cpu, mem, hvt = state['bw'], state['cpu'], state['mem'], state['hvt']
        src, dest = request['source'], request['dest'][d_idx]

        path, dist, links = self._get_path_from_db(src, dest, k)
        if not path:
            return 0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dest, 0

        tree = np.zeros(self.link_num)
        hvt_new = np.zeros((self.node_num, self.type_num))
        usable = [n for n in path if n in self.DC]

        # 检查资源
        if len(usable) < len(request['vnf']):
            return 0, path, tree, hvt_new, False, dest, 0
        for lid in links:
            if lid - 1 >= len(bw) or bw[lid - 1] < request['bw_origin']:
                return 0, path, tree, hvt_new, False, dest, 0

        # VNF 放置
        j, i = 0, 0
        while j < len(request['vnf']):
            if i >= len(usable):
                return 0, path, tree, hvt_new, False, dest, 0
            node, vnf_t = usable[i] - 1, request['vnf'][j] - 1
            if hvt[node, vnf_t] == 0:
                if cpu[node] < request['cpu_origin'][j] or mem[node] < request['memory_origin'][j]:
                    i += 1
                    continue
            hvt_new[node, vnf_t] = 1
            j, i = j + 1, i + 1

        if np.sum(hvt_new) != len(request['vnf']):
            return 0, path, tree, hvt_new, False, dest, 0

        for lid in links:
            tree[lid - 1] = 1

        # 计算成本
        cost = self._calculate_cost(request, state, tree, hvt_new)

        # 计算得分
        score = self._calc_score(src, dest, dist, len(usable),
                                 np.sum(cpu[np.array(path) - 1]),
                                 np.sum(mem[np.array(path) - 1]),
                                 np.sum(bw[np.array(links) - 1]))

        return score, path, tree, hvt_new, True, dest, cost

    def _calc_eval1(self, d_idx: int, k: int, i_idx: int, tree1_path: List[int],
                    request: Dict, tree1_hvt: np.ndarray, state: Dict, nodes_on_tree: Set[int]):
        """评估从树上第 i_idx 个节点到目的节点 d_idx 的第 k 条路径"""
        hvt = tree1_hvt.copy()
        tree = np.zeros(self.link_num)
        tree_paths = tree1_path[:i_idx + 1]

        connect_node = tree1_path[i_idx]
        dest_node = request['dest'][d_idx]

        paths, dist, links = self._get_path_from_db(connect_node, dest_node, k)

        if not paths or len(paths) < 2:
            return 0, [], tree, hvt, False, dest_node, 0

        # 检测环路
        arr1 = set(paths[1:])
        arr2 = set(tree_paths)
        if arr1 & arr2:
            return 0, paths, tree, hvt, False, dest_node, 0
        arr4 = nodes_on_tree - set(tree_paths)
        if arr1 & arr4:
            return 0, paths, tree, hvt, False, dest_node, 0
        if i_idx + 1 < len(tree1_path):
            arr6 = set(tree1_path[i_idx + 1:])
            if arr1 & arr6:
                return 0, paths, tree, hvt, False, dest_node, 0

        usable_on_path = [n for n in paths[1:] if n in self.DC]
        deployed_on_path = [n for n in tree_paths if n in self.DC]

        for lid in links:
            if lid - 1 >= len(state['bw']) or state['bw'][lid - 1] < request['bw_origin']:
                return 0, paths, tree, hvt, False, dest_node, 0

        CPU_status = sum(state['cpu'][n - 1] for n in paths[1:] if n in self.DC)
        Memory_status = sum(state['mem'][n - 1] for n in paths[1:] if n in self.DC)
        Bandwidth_status = sum(state['bw'][lid - 1] for lid in links)

        shared_path_deployed = sum(
            1 for vnf_type in request['vnf']
            if any(hvt[n - 1, vnf_type - 1] > 0 for n in deployed_on_path)
        )
        undeployed_vnf = len(request['vnf']) - shared_path_deployed

        if undeployed_vnf == 0:
            eval_score = self._calc_score(
                connect_node, dest_node, dist,
                len(deployed_on_path), CPU_status, Memory_status, Bandwidth_status
            )
            for lid in links:
                tree[lid - 1] = 1
            cost = self._calculate_cost(request, state, tree, hvt)
            return eval_score, paths, tree, hvt, True, 0, cost
        else:
            if len(usable_on_path) < undeployed_vnf:
                return 0, paths, tree, hvt, False, dest_node, 0

            j, g = shared_path_deployed, 0
            while j < len(request['vnf']) and g < len(usable_on_path):
                node_idx = usable_on_path[g] - 1
                vnf_type = request['vnf'][j] - 1
                if hvt[node_idx, vnf_type] == 0:
                    if (state['cpu'][node_idx] < request['cpu_origin'][j] or
                            state['mem'][node_idx] < request['memory_origin'][j]):
                        g += 1
                        continue
                hvt[node_idx, vnf_type] = 1
                j += 1
                g += 1

            total_deployed = sum(
                1 for vnf_type in request['vnf']
                if any(hvt[n - 1, vnf_type - 1] > 0 for n in (deployed_on_path + usable_on_path))
            )
            if total_deployed != len(request['vnf']):
                return 0, paths, tree, hvt, False, dest_node, 0

            eval_score = self._calc_score(
                connect_node, dest_node, dist,
                len(usable_on_path), CPU_status, Memory_status, Bandwidth_status
            )
            for lid in links:
                tree[lid - 1] = 1
            cost = self._calculate_cost(request, state, tree, hvt)
            return eval_score, paths, tree, hvt, True, 0, cost

    # ----------------------------------------------------
    # ✅ 修复 #1: 替换为您的新 _calculate_cost 方法
    # ----------------------------------------------------
    def _calculate_cost(self, request: Dict, state: Dict, tree: np.ndarray, hvt: np.ndarray) -> float:
        """计算部署此方案的资源成本 (用于RL奖励)"""
        bw_cost, cpu_cost, mem_cost = 0, 0, 0

        used_links = np.where(tree > 0)[0]
        if used_links.size > 0:
            new_links_mask = (state['bw_ref_count'][used_links] == 0)
            bw_cost = np.sum(new_links_mask) * request['bw_origin']

        for node, vnf_t in np.argwhere(hvt > 0):
            if state['hvt'][node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    cpu_cost += request['cpu_origin'][j]
                    mem_cost += request['memory_origin'][j]
                except ValueError:
                    pass

        # ✅ 修复: 定义归一化权重
        # 权重设计原则:
        # - 带宽是共享资源，权重较低
        # - CPU和内存是节点独占资源，权重较高
        # - 归一化到容量，使不同资源类型可比较

        bw_weight = 1.0 / self.bandwidth_capacity  # 归一化到 [0, 1]
        cpu_weight = 10.0 / self.cpu_capacity  # CPU更重要
        mem_weight = 10.0 / self.memory_capacity  # 内存同样重要

        # 计算归一化成本
        total_cost = (bw_cost * bw_weight) + (cpu_cost * cpu_weight) + (mem_cost * mem_weight)

        # 归一化到 [0, 10] 范围，便于奖励函数处理
        # 假设最坏情况：使用所有资源
        max_possible_cost = (
                self.link_num * self.bandwidth_capacity * bw_weight +
                self.dc_num * self.cpu_capacity * cpu_weight +
                self.dc_num * self.memory_capacity * mem_weight
        )

        # 避免除以零
        if max_possible_cost == 0:
            return 0.0

        normalized_cost = (total_cost / max_possible_cost) * 10.0
        return np.clip(normalized_cost, 0, 10)

    def _calc_atnp(self, tree1: Dict, tree1_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """
        (专家函数): 找到将目的节点 d 连接到树 tree1 的最佳方案
        """
        request = state['request']

        if tree1.get('eval', 0) == 0:
            return {
                'tree': tree1['tree'].copy(), 'hvt': tree1['hvt'].copy(),
                'feasible': tree1.get('feasible', False),
                'infeasible_dest': tree1.get('infeasible_dest', 0)
            }, 0, (0, 0), 0  # (plan, eval, action, cost)

        best_eval = -1
        best_plan = None
        best_action = (0, 0)  # (i_idx, k)
        best_cost = 0

        # 遍历树上的所有可能连接点 (i_idx)
        for i_idx in range(len(tree1_path)):
            # 遍历 K 条路径 (k)
            for k in range(1, self.k_path_count + 1):
                eval_val, paths, tree_new, hvt_new, feasible, infeasible_dest, cost = \
                    self._calc_eval1(
                        d_idx, k, i_idx, tree1_path, request,
                        tree1['hvt'], state, nodes_on_tree
                    )

                if feasible and eval_val > best_eval:
                    best_eval = eval_val
                    best_action = (i_idx, k - 1)  # (0-indexed)
                    best_cost = cost
                    best_plan = {
                        'tree': tree_new, 'hvt': hvt_new, 'new_path_full': paths,
                        'connect_idx': i_idx, 'feasible': True, 'infeasible_dest': 0
                    }

        if best_plan is None:
            return {
                'tree': tree1['tree'].copy(), 'hvt': tree1['hvt'].copy(),
                'feasible': False, 'infeasible_dest': request['dest'][d_idx]
            }, 0, (0, 0), 0

        return best_plan, best_eval, best_action, best_cost

    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> \
            Tuple[Optional[Dict], List[Tuple[int, Tuple[int, int], float]]]:
        """
        (专家函数): 运行 MSFC-CE 算法并记录"专家决策"
        返回: (最终方案, 轨迹)
        轨迹 = [(high_level_goal, low_level_action, cost), ...]
        """
        dest_num = len(request['dest'])
        network_state['request'] = request

        # (高层, 低层, 成本) 轨迹
        expert_trajectory = []

        # 阶段1: 找到所有 S->d 的最佳路径
        tree_set = []
        best_k_set = []
        best_cost_set = []
        for d_idx in range(dest_num):
            best_eval, best_result, best_k, best_cost = -1, None, 0, 0
            for k in range(1, self.k_path_count + 1):
                eval_val, paths, tree, hvt, feasible, _, cost = self._calc_eval(
                    request, d_idx, k, network_state)
                if feasible and eval_val > best_eval:
                    best_eval, best_k, best_cost = eval_val, (k - 1), cost
                    best_result = {
                        'eval': eval_val, 'paths': paths, 'tree': tree, 'hvt': hvt
                    }
            tree_set.append(best_result if best_result else {'eval': -1})
            best_k_set.append(best_k)
            best_cost_set.append(best_cost)

        # 贪心构建多播树
        best_d_idx = np.argmax([t.get('eval', -1) for t in tree_set])
        if tree_set[best_d_idx]['eval'] <= 0:
            return None, []  # 阻塞

        # 记录第一个高层和低层决策
        high_level_goal = best_d_idx
        low_level_action = (0, best_k_set[best_d_idx])
        cost = best_cost_set[best_d_idx]
        expert_trajectory.append((high_level_goal, low_level_action, cost))

        current_tree = {
            'id': request['id'],
            'tree': tree_set[best_d_idx]['tree'],
            'hvt': tree_set[best_d_idx]['hvt'],
            'paths_map': {request['dest'][best_d_idx]: tree_set[best_d_idx]['paths']}
        }
        nodes_on_tree = set(tree_set[best_d_idx]['paths'])
        unadded = set(range(dest_num)) - {best_d_idx}

        # ----------------------------------------------------
        # ✅ 修复 #4: (方案 2) 模拟状态更新
        # ----------------------------------------------------

        # 备份原始状态，以便函数退出时恢复
        # 注意：我们只备份引用类型（数组），值类型（如 link_ref_count）不需要
        # （但 network_state 并没有包含 link_ref_count... 这是一个潜在问题，
        # 专家目前没有考虑 ref_count。为保持与您代码一致，我们只备份传入的）

        # 拷贝 network_state 字典中的数组，以进行本地修改
        local_network_state = {
            'bw': network_state['bw'].copy(),
            'cpu': network_state['cpu'].copy(),
            'mem': network_state['mem'].copy(),
            'hvt': network_state['hvt'].copy(),
            'bw_ref_count': network_state['bw_ref_count'].copy(),  # 假设 ref_count 也传入
            'request': request  # request 是共享的
        }

        while unadded:
            best_eval, best_plan, best_d, best_action, best_cost = -1, None, -1, (0, 0), 0

            for d_idx in unadded:
                # 遍历所有已在树上的路径，看从哪里连接
                for conn_path in current_tree['paths_map'].values():
                    t, m, action, cost = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, local_network_state, nodes_on_tree  # ✅ 使用本地状态
                    )
                    if t.get('feasible') and m > best_eval:
                        best_eval, best_plan, best_d = m, t, d_idx
                        best_action, best_cost = action, cost

            if best_d == -1:
                return None, []  # 阻塞

            # 记录后续的高层和低层决策
            high_level_goal = best_d
            low_level_action = best_action  # (i_idx, k_idx)
            cost = best_cost
            expert_trajectory.append((high_level_goal, low_level_action, cost))

            # 合并树
            current_tree['tree'] = np.logical_or(current_tree['tree'], best_plan['tree']).astype(float)
            current_tree['hvt'] = np.maximum(current_tree['hvt'], best_plan['hvt'])
            current_tree['paths_map'][request['dest'][best_d]] = best_plan['new_path_full']
            nodes_on_tree.update(best_plan['new_path_full'])
            unadded.remove(best_d)

            # ✅ 修复 #4: 临时在 local_network_state 上应用资源变化
            for link_idx in np.where(best_plan['tree'] > 0)[0]:
                if local_network_state['bw_ref_count'][link_idx] == 0:
                    local_network_state['bw'][link_idx] -= request['bw_origin']
                local_network_state['bw_ref_count'][link_idx] += 1

            for node, vnf_t in np.argwhere(best_plan['hvt'] > 0):
                if local_network_state['hvt'][node, vnf_t] == 0:
                    try:
                        j = request['vnf'].index(vnf_t + 1)
                        local_network_state['cpu'][node] -= request['cpu_origin'][j]
                        local_network_state['mem'][node] -= request['memory_origin'][j]
                    except ValueError:
                        pass
                local_network_state['hvt'][node, vnf_t] += 1

        return current_tree, expert_trajectory


# =======================================
# === 文件 4: hirl_sfc_models.py
# =======================================

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


# =======================================
# === 文件 5: hirl_sfc_env.py
# =======================================
# (已移除 'from expert_msfce import ...' 因为类已在上方定义)

class SFC_HIRL_Env(gym.Env):
    """
    分层SFC环境。
    这个环境将取代 DynamicSimulator，并允许增量式(Step-wise)执行。
    """

    def __init__(self, input_dir: Path, topo: np.ndarray, dc_nodes: List[int], capacities: Dict):
        super(SFC_HIRL_Env, self).__init__()

        self.expert = MSFCE_Solver(input_dir / "US_Backbone_path.mat", topo, dc_nodes, capacities)

        self.T = 400  # 总时间步
        self.n, self.L, self.K_vnf = self.expert.node_num, self.expert.link_num, self.expert.type_num
        self.K_path = self.expert.k_path_count

        # 资源状态 (来自 DynamicSimulator)
        self.B_cap = capacities['bandwidth']
        self.C_cap = capacities['cpu']
        self.M_cap = capacities['memory']

        self.B = np.full(self.L, self.B_cap)
        self.C = np.full(self.n, self.C_cap)
        self.M = np.full(self.n, self.M_cap)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.link_ref_count = np.zeros(self.L, dtype=int)

        # 加载数据 (来自 DynamicSimulator)
        try:
            reqs = sio.loadmat(input_dir / "sorted_requests.mat")['sorted_requests']
            self.requests = [parse_mat_request(r) for r in reqs]
            self.req_map = {r['id']: r for r in self.requests}
            events_mat = sio.loadmat(input_dir / "event_list.mat")['event_list']
            self.events = []
            for t_idx in range(events_mat.shape[0]):
                self.events.append({
                    'arrive': events_mat[t_idx, 0]['arrive_event'].flatten().astype(int),
                    'leave': events_mat[t_idx, 0]['leave_event'].flatten().astype(int)
                })
        except FileNotFoundError:
            print(f"❌ 致命错误: 找不到 {input_dir} 下的 .mat 数据文件")
            raise
        except Exception as e:
            print(f"❌ 致命错误: 加载 .mat 文件失败: {e}")
            raise

        # HIRL 状态
        self.t = 0
        self.current_request: Optional[Dict] = None
        self.unadded_dest_indices: Set[int] = set()
        self.current_tree: Optional[Dict] = None
        self.nodes_on_tree: Set[int] = set()
        self.served_requests = []  # (req, plan)

        # ----------------------------------------------------
        # ✅ 修复 #3: 新增: 稳定的路径索引管理
        # ----------------------------------------------------
        self.tree_path_list: List[List[int]] = []  # 有序路径列表
        self.tree_path_to_idx: Dict[tuple, int] = {}  # 路径 -> 索引映射

        # 定义状态和动作空间
        # 状态 = (CPU 负载, Mem 负载, BW 负载, HVT 状态) + (请求信息)
        state_size = self.n + self.n + self.L + (self.n * self.K_vnf)
        req_size = 10  # 简化的请求特征向量
        self.STATE_VECTOR_SIZE = state_size + req_size

        # 高层动作 = 选择哪个目的地 (假设最多10个目的地)
        self.NB_HIGH_LEVEL_GOALS = 10

        # 低层动作 = (连接到哪条路径 i, 使用哪条路径 k)
        self.MAX_PATHS_IN_TREE = 10  # 假设一棵树最多有10条路径
        self.NB_LOW_LEVEL_ACTIONS = self.MAX_PATHS_IN_TREE * self.K_path

        self.action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32)

    def _get_network_state_dict(self) -> Dict:
        """获取专家求解器所需格式的当前网络状态"""
        return {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }

    # ----------------------------------------------------
    # ✅ 修复 #2: 替换为您的新 _get_flat_state 方法
    # ----------------------------------------------------
    def _get_flat_state(self) -> np.ndarray:
        """
        将当前网络状态和请求扁平化为单个向量

        状态向量结构:
        [CPU负载(n维) | 内存负载(n维) | 带宽负载(L维) | HVT状态(n*K维) | 请求特征(固定维)]
        """
        net_state_dict = self._get_network_state_dict()

        # 1. 归一化网络资源状态 (当前剩余/总容量)
        cpu_norm = net_state_dict['cpu'] / self.C_cap  # (n,)
        mem_norm = net_state_dict['mem'] / self.M_cap  # (n,)
        bw_norm = net_state_dict['bw'] / self.B_cap  # (L,)

        # 2. HVT状态归一化 (假设最大引用计数为10)
        hvt_norm = np.clip(net_state_dict['hvt'].flatten() / 10.0, 0, 1)  # (n*K,)

        # 3. 编码当前请求
        req_vec = np.zeros(10)
        if self.current_request:
            # 特征 0: 带宽需求 (归一化)
            req_vec[0] = self.current_request['bw_origin'] / self.B_cap

            # 特征 1-2: CPU和内存需求的平均值 (归一化)
            if self.current_request['cpu_origin']:
                req_vec[1] = np.mean(self.current_request['cpu_origin']) / self.C_cap
            if self.current_request['memory_origin']:
                req_vec[2] = np.mean(self.current_request['memory_origin']) / self.M_cap

            # 特征 3: VNF链长度 (归一化到最大值8)
            req_vec[3] = len(self.current_request['vnf']) / 8.0

            # 特征 4: 目的节点数量 (归一化到最大值10)
            req_vec[4] = len(self.current_request['dest']) / 10.0

            # 特征 5: 已完成的目的节点比例
            if len(self.current_request['dest']) > 0:
                completed = len(self.current_request['dest']) - len(self.unadded_dest_indices)
                req_vec[5] = completed / len(self.current_request['dest'])

            # 特征 6: 源节点是否为DC (0或1)
            req_vec[6] = 1.0 if self.current_request['source'] in self.expert.DC else 0.0

            # 特征 7: 当前树的规模 (节点数 / 总节点数)
            if self.nodes_on_tree:
                req_vec[7] = len(self.nodes_on_tree) / self.n

            # 特征 8: 当前树使用的链路数 / 总链路数
            if self.current_tree:
                req_vec[8] = np.sum(self.current_tree['tree'] > 0) / self.L

            # 特征 9: 剩余未连接目的节点数 / 总目的节点数
            req_vec[9] = len(self.unadded_dest_indices) / max(1, len(self.current_request['dest']))

        # 4. 拼接所有部分
        flat_state = np.concatenate([
            cpu_norm,  # n维
            mem_norm,  # n维
            bw_norm,  # L维
            hvt_norm  # n*K维
        ])

        # 5. 组合为最终状态向量
        final_state = np.zeros(self.STATE_VECTOR_SIZE)

        # 网络状态部分
        net_state_len = len(flat_state)
        if net_state_len <= self.STATE_VECTOR_SIZE - 10:
            final_state[:net_state_len] = flat_state
        else:
            # 如果超长，截断 (优先保留CPU、Mem、BW，可能截断部分HVT)
            final_state[:-10] = flat_state[:self.STATE_VECTOR_SIZE - 10]

        # 请求特征部分 (最后10维)
        final_state[-10:] = req_vec

        return final_state.astype(np.float32)

    def _handle_leave_events(self, t: int):
        """(来自 DynamicSimulator) 处理离开事件"""
        if t >= len(self.events):
            return
        leave_ids = self.events[t]['leave']
        if leave_ids.size == 0:
            return

        leave_set = set(leave_ids)
        remaining_reqs = []

        for req, tree in self.served_requests:
            if req['id'] in leave_set:
                # 释放带宽
                for link_idx in np.where(tree['tree'] > 0)[0]:
                    if self.link_ref_count[link_idx] > 0:
                        self.link_ref_count[link_idx] -= 1
                        if self.link_ref_count[link_idx] == 0:
                            self.B[link_idx] += req['bw_origin']
                # 释放 VNF
                for node, vnf_t in np.argwhere(tree['hvt'] > 0):
                    if self.hvt_all[node, vnf_t] > 0:
                        self.hvt_all[node, vnf_t] -= 1
                        if self.hvt_all[node, vnf_t] == 0:
                            try:
                                j = req['vnf'].index(vnf_t + 1)
                                self.C[node] += req['cpu_origin'][j]
                                self.M[node] += req['memory_origin'][j]
                            except ValueError:
                                pass
            else:
                remaining_reqs.append((req, tree))
        self.served_requests = remaining_reqs

    def reset_request(self) -> Tuple[Optional[Dict], np.ndarray]:
        """
        重置环境以处理一个新请求。
        返回 (请求字典, 初始高层状态)
        """
        # 1. 推进时间，直到找到一个新请求
        self.current_request = None
        while self.current_request is None and self.t < self.T:
            # 处理上一时间步的离开
            if self.t > 0:
                self._handle_leave_events(self.t - 1)

            if self.t >= len(self.events):
                self.t += 1
                continue

            arrive_ids = self.events[self.t]['arrive']
            self.t += 1

            if arrive_ids.size > 0:
                req_id = arrive_ids[0]
                if req_id in self.req_map:
                    self.current_request = self.req_map[req_id]

        if self.current_request is None:
            return None, self._get_flat_state()

            # 2. 初始化 HIRL 状态
        self.unadded_dest_indices = set(range(len(self.current_request['dest'])))
        self.current_tree = {
            'id': self.current_request['id'],
            'tree': np.zeros(self.L),
            'hvt': np.zeros((self.n, self.K_vnf)),
            'paths_map': {}
        }
        self.nodes_on_tree = set([self.current_request['source']])

        # ----------------------------------------------------
        # ✅ 修复 #3: 重置路径管理
        # ----------------------------------------------------
        self.tree_path_list = []
        self.tree_path_to_idx = {}

        return self.current_request, self._get_flat_state()

    def get_expert_high_level_goal(self, state_vec: np.ndarray) -> int:
        """
        (专家预言机) 查询高层专家：下一步应该连接哪个目的地?
        返回: 目的节点在**原始dest数组**中的索引 (0-9)
        """
        if not self.current_request or not self.unadded_dest_indices:
            return 0

        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request

        if not self.current_tree['paths_map']:
            # 阶段1: 寻找树干 (S->d)
            tree_set = []
            for d_idx in self.unadded_dest_indices:
                best_eval = -1
                for k in range(1, self.K_path + 1):
                    eval_val, paths, tree, hvt_new, feasible, dest, cost = \
                        self.expert._calc_eval(self.current_request, d_idx, k, network_state)
                    if feasible and eval_val > best_eval:
                        best_eval = eval_val
                tree_set.append((d_idx, best_eval))

            if not tree_set or max(tree_set, key=lambda item: item[1])[1] <= 0:
                return list(self.unadded_dest_indices)[0]

            best_d_idx, _ = max(tree_set, key=lambda item: item[1])
            return best_d_idx

        else:
            # 阶段2: 寻找最佳分支 (Tree->d)
            best_eval, best_d = -1, -1
            for d_idx in self.unadded_dest_indices:
                for conn_path in self.current_tree['paths_map'].values():
                    _, m, _, _ = self.expert._calc_atnp(
                        self.current_tree, conn_path, d_idx, network_state, self.nodes_on_tree
                    )
                    if m > best_eval:
                        best_eval, best_d = m, d_idx
            return best_d if best_d != -1 else list(self.unadded_dest_indices)[0]

    # ----------------------------------------------------
    # ✅ 修复 #3: 新增辅助方法
    # ----------------------------------------------------
    def _add_path_to_tree(self, path: List[int]):
        """添加路径到树的管理结构"""
        path_tuple = tuple(path)
        if path_tuple not in self.tree_path_to_idx:
            idx = len(self.tree_path_list)
            if idx < self.MAX_PATHS_IN_TREE:  # 防止列表超出动作空间
                self.tree_path_list.append(path)
                self.tree_path_to_idx[path_tuple] = idx
                return idx
        return self.tree_path_to_idx.get(path_tuple, 0)  # 返回现有索引

    # ----------------------------------------------------
    # ✅ 修复 #3: 替换 _get_path_for_i_idx
    # ----------------------------------------------------
    def _get_path_for_i_idx(self, i_idx: int) -> List[int]:
        """✅ 修复: 根据 i_idx 获取树上的特定连接路径"""
        if not self.current_tree['paths_map']:
            # 对于 S->d 阶段, i_idx 总是0，返回源节点
            return [self.current_request['source']]

        # ✅ 使用稳定的路径列表
        if not self.tree_path_list:
            # 如果还没初始化，从 paths_map 构建
            self.tree_path_list = list(self.current_tree['paths_map'].values())
            self.tree_path_to_idx = {
                tuple(path): idx for idx, path in enumerate(self.tree_path_list)
            }

        # 安全地获取路径
        if i_idx < len(self.tree_path_list):
            return self.tree_path_list[i_idx]
        else:
            # 如果索引超出范围，返回第一条路径（兜底）
            return self.tree_path_list[0] if self.tree_path_list else [self.current_request['source']]

    # ----------------------------------------------------
    # ✅ 修复 #3: 替换 get_valid_low_level_actions
    # ----------------------------------------------------
    def get_valid_low_level_actions(self) -> List[int]:
        """✅ 修复: 返回当前状态下有效的低层动作ID列表"""
        valid_actions = []

        if not self.current_tree['paths_map']:
            # S->d 阶段: 只有 (i=0, k=0-4) 有效
            for k in range(self.K_path):
                valid_actions.append(0 * self.K_path + k)
        else:
            # Tree->d 阶段: 遍历所有树上的路径
            num_paths = len(self.tree_path_list) if self.tree_path_list else len(self.current_tree['paths_map'])

            for i in range(num_paths):
                for k in range(self.K_path):
                    action_id = i * self.K_path + k
                    # 确保动作ID不超过最大值
                    if action_id < self.NB_LOW_LEVEL_ACTIONS:
                        valid_actions.append(action_id)

        # 确保总有至少一个动作可选
        if not valid_actions:
            return [0]

        return valid_actions

    def _decode_low_level_action(self, action: int) -> Tuple[int, int]:
        """将扁平化的低层动作ID (0-N) 解码为 (i_idx, k_idx)"""
        k_idx = action % self.K_path
        i_idx = action // self.K_path
        # 限制 i_idx 在当前树的路径数内 (或最大动作空间)
        num_paths = max(1, len(self.tree_path_list))
        i_idx = i_idx % min(num_paths, self.MAX_PATHS_IN_TREE)
        return i_idx, k_idx

    def step_low_level(self, goal_dest_idx: int, low_level_action: int) -> \
            Tuple[np.ndarray, float, bool, bool]:
        """
        (增量式执行) 执行一个低层动作。
        返回: (next_state, cost, sub_task_done, request_done)
        """
        if self.current_request is None or goal_dest_idx not in self.unadded_dest_indices:
            # 目标无效或已完成
            return self._get_flat_state(), 0.0, True, not (self.unadded_dest_indices or self.current_request)

        i_idx, k_idx = self._decode_low_level_action(low_level_action)
        k = k_idx + 1
        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request

        plan = None
        cost = 0.0
        feasible = False

        if not self.current_tree['paths_map']:
            # 阶段1: 尝试 S -> d (i_idx 必须为 0)
            eval_val, paths, tree, hvt, feasible, _, cost = self.expert._calc_eval(
                self.current_request, goal_dest_idx, k, network_state
            )
            if feasible:
                plan = {'tree': tree, 'hvt': hvt, 'new_path_full': paths, 'feasible': True}
        else:
            # 阶段2: 尝试 Tree -> d
            conn_path = self._get_path_for_i_idx(i_idx)
            plan, eval_val, _, cost = self.expert._calc_atnp(
                self.current_tree, conn_path, goal_dest_idx, network_state, self.nodes_on_tree
            )
            if plan['feasible']:
                feasible = True

        if feasible:
            # 成功！应用资源
            self._apply_deployment(self.current_request, plan)
            self.unadded_dest_indices.remove(goal_dest_idx)
            dest_node = self.current_request['dest'][goal_dest_idx]

            # ----------------------------------------------------
            # ✅ 修复 #3: 更新路径管理
            # ----------------------------------------------------
            new_path = plan['new_path_full']
            self._add_path_to_tree(new_path)  # 添加到稳定列表
            self.current_tree['paths_map'][dest_node] = new_path
            self.nodes_on_tree.update(new_path)

            sub_task_done = True
            if not self.unadded_dest_indices:
                # 整个请求完成了
                self.served_requests.append((self.current_request, self.current_tree))
        else:
            # 失败！
            cost = 10.0  # 惩罚
            sub_task_done = True  # 强制结束这个子任务

        request_done = not self.unadded_dest_indices
        return self._get_flat_state(), cost, sub_task_done, request_done

    def _apply_deployment(self, request: Dict, plan: Dict):
        """(来自 DynamicSimulator) 应用部署方案 (占用资源)"""
        tree_branch = plan['tree']
        hvt_branch = plan['hvt']

        # 合并到主树
        self.current_tree['tree'] = np.logical_or(self.current_tree['tree'], tree_branch).astype(float)

        # 占用带宽
        for link_idx in np.where(tree_branch > 0)[0]:
            if self.link_ref_count[link_idx] == 0:
                self.B[link_idx] -= request['bw_origin']
            self.link_ref_count[link_idx] += 1

        # 占用 VNF
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            if self.hvt_all[node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    self.C[node] -= request['cpu_origin'][j]
                    self.M[node] -= request['memory_origin'][j]
                except ValueError:
                    pass
            self.hvt_all[node, vnf_t] += 1

        # HVT 合并必须在资源占用后
        self.current_tree['hvt'] = np.maximum(self.current_tree['hvt'], hvt_branch)


# =======================================
# === 文件 6: hirl_sfc_agent.py
# =======================================
# (已移除 'from hirl_utils import ...' 和 'from hirl_sfc_models import ...'
#  因为类已在上方定义)

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


# =======================================
# === 文件 7: train_hirl_sfc.py
# =======================================
# (已移除本地导入，因为所有类和 'H' 类已在上方定义)

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

    # --- 4. 阶段 1: 模仿学习 (预训练) ---
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

        # 遍历专家轨迹的所有步骤
        for step_idx, (exp_goal, exp_action_tuple, exp_cost) in enumerate(expert_traj):
            if exp_goal not in env.unadded_dest_indices:
                continue

            exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

            # 在环境中执行
            next_high_level_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

            # 存储
            reward = low_level_agent.criticize(sub_task_completed=True, cost=cost, request_failed=False)
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
            t += 1

            if t % 1000 == 0:
                print(f"预训练... {t}/{H.PRE_TRAIN_STEPS}")

            if request_done:
                break

        current_request, high_level_state = env.reset_request()

    print("--- 4. 阶段 2: 混合 IL/RL 训练 ---")
    low_level_agent.randomPlay = False
    stepCount = 0
    episodeCount = 0

    # ---
    # ✅ 步骤 2: 在主循环前初始化跟踪列表
    # ---
    tracking_data = {
        'episode': [],
        'reward': [],
        'acceptance_rate': [],
        'avg_cpu_util': [],
        'avg_mem_util': [],
        'avg_bw_util': []
    }
    total_requests_arrived = 0
    total_requests_served = 0

    while episodeCount < H.EPISODE_LIMIT and stepCount < H.STEPS_LIMIT and env.t < env.T:

        current_request, high_level_state = env.reset_request()
        if current_request is None:
            break

        total_requests_arrived += 1  # 跟踪到达的请求
        request_done = False
        episode_reward = 0
        episode_steps = 0

        while not request_done:
            # --- A. 高层决策 (元控制器) ---
            high_level_state_v = np.reshape(high_level_state, (1, -1))
            goal_probs = metacontroller.predict(high_level_state_v)
            goal = metacontroller.sample(goal_probs)

            true_goal = env.get_expert_high_level_goal(high_level_state_v)
            metacontroller.collect(high_level_state, true_goal)

            if goal != true_goal:
                print(f"⚠️ 高层错误 (代理选择 {goal}, 专家选择 {true_goal}), 丢弃轨迹")
                request_done = True
                continue

            if goal not in env.unadded_dest_indices:
                goal = true_goal
                if goal >= NB_GOALS or goal not in env.unadded_dest_indices:
                    request_done = True
                    continue

            goal_one_hot = np.reshape(to_categorical(goal, num_classes=NB_GOALS), (1, -1))

            # --- B. 低层执行 (代理) ---
            sub_task_done = False
            low_level_state = high_level_state

            while not sub_task_done:
                low_level_state_v = np.reshape(low_level_state, (1, -1))

                valid_actions = env.get_valid_low_level_actions()
                if not valid_actions:
                    action = 0
                else:
                    action = low_level_agent.selectMove(low_level_state_v, goal_one_hot, valid_actions)

                # ✅ 修复: 调用调试工具
                validate_action_space(env, action)

                # B2. 环境执行
                next_low_level_state, cost, sub_task_done, request_done = env.step_low_level(goal, action)

                # B3. 计算内部奖励 (RL)
                reward = low_level_agent.criticize(sub_task_done, cost, request_failed=(not sub_task_done))
                episode_reward += reward

                # B4. 存储经验
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
                    break

            high_level_state = low_level_state

        # ---
        # ✅ 步骤 3: 在回合结束时 (print 之前) 收集数据
        # ---

        # 1. 检查请求是否成功 (即所有目的地都已连接)
        # (注意: 如果高层出错, request_done=True 但 unadded_dest_indices 可能还有)
        # 我们只统计真正 *完成* 的请求
        if not env.unadded_dest_indices:
            total_requests_served += 1

        # 2. 计算当前统计数据
        # 业务请求接受率
        if total_requests_arrived > 0:
            current_acceptance_rate = (total_requests_served / total_requests_arrived) * 100.0
        else:
            current_acceptance_rate = 0.0

        # 节点资源消耗 (利用率)
        avg_cpu_util = (1.0 - np.mean(env.C) / env.C_cap) * 100.0
        avg_mem_util = (1.0 - np.mean(env.M) / env.M_cap) * 100.0
        # 带宽消耗 (利用率)
        avg_bw_util = (1.0 - np.mean(env.B) / env.B_cap) * 100.0

        # 3. 存储所有数据
        tracking_data['episode'].append(episodeCount)
        tracking_data['reward'].append(episode_reward)  # 奖励趋势
        tracking_data['acceptance_rate'].append(current_acceptance_rate)
        tracking_data['avg_cpu_util'].append(avg_cpu_util)
        tracking_data['avg_mem_util'].append(avg_mem_util)
        tracking_data['avg_bw_util'].append(avg_bw_util)

        print(f"--- 回合 {episodeCount} (T={env.t}) ---")
        print(f"总步数: {stepCount}, Epsilon: {low_level_agent.controllerEpsilon:.4f}")
        print(f"回合奖励: {episode_reward:.3f}, 回合步数: {episode_steps}")
        print(f"当前接受率: {current_acceptance_rate:.2f}% ({total_requests_served}/{total_requests_arrived})")  # 实时打印
        print(f"当前资源利用率 CPU: {avg_cpu_util:.2f}%, MEM: {avg_mem_util:.2f}%, BW: {avg_bw_util:.2f}%")  # 实时打印

        # ✅ 修复: 调用调试工具
        # visualize_state(env) # 我们将其注释掉，因为它在每个回合都打印，信息过多

        episodeCount += 1

        if episodeCount % 50 == 0:
            print("保存模型...")
            model_path = H.OUTPUT_DIR / f"sfc_hirl_model_ep{episodeCount}"
            hdqn_net.saveWeight(str(model_path))

    print("--- 5. 训练完成 ---")
    hdqn_net.saveWeight(str(H.OUTPUT_DIR / "sfc_hirl_model_final"))
    print("最终模型已保存。")

    # ---
    # ✅ 步骤 4: 在 main() 函数末尾添加绘图和保存
    # ---
    print("--- 6. 生成分析图表 ---")

    # 1. 保存原始数据到 CSV
    df = pd.DataFrame(tracking_data)
    df_path = H.OUTPUT_DIR / "training_metrics.csv"
    try:
        df.to_csv(df_path, index=False)
        print(f"训练指标已保存到 {df_path}")
    except Exception as e:
        print(f"保存 CSV 失败: {e}")

    # 2. 创建平滑函数 (用于更清晰的图表)
    def smooth_rewards(values, window_size=100):
        if len(values) == 0:
            return []
        if window_size > len(values):
            window_size = len(values)
        if window_size == 0:
            return values
        return pd.Series(values).rolling(window_size, min_periods=1).mean()

    # 3. 绘制图表 (增加检查，防止数据为空)
    try:
        if not df.empty:
            # 绘制奖励趋势
            plt.figure(figsize=(12, 8))
            plt.plot(df['episode'], df['reward'], label='原始每回合奖励', alpha=0.3)
            plt.plot(df['episode'], smooth_rewards(df['reward'], 100), label='平滑奖励 (窗口=100)', color='red')
            plt.title('奖励趋势 (Reward Trend)')
            plt.xlabel('回合 (Episode)')
            plt.ylabel('奖励 (Reward)')
            plt.legend()
            plt.grid(True)
            reward_path = H.OUTPUT_DIR / "reward_trend.png"
            plt.savefig(reward_path)
            print(f"奖励趋势图已保存到 {reward_path}")
            plt.close()

            # 绘制业务请求接受率
            plt.figure(figsize=(12, 8))
            plt.plot(df['episode'], df['acceptance_rate'])
            plt.title('请求接受率 (Acceptance Rate)')
            plt.xlabel('回合 (Episode)')
            plt.ylabel('接受率 (%)')
            plt.ylim(0, 105)  # 锁定 Y 轴在 0-100%
            plt.grid(True)
            acceptance_path = H.OUTPUT_DIR / "acceptance_rate.png"
            plt.savefig(acceptance_path)
            print(f"接受率图已保存到 {acceptance_path}")
            plt.close()

            # 绘制资源消耗
            plt.figure(figsize=(12, 8))
            plt.plot(df['episode'], smooth_rewards(df['avg_cpu_util'], 100), label='平均CPU利用率')
            plt.plot(df['episode'], smooth_rewards(df['avg_mem_util'], 100), label='平均内存利用率')
            plt.plot(df['episode'], smooth_rewards(df['avg_bw_util'], 100), label='平均带宽利用率')
            plt.title('资源利用率 (平滑窗口=100)')
            plt.xlabel('回合 (Episode)')
            plt.ylabel('利用率 (%)')
            plt.legend()
            plt.grid(True)
            resource_path = H.OUTPUT_DIR / "resource_utilization.png"
            plt.savefig(resource_path)
            print(f"资源利用率图已保存到 {resource_path}")
            plt.close()
        else:
            print("⚠️ 警告: 没有收集到训练数据 (tracking_data 为空)，跳过绘图。")

    except Exception as e:
        print(f"绘图失败: {e}")
        print("请确保已安装 matplotlib 和 pandas: pip install matplotlib pandas")

    print("--- 分析完成 ---")


# ============================================
# ✅ 修复: 添加调试和可视化工具
# ============================================

def visualize_state(env):
    """可视化当前环境状态（调试用）"""
    print(f"\n=== 当前状态 (T={env.t}) ===")
    if env.current_request:
        print(f"请求 ID: {env.current_request['id']}")
        print(f"源: {env.current_request['source']}, 目的: {env.current_request['dest']}")
        print(f"未完成目的: {env.unadded_dest_indices}")
        print(f"树上节点数: {len(env.nodes_on_tree)}")
        print(f"树上路径数: {len(env.current_tree['paths_map']) if env.current_tree else 0}")
    print(f"平均CPU利用率: {1 - np.mean(env.C / env.C_cap):.2%}")
    print(f"平均带宽利用率: {1 - np.mean(env.B / env.B_cap):.2%}")
    print(f"已服务请求数: {len(env.served_requests)}")
    print("=" * 40)


def validate_action_space(env, action):
    """验证动作是否有效（调试用）"""
    valid_actions = env.get_valid_low_level_actions()
    if action not in valid_actions:
        print(f"⚠️ 警告: 动作 {action} 不在有效动作集 {valid_actions} 中")
        return False
    return True


if __name__ == "__main__":
    main()