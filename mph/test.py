#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_msfce_fused.py
# @Date    : 2025-11-10
# @Desc    : 将 MSFC-CE 专家算法与分层模仿-强化学习 (HIRL) 框架融合的单一文件实现

import numpy as np
import scipy.io as sio
import time
import random
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import namedtuple, deque

# 确保 Keras (TensorFlow) 相关的库已导入
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Input, concatenate, Lambda
    from tensorflow.keras import optimizers
    from tensorflow.keras import backend as K
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("❌ 错误: 缺少 TensorFlow (Keras)。请安装 tensorflow。")
    exit(1)

# 确保 Gym (用于环境)
try:
    import gym
    from gym import spaces
except ImportError:
    print("❌ 错误: 缺少 Gym。请安装 gym。")
    exit(1)


# ==============================================================
# ✅ 第 1 部分: HIRL 依赖库 (来自 Montezuma 项目)
# 包含: SegmentTree, LinearSchedule, PrioritizedReplayBuffer
# ==============================================================

class SegmentTree(object):
    """
    Segment Tree: 优先经验回放 (PER) 的底层数据结构。
    (代码来自: replay_buffer.py 和 segment_tree.py)
    """

    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation
        self.neutral_element = neutral_element

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.value[node]
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
        end -= 1
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
    (代码来自: schedules.py)
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
    (代码来自: replay_buffer.py)
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
        data = (args)
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

    def sample(self, batch_size, beta):
        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.reduce() / self._it_sum.reduce()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        # (s, g, a, r, s', done)
        states, goals, actions, rewards, next_states, dones = [], [], [], [], [], []

        for idx in idxes:
            data = self._storage[idx]
            s, g, a, r, ns, d = data
            states.append(s)
            goals.append(g)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

            p_sample = self._it_sum[idx] / self._it_sum.reduce()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)

        return (
            np.array(states),
            np.array(goals),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights),
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


# ==============================================================
# ✅ 第 2 部分: MSFC-CE 专家 (来自 msfce_simulator_fixed.py)
# ==============================================================

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
        self.path_db = sio.loadmat(path_db_file)['Paths']
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
        return lid - 1, self.link_map

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
        except:
            return [], 0, []

    def _get_kth_path_max_distance(self, src: int, dst: int, kpath: int) -> int:
        try:
            return int(self.path_db[src - 1, dst - 1]['pathsdistance'][kpath - 1][0])
        except:
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
            if bw[lid - 1] < request['bw_origin']:
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
            if state['bw'][lid - 1] < request['bw_origin']:
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

        # TODO: 为成本定义一个归一化权重
        # 这里的 1, 10, 10 是示例权重，需要调整
        total_cost = (bw_cost * 1) + (cpu_cost * 10) + (mem_cost * 10)
        return total_cost / 1000.0  # 返回一个归一化的成本值

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
        # 高层目标 = 哪个目的地 (best_d_idx)
        # 低层动作 = (i_idx=0 (源点), k_idx=best_k_set[best_d_idx])
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

        while unadded:
            best_eval, best_plan, best_d, best_action, best_cost = -1, None, -1, (0, 0), 0

            for d_idx in unadded:
                # 遍历所有已在树上的路径，看从哪里连接
                for conn_path in current_tree['paths_map'].values():
                    t, m, action, cost = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, network_state, nodes_on_tree
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

            # TODO: 关键 - 专家决策时"不"更新状态，但在为 HIRL 训练时，
            # 我们可能需要在这里"模拟"更新 network_state 以获得更真实的轨迹。
            # 为简单起见，目前假设
            # network_state 在一个请求中保持不变。

        return current_tree, expert_trajectory


# ==============================================================
# ✅ 第 3 部分: HIRL 强化学习环境 (SFC 版本)
# ==============================================================

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

        # HIRL 状态
        self.t = 0
        self.current_request: Optional[Dict] = None
        self.unadded_dest_indices: Set[int] = set()
        self.current_tree: Optional[Dict] = None
        self.nodes_on_tree: Set[int] = set()
        self.served_requests = []  # (req, plan)

        # 定义状态和动作空间
        # TODO: 这是一个关键且困难的定义。这里使用一个简化的占位符。
        # 状态 = (CPU 负载, Mem 负载, BW 负载, HVT 状态) + (请求信息)
        state_size = self.n + self.n + self.L + (self.n * self.K_vnf)
        req_size = 10  # 简化的请求特征向量
        self.STATE_VECTOR_SIZE = state_size + req_size

        # 高层动作 = 选择哪个目的地 (假设最多10个目的地)
        self.NB_HIGH_LEVEL_GOALS = 10

        # 低层动作 = (连接到哪个节点 i, 使用哪条路径 k)
        # 假设最多 28 个节点 * 5 条 K-path
        self.MAX_NODES_IN_TREE = 28
        self.NB_LOW_LEVEL_ACTIONS = self.MAX_NODES_IN_TREE * self.K_path

        self.action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32)

    def _get_network_state_dict(self) -> Dict:
        """获取专家求解器所需格式的当前网络状态"""
        return {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }

    def _get_flat_state(self) -> np.ndarray:
        """(TODO) 将当前网络状态和请求扁平化为单个向量"""
        # 这是一个占位符。你需要一个复杂的函数来正确编码状态。
        net_state_dict = self._get_network_state_dict()
        cpu_norm = net_state_dict['cpu'] / self.C_cap
        mem_norm = net_state_dict['mem'] / self.M_cap
        bw_norm = net_state_dict['bw'] / self.B_cap
        hvt_norm = net_state_dict['hvt'].flatten() / 5.0  # 假设最大引用计数为5

        # 编码请求 (占位符)
        req_vec = np.zeros(10)
        if self.current_request:
            req_vec[0] = self.current_request['bw_origin'] / self.B_cap
            # ... 更多请求特征

        flat_state = np.concatenate([
            cpu_norm, mem_norm, bw_norm, hvt_norm
        ])

        # 截断或填充到固定大小
        final_state = np.zeros(self.STATE_VECTOR_SIZE)
        len_to_copy = min(len(flat_state), self.STATE_VECTOR_SIZE - 10)
        final_state[:len_to_copy] = flat_state[:len_to_copy]
        final_state[-10:] = req_vec

        return final_state.astype(np.float32)

    def _handle_leave_events(self, t: int):
        """(来自 DynamicSimulator) 处理离开事件"""
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

            # 获取当前时间步的到达
            arrive_ids = self.events[self.t]['arrive']
            self.t += 1

            if arrive_ids.size > 0:
                # TODO: 目前只处理第一个到达的请求
                req_id = arrive_ids[0]
                if req_id in self.req_map:
                    self.current_request = self.req_map[req_id]

        if self.current_request is None:
            return None, self._get_flat_state()  # 仿真结束

        # 2. 初始化 HIRL 状态
        self.unadded_dest_indices = set(range(len(self.current_request['dest'])))
        self.current_tree = {
            'id': self.current_request['id'],
            'tree': np.zeros(self.L),
            'hvt': np.zeros((self.n, self.K_vnf)),
            'paths_map': {}  # (dest_node -> path_nodes)
        }
        self.nodes_on_tree = set([self.current_request['source']])

        return self.current_request, self._get_flat_state()

    def get_expert_high_level_goal(self, state_vec: np.ndarray) -> int:
        """
        (专家预言机) 查询高层专家：下一步应该连接哪个目的地？
        """
        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request

        if not self.current_tree['paths_map']:
            # 阶段1: 寻找树干 (S->d)
            tree_set = []
            for d_idx in self.unadded_dest_indices:
                best_eval = -1
                for k in range(1, self.K_path + 1):
                    eval_val, ... = self.expert._calc_eval(self.current_request, d_idx, k, network_state)
                    if feasible and eval_val > best_eval:
                        best_eval = eval_val
                tree_set.append((d_idx, best_eval))

            if not tree_set: return 0
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

    def _decode_low_level_action(self, action: int) -> Tuple[int, int]:
        """将扁平化的低层动作ID (0-N) 解码为 (i_idx, k_idx)"""
        k_idx = action % self.K_path
        i_idx = action // self.K_path
        # 确保 i_idx 在当前树的节点范围内
        i_idx = i_idx % max(1, len(self.current_tree['paths_map']))
        return i_idx, k_idx

    def _get_path_for_i_idx(self, i_idx: int) -> List[int]:
        """(TODO) 根据 i_idx 获取树上的特定连接路径"""
        if not self.current_tree['paths_map']:
            return [self.current_request['source']]  # 对于 S->d, i_idx 总是0

        # 这是一个占位符，需要一个稳定的方式将 i_idx 映射到一条路径
        path_list = list(self.current_tree['paths_map'].values())
        return path_list[i_idx % len(path_list)]

    def step_low_level(self, goal_dest_idx: int, low_level_action: int) -> \
            Tuple[np.ndarray, float, bool, bool]:
        """
        (增量式执行) 执行一个低层动作。
        返回: (next_state, cost, sub_task_done, request_done)
        """
        if self.current_request is None or goal_dest_idx not in self.unadded_dest_indices:
            # 目标无效或已完成
            return self._get_flat_state(), 0.0, True, not self.unadded_dest_indices

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
                plan = {'tree': tree, 'hvt': hvt, 'new_path_full': paths}
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
            self.current_tree['paths_map'][dest_node] = plan['new_path_full']
            self.nodes_on_tree.update(plan['new_path_full'])

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


# ==============================================================
# ✅ 第 4 部分: HIRL 神经网络模型 (SFC MLP 版本)
# ==============================================================

# (来自: hybrid_model_atari.py)
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
        self.input_shape = state_shape
        self.n_goals = n_goals
        self.replay_hist = [None] * 1000  # DAgger 聚合缓冲区
        self.ind = 0
        self.count = 0

        rmsProp = optimizers.RMSprop(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)
        self.meta_controller = self._build_model()
        self.meta_controller.compile(loss='categorical_crossentropy', optimizer=rmsProp)

    def _build_model(self):
        model = Sequential()
        model.add(Dense(256, activation='relu', input_shape=self.input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_goals, activation='softmax'))
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
        rmsProp = optimizers.RMSprop(lr=self.lr, rho=0.95, epsilon=1e-08, decay=0.0)
        model.compile(loss='mse', optimizer=rmsProp)  # 'mse' 只是占位符，实际 loss 在 Agent 中计算
        return model

    def saveWeight(self, file_prefix):
        self.controllerNet.save_weights(f"{file_prefix}_controller.h5")

    def loadWeight(self, file_prefix):
        self.controllerNet.load_weights(f"{file_prefix}_controller.h5")
        self.targetControllerNet.set_weights(self.controllerNet.get_weights())


# ==============================================================
# ✅ 第 5 部分: HIRL 代理 (SFC 版本)
# ==============================================================

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

    def selectMove(self, state, goal_one_hot):
        """低层动作选择 (Epsilon-Greedy)"""
        if self.controllerEpsilon < random.random():
            # (1, state_dim), (1, goal_dim)
            q_values = self.net.controllerNet.predict([state, goal_one_hot], verbose=0)
            return np.argmax(q_values[0])
        return random.choice(self.action_set)

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
        """存储 (s, g, a, r, s', done)"""
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
        rmsProp = optimizers.RMSprop(lr=self.net.lr, rho=0.95, epsilon=1e-08, decay=0.0)
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


# ==============================================================
# ✅ 第 6 部分: 主训练循环 (SFC HIRL 版本)
# ==============================================================

# (来自 atari)
ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])


def main():
    """主训练循环"""

    # --- 1. 配置 ---
    print("--- 1. 配置 HIRL-SFC 实验 ---")
    INPUT_DIR = Path('E:/pycharmworkspace/SFC-master/mph/mat')
    OUTPUT_DIR = Path('E:/pycharmworkspace/SFC-master/mph/out_hirl')
    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True)

    CAPACITIES = {'cpu': 2000.0, 'memory': 1100.0, 'bandwidth': 500.0}
    DC_NODES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 27, 28]
    TOPOLOGY_MATRIX = np.array([
        # ... (从 msfce_simulator_fixed.py 复制巨大的拓扑矩阵) ...
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

    # --- 2. 超参数 (Hyperparameters) ---
    # (来自: hyperparameters.py 和 DRL 经验)
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

    # --- 3. 初始化 ---
    print("--- 2. 初始化环境、专家和代理 ---")
    env = SFC_HIRL_Env(INPUT_DIR, TOPOLOGY_MATRIX, DC_NODES, CAPACITIES)

    # 获取SFC环境的特定维度
    STATE_SHAPE = env.observation_space.shape
    NB_GOALS = env.NB_HIGH_LEVEL_GOALS
    NB_ACTIONS = env.NB_LOW_LEVEL_ACTIONS

    # 高层
    metacontroller = MetaControllerNN(state_shape=STATE_SHAPE, n_goals=NB_GOALS, lr=LR)

    # 低层 (H-DQN 架构)
    hdqn_net = Hdqn_SFC(state_shape=STATE_SHAPE, n_goals=NB_GOALS, n_actions=NB_ACTIONS, lr=LR)
    low_level_agent = Agent_SFC(
        net=hdqn_net,
        n_actions=NB_ACTIONS,
        mem_cap=EXP_MEMORY,
        exploration_steps=EXPLORATION_STEPS,
        train_freq=TRAIN_FREQ,
        hard_update=HARD_UPDATE_FREQUENCY,
        n_samples=BATCH_SIZE,
        gamma=GAMMA
    )
    low_level_agent.compile()  # 构建 Keras 训练模型

    print(f"状态向量大小: {STATE_SHAPE}")
    print(f"高层目标(子任务)数量: {NB_GOALS}")
    print(f"低层动作数量: {NB_ACTIONS}")

    # --- 4. 阶段 1: 模仿学习 (预训练) ---
    print(f"--- 3. 阶段 1: 模仿学习预训练 ( {PRE_TRAIN_STEPS} 步) ---")
    stepCount = 0
    current_request, high_level_state = env.reset_request()

    for t in range(PRE_TRAIN_STEPS):
        if current_request is None:
            break  # 仿真结束

        # 1. 查询高层专家
        high_level_state_v = np.reshape(high_level_state, (1, -1))
        true_goal = env.get_expert_high_level_goal(high_level_state_v)

        # 2. 查询低层专家
        # (TODO: 需要一个 'get_expert_low_level_action' 函数)
        # 为简化，我们直接执行专家的第一步
        _, expert_traj = env.expert.solve_request_for_expert(
            current_request, env._get_network_state_dict()
        )
        if not expert_traj:
            current_request, high_level_state = env.reset_request()
            continue

        # 只使用专家轨迹的第一步进行预训练
        exp_goal, exp_action_tuple, exp_cost = expert_traj[0]
        exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

        # 3. 在环境中执行专家动作
        next_high_level_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

        # 4. 存储模仿数据
        reward = low_level_agent.criticize(sub_task_completed=True, cost=cost, request_failed=False)
        exp = ActorExperience(high_level_state, exp_goal, exp_action, reward, next_high_level_state, sub_task_done)
        low_level_agent.store(exp)

        # 5. 训练
        if t % low_level_agent.trainFreq == 0:
            low_level_agent.update(t)

        metacontroller.collect(high_level_state, exp_goal)
        if t % META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
            metacontroller.train()

        # 6. 推进
        high_level_state = next_high_level_state
        if request_done:
            current_request, high_level_state = env.reset_request()

        if t % 1000 == 0:
            print(f"预训练... {t}/{PRE_TRAIN_STEPS}")

    print("--- 4. 阶段 2: 混合 IL/RL 训练 ---")
    low_level_agent.randomPlay = False  # 结束随机播放
    stepCount = 0
    episodeCount = 0

    while episodeCount < EPISODE_LIMIT and stepCount < STEPS_LIMIT and env.t < env.T:

        current_request, high_level_state = env.reset_request()
        if current_request is None:
            break  # 仿真结束

        request_done = False
        episode_reward = 0
        episode_steps = 0

        while not request_done:
            # --- A. 高层决策 (元控制器) ---
            high_level_state_v = np.reshape(high_level_state, (1, -1))
            goal_probs = metacontroller.predict(high_level_state_v)
            goal = metacontroller.sample(goal_probs)  # 采样一个子任务

            # (DAgger) 查询专家
            true_goal = env.get_expert_high_level_goal(high_level_state_v)
            metacontroller.collect(high_level_state, true_goal)

            # (如果子任务无效，则跳过)
            if goal not in env.unadded_dest_indices:
                goal = true_goal  # 强制使用专家
                if goal not in env.unadded_dest_indices:
                    # 专家也认为完成了
                    request_done = True
                    continue

            goal_one_hot = np.reshape(to_categorical(goal, num_classes=NB_GOALS), (1, -1))

            # --- B. 低层执行 (代理) ---
            sub_task_done = False
            low_level_state = high_level_state

            while not sub_task_done:
                low_level_state_v = np.reshape(low_level_state, (1, -1))

                # B1. 低层选择动作
                action = low_level_agent.selectMove(low_level_state_v, goal_one_hot)

                # B2. 环境执行
                next_low_level_state, cost, sub_task_done, request_done = env.step_low_level(goal, action)

                # B3. 计算内部奖励 (RL)
                reward = low_level_agent.criticize(sub_task_done, cost, request_failed=(not sub_task_done))
                episode_reward += reward

                # B4. 存储经验
                exp = ActorExperience(low_level_state, goal, action, reward, next_low_level_state, sub_task_done)
                low_level_agent.store(exp)

                # B5. 训练
                if stepCount % low_level_agent.trainFreq == 0:
                    loss, avgQ, avgTD = low_level_agent.update(stepCount)
                    if stepCount % 1000 == 0:
                        print(f"Step {stepCount} | Q: {avgQ:.3f}, TD: {avgTD:.3f}, Loss: {loss:.3f}")

                if stepCount % META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                    metacontroller.train()

                low_level_agent.annealControllerEpsilon(stepCount)

                low_level_state = next_low_level_state
                stepCount += 1
                episode_steps += 1

                if request_done:
                    break  # 整个请求完成了

            high_level_state = low_level_state

        print(f"--- 回合 {episodeCount} (T={env.t}) ---")
        print(f"总步数: {stepCount}, Epsilon: {low_level_agent.controllerEpsilon:.4f}")
        print(f"回合奖励: {episode_reward:.3f}, 回合步数: {episode_steps}")
        episodeCount += 1

        if episodeCount % 50 == 0:
            print("保存模型...")
            hdqn_net.saveWeight(OUTPUT_DIR / f"sfc_hirl_model_ep{episodeCount}")

    print("--- 5. 训练完成 ---")
    hdqn_net.saveWeight(OUTPUT_DIR / "sfc_hirl_model_final")
    print("最终模型已保存。")


if __name__ == "__main__":
    main()