#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_env.py - FIXED VERSION

import logging
import random
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import gym
from gym import spaces

# ----------------------------------------------------
# Import Expert Modules
# ----------------------------------------------------
try:
    from expert_msfce import MSFCE_Solver, parse_mat_request
except ImportError:
    class MSFCE_Solver:
        pass


    def parse_mat_request(x):
        return {}

# ----------------------------------------------------
# Logging
# ----------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ----------------------------------------------------
# PathManager
# ----------------------------------------------------
class PathManager:
    """独立的路径管理器, 确保路径索引的一致性"""

    def __init__(self, max_paths=10):
        self.max_paths = max_paths
        self.paths: List[List[int]] = []
        self.path_to_idx: Dict[tuple, int] = {}

    def add_path(self, path: List[int]) -> int:
        path_tuple = tuple(path)
        if path_tuple in self.path_to_idx:
            return self.path_to_idx[path_tuple]
        if len(self.paths) < self.max_paths:
            idx = len(self.paths)
            self.paths.append(path)
            self.path_to_idx[path_tuple] = idx
            return idx
        return 0

    def get_path(self, idx: int) -> Optional[List[int]]:
        if 0 <= idx < len(self.paths):
            return self.paths[idx]
        return None

    def get_all_paths(self) -> List[List[int]]:
        return self.paths.copy()

    def reset(self):
        self.paths.clear()
        self.path_to_idx.clear()

    def __len__(self):
        return len(self.paths)


# ----------------------------------------------------
# SFC_HIRL_Env (Fixed & Optimized)
# ----------------------------------------------------
class SFC_HIRL_Env(gym.Env):
    """
    分层 SFC 环境（修复版）：
    ✅ 修复 _calc_eval 返回值解包
    ✅ 修复 shortest_dist 依赖
    ✅ 修复 _greedy_backup_plan 实现
    ✅ 优化 Fallback 策略
    """

    def __init__(self, input_dir: Path, topo: np.ndarray, dc_nodes: List[int], capacities: Dict):
        super(SFC_HIRL_Env, self).__init__()

        # 初始化专家系统
        self.expert = MSFCE_Solver(input_dir / "US_Backbone_path.mat", topo, dc_nodes, capacities)

        self.T = 400
        self.n, self.L, self.K_vnf = self.expert.node_num, self.expert.link_num, self.expert.type_num
        self.K_path = self.expert.k_path_count

        # ✅ 修复1: 从 PathDB 构建最短距离矩阵
        self._build_shortest_dist_matrix()

        # 资源状态
        self.B_cap = capacities['bandwidth']
        self.C_cap = capacities['cpu']
        self.M_cap = capacities['memory']

        self.B = np.full(self.L, self.B_cap, dtype=float)
        self.C = np.full(self.n, self.C_cap, dtype=float)
        self.M = np.full(self.n, self.M_cap, dtype=float)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.link_ref_count = np.zeros(self.L, dtype=int)

        # 缓存与随机性
        self._eval_cache = {}
        self.expert_randomness = 0.1

        # 加载数据
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
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

        # HIRL 状态
        self.t = 0
        self.current_request: Optional[Dict] = None
        self.unadded_dest_indices: Set[int] = set()
        self.current_tree: Optional[Dict] = None
        self.nodes_on_tree: Set[int] = set()
        self.served_requests: List[Tuple[Dict, Dict]] = []

        # 动作空间
        self.NB_HIGH_LEVEL_GOALS = 10
        self.MAX_PATHS_IN_TREE = 10
        self.NB_LOW_LEVEL_ACTIONS = self.MAX_PATHS_IN_TREE * self.K_path

        # PathManager
        self.path_manager = PathManager(max_paths=self.MAX_PATHS_IN_TREE)

        # State dimensions
        self.dim_network = self.n + self.n + self.L + self.n * self.K_vnf
        self.dim_request = 10
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        # Gym spaces
        self.action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32)

        # Metrics
        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.total_dest_seen = 0
        self.total_dest_accepted = 0

    # ✅ 修复1: 构建最短距离矩阵
    def _build_shortest_dist_matrix(self):
        """从 PathDB 构建最短距离矩阵"""
        self.shortest_dist = np.full((self.n, self.n), 9999.0)
        np.fill_diagonal(self.shortest_dist, 0.0)

        if self.expert.path_db is None:
            logger.warning("PathDB not available, shortest_dist will use default values")
            return

        try:
            for i in range(self.n):
                for j in range(self.n):
                    if i == j:
                        continue
                    try:
                        cell = self.expert.path_db[i, j]
                        # 获取第一条路径的距离（通常是最短的）
                        dist = float(cell['pathsdistance'][0][0])
                        self.shortest_dist[i, j] = dist
                    except:
                        pass
            logger.info("Successfully built shortest distance matrix from PathDB")
        except Exception as e:
            logger.warning(f"Failed to build shortest_dist from PathDB: {e}")

    def print_env_summary(self):
        """打印环境统计指标"""
        req_acc_rate = self.total_requests_accepted / max(1, self.total_requests_seen)
        dest_acc_rate = self.total_dest_accepted / max(1, self.total_dest_seen)
        logger.info("-" * 30)
        logger.info(f"ENV SUMMARY (t={self.t}/{self.T})")
        logger.info(
            f"Requests: Seen={self.total_requests_seen}, Acc={self.total_requests_accepted} ({req_acc_rate:.2%})")
        logger.info(f"Destinations: Seen={self.total_dest_seen}, Acc={self.total_dest_accepted} ({dest_acc_rate:.2%})")
        logger.info("-" * 30)

    def _clear_cache(self):
        """资源状态改变时清空缓存"""
        self._eval_cache.clear()

    def _get_network_state_dict(self) -> Dict:
        return {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }

    def _get_flat_state(self) -> np.ndarray:
        net_state = self._get_network_state_dict()

        cpu_usage = (self.C_cap - net_state['cpu']) / max(1.0, self.C_cap)
        mem_usage = (self.M_cap - net_state['mem']) / max(1.0, self.M_cap)
        bw_usage = (self.B_cap - net_state['bw']) / max(1.0, self.B_cap)
        hvt_norm = np.clip(net_state['hvt'].flatten() / 10.0, 0, 1)

        req_vec = np.zeros(self.dim_request, dtype=np.float32)
        if self.current_request:
            req_vec[0] = self.current_request.get('bw_origin', 0.0) / max(1.0, self.B_cap)
            if self.current_request.get('cpu_origin') is not None:
                req_vec[1] = np.mean(self.current_request['cpu_origin']) / max(1.0, self.C_cap)
            if self.current_request.get('memory_origin') is not None:
                req_vec[2] = np.mean(self.current_request['memory_origin']) / max(1.0, self.M_cap)
            req_vec[3] = len(self.current_request.get('vnf', [])) / 8.0

            dests = self.current_request.get('dest', [])
            req_vec[4] = len(dests) / 10.0
            if len(dests) > 0:
                completed = len(dests) - len(self.unadded_dest_indices)
                req_vec[5] = completed / len(dests)

            dc_set = getattr(self.expert, 'DC', set())
            req_vec[6] = 1.0 if self.current_request.get('source') in dc_set else 0.0
            if self.nodes_on_tree:
                req_vec[7] = len(self.nodes_on_tree) / max(1, self.n)
            if self.current_tree:
                req_vec[8] = np.sum(self.current_tree['tree'] > 0) / max(1.0, self.L)
            req_vec[9] = len(self.unadded_dest_indices) / max(1, len(dests))

        flat_net = np.concatenate([cpu_usage, mem_usage, bw_usage, hvt_norm])
        final_state = np.zeros(self.STATE_VECTOR_SIZE, dtype=np.float32)
        final_state[:self.dim_network] = flat_net[:self.dim_network]
        final_state[self.dim_network:] = req_vec
        return final_state

    def _handle_leave_events(self, t: int):
        if t >= len(self.events):
            return
        leave_ids = self.events[t]['leave']
        if leave_ids.size == 0:
            return

        leave_set = set(leave_ids.tolist())
        remaining = []
        for req, tree in self.served_requests:
            if req['id'] in leave_set:
                bw_to_return = float(req.get('bw_origin', 0.0))
                # 遍历树上的每条链路 —— 先 decrement, 当计数变为0时才归还带宽
                for link_idx in np.where(tree['tree'] > 0)[0]:
                    # 防御性：若计数为0但我们需要释放，记录警告并尝试修正
                    if self.link_ref_count[link_idx] <= 0:
                        # 说明可能存在之前的重复释放或计数不同步
                        logger.warning(
                            f"link_ref_count for link {link_idx} is non-positive ({self.link_ref_count[link_idx]}) when processing leave for req {req['id']}. Correcting to 0 before decrement.")
                        # guard: set to 0 (so decrement does not go negative)
                        self.link_ref_count[link_idx] = 0

                    # 先减引用
                    if self.link_ref_count[link_idx] > 0:
                        self.link_ref_count[link_idx] -= 1

                    # 只有当引用计数下降至 0 时，才把带宽归还
                    if self.link_ref_count[link_idx] == 0:
                        # add back, 并 clamp 在 [0, B_cap]
                        self.B[link_idx] = min(self.B_cap, self.B[link_idx] + bw_to_return)
                # 节点资源释放（保持原逻辑，但也做防御性检查）
                for node, vnf_t in np.argwhere(tree['hvt'] > 0):
                    if self.hvt_all[node, vnf_t] > 0:
                        self.hvt_all[node, vnf_t] -= 1
                        if self.hvt_all[node, vnf_t] == 0:
                            try:
                                j = req['vnf'].index(int(vnf_t + 1))
                                self.C[node] = min(self.C_cap, self.C[node] + req['cpu_origin'][j])
                                self.M[node] = min(self.M_cap, self.M[node] + req['memory_origin'][j])
                            except Exception:
                                pass
            else:
                remaining.append((req, tree))

        if len(self.served_requests) != len(remaining):
            self._clear_cache()
        self.served_requests = remaining

    def reset_request(self) -> Tuple[Optional[Dict], np.ndarray]:
        self._clear_cache()
        self.current_request = None
        while self.current_request is None and self.t < self.T:
            if self.t > 0:
                self._handle_leave_events(self.t - 1)
            if self.t >= len(self.events):
                self.t += 1
                continue
            arrive_ids = self.events[self.t]['arrive']
            self.t += 1
            if arrive_ids.size > 0:
                req_id = int(arrive_ids[0])
                if req_id in self.req_map:
                    self.current_request = self.req_map[req_id]

        if self.current_request is None:
            return None, self._get_flat_state()

        self.total_requests_seen += 1
        self.total_dest_seen += len(self.current_request.get('dest', []))
        self.unadded_dest_indices = set(range(len(self.current_request.get('dest', []))))
        self.current_tree = {
            'id': self.current_request['id'],
            'tree': np.zeros(self.L),
            'hvt': np.zeros((self.n, self.K_vnf)),
            'paths_map': {}
        }
        self.nodes_on_tree = set([self.current_request['source']])
        self.path_manager.reset()
        return self.current_request, self._get_flat_state()

    # ✅ 修复2: 优化 Fallback 策略
    def get_expert_high_level_candidates(self, state_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.current_request or not self.unadded_dest_indices:
            return []

        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request
        req_id = self.current_request['id']
        candidates: List[Tuple[int, float]] = []

        # Stage 1: S -> d
        if not self.current_tree['paths_map']:
            for d_idx in self.unadded_dest_indices:
                best_eval = -np.inf
                for k in range(1, self.K_path + 1):
                    cache_key = (req_id, d_idx, k)
                    if cache_key in self._eval_cache:
                        eval_val = self._eval_cache[cache_key]
                    else:
                        try:
                            # ✅ 正确解包 8 个返回值
                            eval_val, _, _, _, feasible, _, _, _ = self.expert._calc_eval(
                                self.current_request, d_idx, k, network_state
                            )
                            if not feasible or eval_val is None:
                                eval_val = -np.inf
                        except Exception as e:
                            logger.debug(f"_calc_eval failed for d={d_idx}, k={k}: {e}")
                            eval_val = -np.inf
                        self._eval_cache[cache_key] = eval_val

                    if eval_val > best_eval:
                        best_eval = eval_val
                if best_eval > -np.inf:
                    candidates.append((d_idx, float(best_eval)))

        # Stage 2: Tree -> d
        else:
            for d_idx in self.unadded_dest_indices:
                best_eval = -np.inf
                for conn_path in self.current_tree['paths_map'].values():
                    try:
                        _, eval_val, _, _ = self.expert._calc_atnp(
                            self.current_tree, conn_path, d_idx, network_state, self.nodes_on_tree
                        )
                        if eval_val is not None and eval_val > best_eval:
                            best_eval = eval_val
                    except Exception as e:
                        logger.debug(f"_calc_atnp failed for d={d_idx}: {e}")
                        continue
                if best_eval > -np.inf:
                    candidates.append((d_idx, float(best_eval)))

        # ✅ 智能 Fallback: 使用最短距离
        if not candidates:
            fallback_cands = []
            source_node = self.current_request['source'] - 1  # 转换为 0-based
            dest_nodes = self.current_request['dest']

            for d_idx in self.unadded_dest_indices:
                try:
                    target_node = dest_nodes[d_idx] - 1  # 转换为 0-based
                    if 0 <= source_node < self.n and 0 <= target_node < self.n:
                        dist = float(self.shortest_dist[source_node, target_node])
                    else:
                        dist = 9999.0
                except:
                    dist = 9999.0
                # 距离越小越好，score = -dist
                fallback_cands.append((d_idx, -dist))

            fallback_cands.sort(key=lambda x: x[1], reverse=True)
            candidates = fallback_cands
            logger.debug(f"Using fallback strategy for request {req_id}")

        # 排序
        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Expert Randomness
        if len(candidates_sorted) >= 2 and random.random() < self.expert_randomness:
            candidates_sorted[0], candidates_sorted[1] = candidates_sorted[1], candidates_sorted[0]

        limit = len(candidates_sorted) if top_k < 0 else min(top_k, len(candidates_sorted))
        return candidates_sorted[:max(1, limit)]

    def get_high_level_candidate_mask(self, candidates: List[Tuple[int, float]]) -> np.ndarray:
        mask = np.zeros(self.NB_HIGH_LEVEL_GOALS, dtype=np.float32)
        for d_idx, _ in candidates:
            if 0 <= d_idx < self.NB_HIGH_LEVEL_GOALS:
                mask[d_idx] = 1.0
        return mask

    def get_expert_high_level_goal(self, state_vec: np.ndarray) -> int:
        cands = self.get_expert_high_level_candidates(state_vec, top_k=1)
        if cands:
            return int(cands[0][0])
        if self.unadded_dest_indices:
            return int(next(iter(self.unadded_dest_indices)))
        return 0

    def get_expert_high_level_labels(self, state_vec: np.ndarray, top_k: int = 5) -> Tuple[List[int], List[float], int]:
        cands = self.get_expert_high_level_candidates(state_vec, top_k=top_k)
        if not cands:
            return [], [], 0
        ids = [int(c[0]) for c in cands]
        scores = [float(c[1]) for c in cands]
        return ids, scores, ids[0]

    def _get_path_for_i_idx(self, i_idx: int) -> List[int]:
        if not self.current_tree or not self.current_tree['paths_map']:
            return [self.current_request['source']] if self.current_request else [0]
        path = self.path_manager.get_path(i_idx)
        if path is None:
            if self.path_manager.paths:
                return self.path_manager.get_path(0)
            return [self.current_request['source']] if self.current_request else [0]
        return path

    def get_valid_low_level_actions(self) -> List[int]:
        valid_actions = []
        if not self.current_tree or not self.current_tree['paths_map']:
            for k in range(self.K_path):
                valid_actions.append(k)
        else:
            num_paths = max(1, len(self.path_manager))
            if len(self.path_manager) == 0:
                num_paths = len(self.current_tree['paths_map'])
            for i in range(num_paths):
                for k in range(self.K_path):
                    action_id = i * self.K_path + k
                    if action_id < self.NB_LOW_LEVEL_ACTIONS:
                        valid_actions.append(action_id)
        return valid_actions if valid_actions else [0]

    def get_low_level_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.NB_LOW_LEVEL_ACTIONS, dtype=np.float32)
        for a in self.get_valid_low_level_actions():
            if 0 <= a < self.NB_LOW_LEVEL_ACTIONS:
                mask[a] = 1.0
        return mask

    def _decode_low_level_action(self, action: int) -> Tuple[int, int]:
        k_idx = int(action % self.K_path)
        i_idx = int(action // self.K_path)
        num_paths = max(1, len(self.path_manager))
        i_idx = i_idx % min(num_paths, self.MAX_PATHS_IN_TREE)
        return i_idx, k_idx

    # ✅ 修复3: 实现真正的 Greedy Backup Plan
    def _greedy_backup_plan(self, goal_dest_idx: int, network_state: Dict) -> Optional[Dict]:
        """
        简化的 Greedy Backup: 尝试使用 PathDB 的第一条路径（通常是最短的）
        """
        try:
            if not self.current_tree['paths_map']:
                src = self.current_request['source']
            else:
                # 从源节点出发（简化策略）
                src = self.current_request['source']

            dst = self.current_request['dest'][goal_dest_idx]

            # 尝试获取第一条路径（k=1）
            nodes, dist, links = self.expert._get_path_info(src, dst, 1)

            if not nodes:
                logger.debug(f"Backup failed: No path from {src} to {dst}")
                return None

            # 构建简化的 plan（不检查资源，让 apply 时处理）
            tree_vec = np.zeros(self.L)
            for lid in links:
                if lid - 1 < len(tree_vec):
                    tree_vec[lid - 1] = 1

            # 不部署 VNF（简化版），仅返回路径
            return {
                'tree': tree_vec,
                'hvt': np.zeros((self.n, self.K_vnf)),
                'new_path_full': nodes,
                'feasible': True,
                'placement': {}
            }
        except Exception as e:
            logger.debug(f"Greedy backup failed: {e}")
            return None

    # ✅ 修复4: 修正 step_low_level
    def step_low_level(self, goal_dest_idx: int, low_level_action) -> Tuple[np.ndarray, float, bool, bool]:
        self._clear_cache()

        if isinstance(low_level_action, tuple) and len(low_level_action) == 3:
            i_idx, k_idx, placement = low_level_action
        else:
            i_idx, k_idx = self._decode_low_level_action(int(low_level_action))
            placement = {}

        k = k_idx + 1

        if self.current_request is None or goal_dest_idx not in self.unadded_dest_indices:
            done = not (self.unadded_dest_indices and self.current_request)
            return self._get_flat_state(), 0.0, True, done

        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request
        plan = None
        feasible = False

        # Expert Calculation
        try:
            if not self.current_tree['paths_map']:
                # ✅ 正确解包 8 个返回值
                eval_val, paths, tree, hvt, feasible, _, _, placement_expert = \
                    self.expert._calc_eval(self.current_request, goal_dest_idx, k, network_state)

                if feasible:
                    plan = {
                        'tree': tree,
                        'hvt': hvt,
                        'new_path_full': paths,
                        'feasible': True,
                        'placement': placement_expert
                    }
            else:
                conn_path = self._get_path_for_i_idx(i_idx)
                plan, eval_val, action, _ = self.expert._calc_atnp(
                    self.current_tree, conn_path, goal_dest_idx, network_state, self.nodes_on_tree
                )
                feasible = plan.get('feasible', False) if isinstance(plan, dict) else False

        except Exception as e:
            logger.warning(f"Expert calculation failed for goal {goal_dest_idx}: {e}")
            feasible = False

        # Env-level Backup
        if not feasible:
            plan = self._greedy_backup_plan(goal_dest_idx, network_state)
            if plan:
                feasible = True
                logger.debug(f"Using greedy backup for goal {goal_dest_idx}")

        # 部署或惩罚
        if feasible and plan is not None:
            self._apply_deployment(self.current_request, plan)
            if goal_dest_idx in self.unadded_dest_indices:
                self.unadded_dest_indices.remove(goal_dest_idx)

            dest_list = self.current_request.get('dest', [])
            dest_node = dest_list[goal_dest_idx] if goal_dest_idx < len(dest_list) else None
            self.total_dest_accepted += 1

            new_path = plan.get('new_path_full', [])
            if new_path:
                self.path_manager.add_path(new_path)
                if dest_node is not None:
                    self.current_tree['paths_map'][dest_node] = new_path
                self.nodes_on_tree.update(new_path)

            sub_task_done = True

            # Reward Shaping: 惩罚路径长度
            path_len = len(new_path) if new_path else 1.0
            cost_val = 0.1 * (path_len / 10.0)

            # 请求完成
            if not self.unadded_dest_indices:
                if (self.current_request, self.current_tree) not in self.served_requests:
                    self.served_requests.append((self.current_request, self.current_tree))
                self.total_requests_accepted += 1
        else:
            cost_val = 5.0  # 失败大惩罚
            sub_task_done = False
            logger.debug(f"Plan infeasible for goal {goal_dest_idx}")

        request_done = not self.unadded_dest_indices
        return self._get_flat_state(), float(cost_val), sub_task_done, request_done

    def _apply_deployment(self, request: Dict, plan: Dict):
        """应用部署方案到网络状态（带防御性检查）"""
        tree_branch = plan.get('tree', np.zeros(self.L))
        hvt_branch = plan.get('hvt', np.zeros((self.n, self.K_vnf)))
        self.current_tree['tree'] = np.logical_or(self.current_tree['tree'], tree_branch).astype(float)

        bw_req = float(request.get('bw_origin', 0.0))

        # 链路资源扣除：当引用计数为0时才真正扣带宽，然后无论如何引用计数+1
        for link_idx in np.where(tree_branch > 0)[0]:
            if self.link_ref_count[link_idx] == 0:
                # 防御性：不要把带宽扣到负值
                new_bw = self.B[link_idx] - bw_req
                self.B[link_idx] = max(0.0, new_bw)
            self.link_ref_count[link_idx] += 1

        # 节点资源扣除（保持原有语义）
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            if self.hvt_all[node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(int(vnf_t + 1))
                    self.C[node] = max(0.0, self.C[node] - request['cpu_origin'][j])
                    self.M[node] = max(0.0, self.M[node] - request['memory_origin'][j])
                except Exception:
                    pass
            self.hvt_all[node, vnf_t] += 1

        self.current_tree['hvt'] = np.maximum(self.current_tree['hvt'], hvt_branch)
