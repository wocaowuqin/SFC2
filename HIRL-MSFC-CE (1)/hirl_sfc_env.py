#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_env.py - FIXED & INTEGRATED VERSION (AttributeError Fix)
from reward_critic_enhanced import RewardCritic
import os
import logging
import random
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import pickle

import gym
from gym import spaces

from failure_visualizer import FailureVisualizer

from vnf_metrics_logger import VNFMetricsLogger  # [新增] 导入 Logger
# [新增] 导入备份系统
from sfc_backup_system import BackupPolicy

# ----------------------------------------------------
# Import Expert Modules
# ----------------------------------------------------
try:
    from expert_msfce import MSFCE_Solver, parse_mat_request
except ImportError:
    class MSFCE_Solver:
        pass

    def parse_mat_request(x):
        return x

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
# SFC_HIRL_Env (Fixed & Optimized & Integrated)
# ----------------------------------------------------
class SFC_HIRL_Env(gym.Env):
    """
    分层 SFC 环境（集成 BackupPolicy 版）：
    ✅ 集成 sfc_backup_system 实现高鲁棒性兜底
    ✅ 修复资源状态类型兼容性
    ✅ 修复 AttributeError: set_tree -> set_current_tree
    """

    def __init__(self, input_dir: Path, topo: np.ndarray, dc_nodes: List[int], capacities: Dict):
        super(SFC_HIRL_Env, self).__init__()

        # =========================================================
        # 1. 基础配置与路径 (Config & Paths)
        # =========================================================
        # [新增] 配置你的输出路径 (使用 raw string 防止转义)
        self.failure_output_dir = r"E:\pycharmworkspace\SFC-master\HIRL-MSFC-CE (1)\out_failue"

        # 自动创建文件夹
        import os
        os.makedirs(self.failure_output_dir, exist_ok=True)

        # [新增] 渲染开关
        self.enable_render = True

        # =========================================================
        # 2. 初始化专家系统 (Expert System) - 获取网络维度
        # =========================================================
        # 必须最先初始化，因为它决定了 self.n (节点数) 和 self.L (链路数)
        self.expert = MSFCE_Solver(input_dir / "US_Backbone_path.mat", topo, dc_nodes, capacities)

        self.T = 400
        self.n, self.L, self.K_vnf = self.expert.node_num, self.expert.link_num, self.expert.type_num
        self.K_path = self.expert.k_path_count

        # =========================================================
        # 3. 资源状态初始化 (Resource State)
        # =========================================================
        # 必须在 Logger 之前初始化，因为 Logger 需要读取这些容量值
        self.B_cap = capacities['bandwidth']
        self.C_cap = capacities['cpu']
        self.M_cap = capacities['memory']

        self.B = np.full(self.L, self.B_cap, dtype=float)
        self.C = np.full(self.n, self.C_cap, dtype=float)
        self.M = np.full(self.n, self.M_cap, dtype=float)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.link_ref_count = np.zeros(self.L, dtype=int)

        # =========================================================
        # 4. 初始化 VNFMetricsLogger (Logger)
        # =========================================================
        # [新增] 计算网络总资源用于 Logger 初始化
        # 防御性判断：处理 numpy array 或 标量
        total_cpu = np.sum(self.C_cap) if isinstance(self.C_cap, (np.ndarray, list)) else self.n * self.C_cap
        total_bw = np.sum(self.B_cap) if isinstance(self.B_cap, (np.ndarray, list)) else self.L * self.B_cap
        total_mem = np.sum(self.M_cap) if isinstance(self.M_cap, (np.ndarray, list)) else self.n * self.M_cap

        network_info = {
            "total_nodes": self.n,
            "total_cpu": float(total_cpu),
            "total_bw": float(total_bw),
            "total_mem": float(total_mem)
        }
        self.metrics_logger = VNFMetricsLogger(network_info)
        # 🔥 新增: 多目标集合管理
        self.destination_set: Set[int] = set()  # 所有目标节点
        self.served_destinations: Set[int] = set()  # 已服务的目标

        # 🔥 新增: VNF共享状态
        self.vnf_sharing_map: Dict[Tuple[int, int], Set[int]] = {}
        # Key: (node_id, vnf_type), Value: set of destination indices sharing this VNF

        # 🔥 新增: 共享策略状态
        self.sharing_strategy: int = 0  # 0-3对应4种策略

        # =========================================================
        # 5. 初始化可视化器 (Visualizer)
        # =========================================================
        # [新增] 将邻接矩阵 topo 转换为边列表
        rows, cols = np.where(topo > 0)
        edges = list(zip(rows.tolist(), cols.tolist()))

        # 实例化，无坐标数据传 None 让其自动布局
        self.visualizer = FailureVisualizer(topo_edges=edges, node_positions=None)

        # =========================================================
        # 6. 统计计数器 (Statistics)
        # =========================================================
        # [新增] 详细统计计数器 (用于计算成功率)
        self.stats_req_total = 0  # 总请求数
        self.stats_req_perfect = 0  # 完全成功的请求数
        self.stats_sub_total = 0  # 总子任务数
        self.stats_sub_success = 0  # 成功的子任务数

        self.stats_total_subtasks = 0  # 兼容旧代码计数器
        self.stats_backup_activated = 0  # 备份触发次数
        self.stats_backup_succeeded = 0  # 备份成功次数

        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.total_dest_seen = 0
        self.total_dest_accepted = 0

        # =========================================================
        # 7. 策略组件初始化 (Policy Components)
        # =========================================================
        # [新增] 初始化奖励函数
        self.reward_critic = RewardCritic(training_phase=1, epoch=0, max_epochs=1200)

        # [新增] 初始化备份策略系统
        self.backup_metrics = {"activation_count": 0, "success_count": 0, "total": 0}
        self.backup_policy = BackupPolicy(
            expert=self.expert,
            n=self.n,
            L=self.L,
            K_vnf=self.K_vnf,
            dc_nodes=dc_nodes
        )
        logger.info("BackupPolicy system initialized.")

        # ✅ 修复: 从 PathDB 构建最短距离矩阵
        self._build_shortest_dist_matrix()

        # =========================================================
        # 8. 数据加载 (Data Loading)
        # =========================================================
        self._eval_cache = {}
        self.expert_randomness = 0.1

        try:
            # 加载请求数据
            req_path = input_dir / "sorted_requests.pkl"
            if not req_path.exists():
                raise FileNotFoundError(f"找不到请求文件: {req_path}")

            with open(req_path, 'rb') as f:
                self.requests = pickle.load(f)
            self.req_map = {r['id']: r for r in self.requests}

            # 加载事件数据
            event_path = input_dir / "event_list.pkl"
            if not event_path.exists():
                raise FileNotFoundError(f"找不到事件文件: {event_path}")

            with open(event_path, 'rb') as f:
                raw_events = pickle.load(f)

            self.events = []
            for evt in raw_events:
                arrive_data = evt.get('arrive_event', evt.get('arrive', []))
                leave_data = evt.get('leave_event', evt.get('leave', []))
                self.events.append({
                    'arrive': np.array(arrive_data, dtype=int).flatten(),
                    'leave': np.array(leave_data, dtype=int).flatten()
                })
            logger.info(f"成功加载数据: {len(self.requests)} 请求, {len(self.events)} 时间步")

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

        # =========================================================
        # 9. 状态与空间定义 (State & Spaces)
        # =========================================================
        # HIRL 运行状态
        self.t = 0
        self.current_request: Optional[Dict] = None
        self.unadded_dest_indices: Set[int] = set()
        self.current_tree: Optional[Dict] = None
        self.nodes_on_tree: Set[int] = set()
        self.served_requests: List[Tuple[Dict, Dict]] = []
        self.current_req_is_perfect = True  # 标记位

        # PathManager
        self.MAX_PATHS_IN_TREE = 10
        self.path_manager = PathManager(max_paths=self.MAX_PATHS_IN_TREE)

        # 动作空间配置
        self.NB_HIGH_LEVEL_GOALS = 10
        self.NB_LOW_LEVEL_ACTIONS = self.MAX_PATHS_IN_TREE * self.K_path

        # 状态空间维度
        self.dim_network = self.n + self.n + self.L + self.n * self.K_vnf
        self.dim_request = 10
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        # Gym spaces
        self.action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32)
    # [新增] 获取备份统计指标的方法
    def get_backup_metrics(self) -> Dict[str, float]:
        """计算并返回备份策略的触发率和成功率"""
        total = max(1, self.stats_total_subtasks)
        activated = self.stats_backup_activated

        # 触发率 = 备份次数 / 总任务数
        activation_rate = (activated / total) * 100.0

        # 成功率 = 备份成功次数 / 备份触发次数
        # (分母用 max(1, activated) 防止除零)
        success_rate = 0.0
        if activated > 0:
            success_rate = (self.stats_backup_succeeded / activated) * 100.0

        return {
            'activation_rate': activation_rate,
            'success_rate': success_rate
        }
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
                # 遍历树上的每条链路
                for link_idx in np.where(tree['tree'] > 0)[0]:
                    if self.link_ref_count[link_idx] <= 0:
                        self.link_ref_count[link_idx] = 0

                    if self.link_ref_count[link_idx] > 0:
                        self.link_ref_count[link_idx] -= 1

                    if self.link_ref_count[link_idx] == 0:
                        self.B[link_idx] = min(self.B_cap, self.B[link_idx] + bw_to_return)

                # 节点资源释放
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
        # 🔥 新增: 初始化多目标集合
        self.destination_set = set(self.current_request.get('dest', []))
        self.served_destinations = set()
        self.vnf_sharing_map = {}

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

        # [修改] 调用正确的 BackupPolicy 接口 (set_tree -> set_current_tree)
        if self.current_request:
            self.backup_policy.set_current_request(self.current_request)
            self.backup_policy.set_current_tree(list(self.nodes_on_tree)) # 只传节点列表，不传{}
            # -----------------------------------------------------------
            # [新增] 2. 开始记录当前请求
            # -----------------------------------------------------------
            # 提取 VNF 链名称 (示例转换，如果你的 vnf 是数字列表)
            vnf_chain_str = [f"VNF_{v}" for v in self.current_request.get('vnf', [])]

            self.metrics_logger.start_deployment(
                request_id=f"REQ_{self.current_request['id']}",
                vnf_chain=vnf_chain_str,
                destinations=self.current_request.get('dest', [])
            )
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # [新增] 2. 开始记录当前请求
        # -----------------------------------------------------------
        # [修改] 调用 logger 时传入 t=self.t
        if hasattr(self, 'metrics_logger'):
            vnf_chain_str = [f"VNF_{v}" for v in self.current_request.get('vnf', [])]
            self.metrics_logger.start_deployment(
                request_id=f"REQ_{self.current_request['id']}",
                vnf_chain=vnf_chain_str,
                destinations=self.current_request.get('dest', []),
                t=self.t  # <--- [新增] 必须加这一行，用于时间轴画图
            )

        return self.current_request, self._get_flat_state()
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

        # Fallback: 使用最短距离
        if not candidates:
            fallback_cands = []
            source_node = self.current_request['source'] - 1
            dest_nodes = self.current_request['dest']

            for d_idx in self.unadded_dest_indices:
                try:
                    target_node = dest_nodes[d_idx] - 1
                    if 0 <= source_node < self.n and 0 <= target_node < self.n:
                        dist = float(self.shortest_dist[source_node, target_node])
                    else:
                        dist = 9999.0
                except:
                    dist = 9999.0
                fallback_cands.append((d_idx, -dist))

            fallback_cands.sort(key=lambda x: x[1], reverse=True)
            candidates = fallback_cands
            logger.debug(f"Using fallback strategy for request {req_id}")

        candidates_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

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

    def step_low_level(self, goal_dest_idx: int, low_level_action):
        """
        增强版 step_low_level (All-in-One Integrated)：
        - 核心逻辑：Expert → Backup → Deployment
        - 智能奖励：RewardCritic
        - 数据埋点：自动记录每一步的资源消耗、路径、失败原因到 VNFMetricsLogger
        - 视觉诊断：仅在出现“部分失败”（Partial Failure）时自动保存可视化图像
        - 全局统计：更新完全/部分成功率计数器
        """
        self._clear_cache()

        # ---- 解码行动 ----
        if isinstance(low_level_action, tuple) and len(low_level_action) == 3:
            i_idx, k_idx, placement = low_level_action
        else:
            i_idx, k_idx = self._decode_low_level_action(int(low_level_action))
            placement = {}

        # [统计] 增加子任务计数 (用于计算部分成功率)
        self.stats_total_subtasks += 1  # 原有计数器
        self.stats_sub_total += 1  # 新增计数器 (Success Rate 计算用)

        k = k_idx + 1

        # ---- 无效请求：直接结束 ----
        if self.current_request is None or goal_dest_idx not in self.unadded_dest_indices:
            done = not (self.unadded_dest_indices and self.current_request)
            return self._get_flat_state(), 0.0, True, done

        # ---- 网络状态准备 ----
        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request

        # ---- 执行过程变量初始化 ----
        feasible = False
        plan = None
        backup_used = False
        backup_level = "primary"
        failure_reason = None

        # ===============================
        # 1) 尝试 Expert 计算
        # ===============================
        try:
            if not self.current_tree['paths_map']:
                # 直接构造第一条路径
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
            logger.warning(f"[Expert Error] goal={goal_dest_idx}, err={e}")
            feasible = False
            failure_reason = "invalid_action"

        # ===============================
        # 2) Expert 失败 → BackupPolicy
        # ===============================
        if not feasible:
            backup_used = True
            self.stats_backup_activated += 1

            # 将 numpy 数组资源转成 dict（BackupPolicy 需要）
            backup_net_state = network_state.copy()
            backup_net_state['cpu'] = {i: float(self.C[i]) for i in range(len(self.C))}
            backup_net_state['mem'] = {i: float(self.M[i]) for i in range(len(self.M))}

            self.backup_policy.set_current_tree(list(self.nodes_on_tree))
            plan = self.backup_policy.get_backup_plan(goal_dest_idx, backup_net_state)

            if plan and plan.get('feasible'):
                feasible = True
                self.stats_backup_succeeded += 1
                backup_level = plan.get("backup_type", "unknown")
            else:
                failure_reason = failure_reason or "resource_exhausted"

        # ===============================
        # 3) 部署或失败处理
        # ===============================
        if feasible and plan is not None:
            # ---- 部署成功 ----
            self._apply_deployment(self.current_request, plan)

            # [统计] 子任务成功数 +1
            self.stats_sub_success += 1

            if goal_dest_idx in self.unadded_dest_indices:
                self.unadded_dest_indices.remove(goal_dest_idx)

            # 记录路径
            dest_list = self.current_request.get('dest', [])
            dest_node = dest_list[goal_dest_idx] if goal_dest_idx < len(dest_list) else None

            new_path = plan.get('new_path_full', [])
            if new_path:
                self.path_manager.add_path(new_path)
                if dest_node is not None:
                    self.current_tree['paths_map'][dest_node] = new_path
                self.nodes_on_tree.update(new_path)

            sub_task_done = True

            # 计算基础 Cost
            path_len = len(new_path) if new_path else 1.0
            cost_val = 0.1 * (path_len / 10.0)

            # 若所有目的节点都完成
            if not self.unadded_dest_indices:
                if (self.current_request, self.current_tree) not in self.served_requests:
                    self.served_requests.append((self.current_request, self.current_tree))
                self.total_requests_accepted += 1

        else:
            # ---- 完全失败 ----
            sub_task_done = False
            cost_val = 5.0
            failure_reason = failure_reason or "routing_deadlock"

            # [统计] 标记该请求不再是“完美”的
            self.current_req_is_perfect = False

            logger.debug(f"[Plan Failed] goal={goal_dest_idx}")

            # -----------------------------
            # [可视化] 仅在部分失败时保存图像
            # -----------------------------
            # 如果树中已经有路径(has_existing_paths=True)，但当前这一步失败了，说明是Partial Failure
            has_existing_paths = len(self.current_tree.get('paths_map', {})) > 0
            if self.enable_render and has_existing_paths:
                self.render_failure(
                    failed_dest_idx=goal_dest_idx,
                    title=f"Partial Failure: Req {self.current_request['id']} (Tree Exists)"
                )

        # ===============================
        # 4) QoS检查 + progress shaping
        # ===============================
        qos_viol = self._compute_qos_violation()
        progress = self._compute_progress(goal_dest_idx)

        # 整个请求是否终止
        request_done = not self.unadded_dest_indices

        # ===========================================================
        # [修改] 5. 准备详细的 Logger 数据 (适配最终版 Logger)
        # ===========================================================

        # 1. 准备基础数据
        dests = self.current_request.get('dest', [])
        dest_node_id = dests[goal_dest_idx] if goal_dest_idx < len(dests) else -1

        # 2. 构建详细的资源使用字典 (用于 Logger 去重计算)
        vnf_placement_info = {}
        link_usage_info = {}

        step_cpu_cons = 0.0
        step_mem_cons = 0.0
        step_bw_cons = 0.0

        if feasible and plan:
            # --- 提取 VNF 放置信息 ---
            # plan['placement'] 通常是 {vnf_index: node_id}
            # 我们需要转成: {node_id: {'cpu': val, 'mem': val}}
            req_cpus = self.current_request.get('cpu_origin', [])
            req_mems = self.current_request.get('memory_origin', [])

            # 如果有 placement 信息
            if 'placement' in plan:
                for v_idx, node in plan['placement'].items():
                    # 获取该 VNF 的具体资源需求
                    c_val = req_cpus[v_idx] if v_idx < len(req_cpus) else 0.0
                    m_val = req_mems[v_idx] if v_idx < len(req_mems) else 0.0

                    # 记录到字典 (如果同一个节点放了多个VNF，累加)
                    if node not in vnf_placement_info:
                        vnf_placement_info[node] = {'cpu': 0.0, 'mem': 0.0}
                    vnf_placement_info[node]['cpu'] += float(c_val)
                    vnf_placement_info[node]['mem'] += float(m_val)

                    # 累计总消耗
                    step_cpu_cons += c_val
                    step_mem_cons += m_val

            # --- 提取链路使用信息 ---
            path_nodes = plan.get('new_path_full', [])
            req_bw = float(self.current_request.get('bw_origin', 0.0))

            if len(path_nodes) > 1:
                for u, v in zip(path_nodes[:-1], path_nodes[1:]):
                    # 确保链路 key 顺序一致，例如总是小号在前 (u, v)
                    link_key = tuple(sorted((u, v)))
                    link_usage_info[link_key] = {'bw': float(req_bw)}

                # 累计总消耗 (跳数 * 带宽)
                step_bw_cons = req_bw * (len(path_nodes) - 1)

        # 3. 组装 step_info
        step_info = {
            "destination": int(dest_node_id),
            "success": bool(feasible),
            "path": plan.get('new_path_full', []) if feasible and plan else [],
            "vnf_placement": vnf_placement_info,  # <--- [新增] 传入放置详情
            "link_usage": link_usage_info,  # <--- [新增] 传入链路详情
            "backup_used": bool(backup_used),
            "backup_level": str(backup_level),
            "failure_reason": str(failure_reason) if not feasible else None
        }

        # 4. 组装消耗概览 (兼容旧接口)
        resource_consumed = {
            "cpu": float(step_cpu_cons),
            "bw": float(step_bw_cons),
            "mem": float(step_mem_cons)
        }

        # 5. 获取网络状态
        current_net_state = {
            "available_cpu": float(np.sum(self.C)),
            "available_bw": float(np.sum(self.B)),
            "available_mem": float(np.sum(self.M)),
            # 传入容量以便计算百分比
            "node_cpu_capacity": float(self.C_cap) if not isinstance(self.C_cap, (list, np.ndarray)) else float(
                np.mean(self.C_cap)),
            "node_mem_capacity": float(self.M_cap) if not isinstance(self.M_cap, (list, np.ndarray)) else float(
                np.mean(self.M_cap)),
            "link_bw_capacity": float(self.B_cap) if not isinstance(self.B_cap, (list, np.ndarray)) else float(
                np.mean(self.B_cap))
        }

        # D. 写入 Logger
        if hasattr(self, 'metrics_logger'):
            self.metrics_logger.record_step(step_info, resource_consumed, current_net_state)
        # ===============================
        # 6) 调用 RewardCritic (核心)
        # ===============================
        reward = self.reward_critic.criticize(
            sub_task_completed=sub_task_done,
            cost=cost_val,
            request_failed=(request_done and not sub_task_done),
            progress_to_goal=progress,
            backup_used=backup_used,
            backup_level=backup_level,
            qos_violations=qos_viol,
            failure_reason=failure_reason,
            agent_action=int(low_level_action) if not isinstance(low_level_action, tuple) else -1,
            expert_action=self.expert_low_level_action(goal_dest_idx),
            state_novelty=self._state_novelty(),
            expert_confidence=1.0
        )

        # ===============================
        # [Logger] 7) 请求结束时的处理
        # ===============================
        if request_done:
            # 更新统计计数器
            self.stats_req_total += 1
            if self.current_req_is_perfect:
                self.stats_req_perfect += 1

            # 通知 Logger 结束部署记录
            if hasattr(self, 'metrics_logger'):
                self.metrics_logger.end_deployment(current_net_state)

                # (可选) 实时监控打印：每10个请求打印一次最近的成功率
                if self.stats_req_total % 10 == 0:
                    realtime_stats = self.metrics_logger.get_realtime_stats()
                    acc = realtime_stats.get('recent_full_acceptance', 0)
                    logger.info(f"[Metrics] Recent Full Acceptance: {acc:.2%}")

        return self._get_flat_state(), float(reward), sub_task_done, request_done


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

    def _shortest_distance(self, src: int, dst: int) -> float:
        """获取两节点间的最短距离"""
        if src == dst:
            return 0.0

        # 使用预计算的矩阵
        if hasattr(self, 'shortest_dist'):
            src_idx = src - 1 if src > 0 else 0
            dst_idx = dst - 1 if dst > 0 else 0
            if 0 <= src_idx < self.n and 0 <= dst_idx < self.n:
                return float(self.shortest_dist[src_idx, dst_idx])

        return 9999.0

    def _find_closest_tree_node_to_goal(self, goal_node: int) -> int:
        """找到树中离目标最近的节点"""
        if not self.nodes_on_tree:
            return self.current_request.get('source', 1)

        min_dist = float('inf')
        closest_node = list(self.nodes_on_tree)[0]

        for node in self.nodes_on_tree:
            dist = self._shortest_distance(node, goal_node)
            if dist < min_dist:
                min_dist = dist
                closest_node = node

        return closest_node
    def _compute_progress(self, goal_idx):
        """
        修复版 progress 计算

        计算当前状态距离目标的改善程度，范围 [-1,1]
        - 负数 = 离目标更远
        - 正数 = 更接近目标

        使用最短路径矩阵而不是 networkx（避免依赖问题）
        """
        try:
            if not self.current_request:
                return 0.0

            dest_list = self.current_request.get('dest', [])
            if goal_idx >= len(dest_list):
                return 0.0

            goal_node = dest_list[goal_idx]

            # 获取当前位置（树中最近添加的节点）
            if self.nodes_on_tree:
                # 找到树中离目标最近的节点
                current_node = self._find_closest_tree_node_to_goal(goal_node)
            else:
                current_node = self.current_request.get('source', 1)

            # 使用预计算的最短路径矩阵
            current_dist = self._shortest_distance(current_node, goal_node)

            # 获取上一步的距离
            prev_dist = getattr(self, "_prev_dist", current_dist)
            self._prev_dist = current_dist

            # 如果第一次调用，返回0
            if prev_dist == current_dist and not hasattr(self, "_progress_initialized"):
                self._progress_initialized = True
                return 0.0

            # 计算进度
            if prev_dist == 0:
                return 1.0 if current_dist == 0 else -1.0

            progress = (prev_dist - current_dist) / max(1, prev_dist)
            return float(np.clip(progress, -1.0, 1.0))

        except Exception as e:
            logger.debug(f"[_compute_progress] Error: {e}")
            return 0

    def _compute_qos_violation(self):
        """
        简易 QoS 违反检测：
        返回形如 {"delay":0.2, "bandwidth":0.1}
        所有值 ∈ [0,1]，表示违反比例
        """
        viol = {}

        # ---- 延迟违反 ----
        if hasattr(self, "current_delay") and hasattr(self, "delay_threshold"):
            if self.current_delay > self.delay_threshold:
                viol["delay"] = min(1.0, (self.current_delay - self.delay_threshold) / self.delay_threshold)

        # ---- 带宽违反 ----
        if hasattr(self, "current_bw_usage") and hasattr(self, "bw_threshold"):
            if self.current_bw_usage > self.bw_threshold:
                viol["bandwidth"] = min(1.0, (self.current_bw_usage - self.bw_threshold) / self.bw_threshold)

        # ---- 丢包率违反 ----
        if hasattr(self, "current_loss") and hasattr(self, "loss_threshold"):
            if self.current_loss > self.loss_threshold:
                viol["packet_loss"] = min(1.0, (self.current_loss - self.loss_threshold) / self.loss_threshold)

        return viol if viol else None

    def _compute_progress(self, goal_dest_idx):
        """
        progress ∈ [-1,1]
        负数 = 离目标更远
        正数 = 更接近目标
        """
        try:
            dest_node = self.current_request["dest"][goal_dest_idx]
            current_node = self.current_state_info.get("current_node", None)

            if current_node is None or dest_node is None:
                return 0.0

            # 使用最短路径距离衡量进度
            import networkx as nx
            d_prev = nx.shortest_path_length(self.G, self.prev_node, dest_node) if hasattr(self, "prev_node") else None
            d_now = nx.shortest_path_length(self.G, current_node, dest_node)

            self.prev_node = current_node

            if d_prev is None:
                return 0.0

            progress = (d_prev - d_now) / max(1, d_prev)
            return float(np.clip(progress, -1.0, 1.0))

        except:
            return 0.0

    def expert_low_level_action(self, goal_dest_idx):
        """
        返回专家推荐的低层动作
        若无法获得专家动作，则返回 -1（表示跳过DAgger奖励）
        """
        try:
            return self.expert_last_action  # 如果你在 expert 调用后保存
        except:
            return -1

    def _state_novelty(self):
        """
        用于判断当前状态是否“新颖”
        值越大（接近1）表示这个状态很少被访问过→鼓励探索
        """
        if not hasattr(self, "state_visit_counter"):
            self.state_visit_counter = {}

        s = tuple(self._get_flat_state().astype(int))

        if s not in self.state_visit_counter:
            self.state_visit_counter[s] = 0
        self.state_visit_counter[s] += 1

        # visit 越少 → 新颖度越高
        novelty = 1.0 / np.sqrt(self.state_visit_counter[s])
        return float(np.clip(novelty, 0.0, 1.0))

    # [新增] 辅助方法：准备数据并调用画图
    def render_failure(self, failed_dest_idx, failed_path=None, title="Failure"):
        if not self.enable_render or not self.current_request:
            return

        # 准备数据
        src = self.current_request['source']
        dests = self.current_request.get('dest', [])
        failed_node = dests[failed_dest_idx] if failed_dest_idx < len(dests) else -1

        success_paths = self.current_tree.get('paths_map', {})

        vnf_placement = {}
        if self.current_tree:
            ns, vs = np.where(self.current_tree['hvt'] > 0)
            for n, v in zip(ns, vs):
                vnf_placement[f"VNF_{v}"] = n

        # -----------------------------
        # [核心] 生成文件名并调用保存
        # -----------------------------
        # 文件名格式: fail_req{请求ID}_goal{目标索引}_node{目标节点号}.png
        req_id = self.current_request.get('id', self.total_requests_seen)
        filename = f"fail_req{req_id}_goal{failed_dest_idx}_node{failed_node}.png"

        # 拼接完整路径
        full_path = os.path.join(self.failure_output_dir, filename)

        self.visualizer.draw_failure_case(
            src=src,
            dests=dests,
            success_paths=success_paths,
            vnf_placement=vnf_placement,
            failed_dest=failed_node,
            failed_path=failed_path,
            title=title,
            save_path=full_path  # <--- 传入路径
        )

        # ========== 修改3: VNF共享检查 (新增方法) ==========
        def can_share_vnf(self, node_id: int, vnf_type: int, dest_idx: int) -> bool:
            """
            检查目标dest_idx是否可以共享节点node_id上的vnf_type实例

            Args:
                node_id: 节点ID (1-based)
                vnf_type: VNF类型 (0-based)
                dest_idx: 目标索引 (0-based)

            Returns:
                True if 可以共享
            """
            key = (node_id, vnf_type)

            # 如果该位置没有VNF实例,不能共享
            node_idx = node_id - 1
            if self.hvt_all[node_idx, vnf_type] == 0:
                return False

            # 检查资源是否充足 (简化版,实际需要更复杂的逻辑)
            # 这里假设每个VNF实例可以服务最多3个目标
            if key in self.vnf_sharing_map:
                return len(self.vnf_sharing_map[key]) < 3

            return True

        def share_vnf(self, node_id: int, vnf_type: int, dest_idx: int):
            """记录VNF共享"""
            key = (node_id, vnf_type)
            if key not in self.vnf_sharing_map:
                self.vnf_sharing_map[key] = set()
            self.vnf_sharing_map[key].add(dest_idx)

        def get_vnf_sharing_rate(self) -> float:
            """
            计算当前请求的VNF共享率

            Returns:
                共享率 ∈ [0, 1]
            """
            if not self.vnf_sharing_map:
                return 0.0

            total_vnf_instances = sum(len(dests) for dests in self.vnf_sharing_map.values())
            unique_vnf_instances = len(self.vnf_sharing_map)

            if total_vnf_instances == 0:
                return 0.0

            # 共享率 = 1 - (独立实例数 / 总需求数)
            return 1.0 - (unique_vnf_instances / total_vnf_instances)