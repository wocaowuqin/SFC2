#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_env.py

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any

import gym
from gym import spaces

# 导入专家和解析器
from expert_msfce import MSFCE_Solver, parse_mat_request


# ----------------------------------------------------
# PathManager (unchanged)
# ----------------------------------------------------
class PathManager:
    """独立的路径管理器,确保一致性"""

    def __init__(self, max_paths=10):
        self.max_paths = max_paths
        self.paths = []  # List[List[int]]
        self.path_to_idx = {}  # Dict[tuple, int]

    def add_path(self, path: List[int]) -> int:
        """添加路径并返回索引"""
        path_tuple = tuple(path)

        # 如果已存在,返回现有索引
        if path_tuple in self.path_to_idx:
            return self.path_to_idx[path_tuple]

        # 如果未满,添加新路径
        if len(self.paths) < self.max_paths:
            idx = len(self.paths)
            self.paths.append(path)
            self.path_to_idx[path_tuple] = idx
            return idx

        # 如果已满,返回 0 (索引0) 作为安全兜底
        return 0

    def get_path(self, idx: int) -> Optional[List[int]]:
        """安全获取路径"""
        if 0 <= idx < len(self.paths):
            return self.paths[idx]
        return None

    def get_all_paths(self) -> List[List[int]]:
        """获取所有路径"""
        return self.paths.copy()

    def reset(self):
        """重置管理器"""
        self.paths.clear()
        self.path_to_idx.clear()

    def __len__(self):
        return len(self.paths)


class SFC_HIRL_Env(gym.Env):
    """
    分层SFC环境（与 MSFCE_Solver 新接口兼容的完整版本）。
    """

    def __init__(self, input_dir: Path, topo: np.ndarray, dc_nodes: List[int], capacities: Dict):
        super(SFC_HIRL_Env, self).__init__()

        # 注意：确保导入的 MSFCE_Solver 是与你提供的版本（_fixed_full）一致
        self.expert = MSFCE_Solver(input_dir / "US_Backbone_path.mat", topo, dc_nodes, capacities)

        self.T = 400  # 总时间步
        self.n, self.L, self.K_vnf = self.expert.node_num, self.expert.link_num, self.expert.type_num
        self.K_path = self.expert.k_path_count

        # 资源状态
        self.B_cap = capacities['bandwidth']
        self.C_cap = capacities['cpu']
        self.M_cap = capacities['memory']

        self.B = np.full(self.L, self.B_cap, dtype=float)
        self.C = np.full(self.n, self.C_cap, dtype=float)
        self.M = np.full(self.n, self.M_cap, dtype=float)
        self.hvt_all = np.zeros((self.n, self.K_vnf), dtype=int)
        self.link_ref_count = np.zeros(self.L, dtype=int)

        # 加载 data (.mat)
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
            print(f"致命错误: 找不到 {input_dir} 下的 .mat 数据文件")
            raise
        except Exception as e:
            print(f"致命错误: 加载 .mat 文件失败: {e}")
            raise

        # HIRL 状态
        self.t = 0
        self.current_request: Optional[Dict] = None
        self.unadded_dest_indices: Set[int] = set()
        self.current_tree: Optional[Dict] = None
        self.nodes_on_tree: Set[int] = set()
        self.served_requests: List[Tuple[Dict, Dict]] = []

        # 高层/低层动作空间参数
        self.NB_HIGH_LEVEL_GOALS = 10
        self.MAX_PATHS_IN_TREE = 10
        self.NB_LOW_LEVEL_ACTIONS = self.MAX_PATHS_IN_TREE * self.K_path

        # PathManager
        self.path_manager = PathManager(max_paths=self.MAX_PATHS_IN_TREE)

        # state dims
        self.dim_cpu = self.n
        self.dim_mem = self.n
        self.dim_bw = self.L
        self.dim_hvt = self.n * self.K_vnf
        self.dim_network = self.dim_cpu + self.dim_mem + self.dim_bw + self.dim_hvt
        self.dim_request = 10
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        print(f"状态维度: 网络={self.dim_network}, 请求={self.dim_request}, 总计={self.STATE_VECTOR_SIZE}")

        # gym spaces (observation normalized 0-1)
        self.action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32)

        # metrics
        self.total_requests_seen = 0
        self.total_requests_accepted = 0
        self.total_dest_seen = 0
        self.total_dest_accepted = 0

    # -----------------------------
    # network state for expert
    # -----------------------------
    def _get_network_state_dict(self) -> Dict:
        return {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }

    # -----------------------------
    # flatten state for agent
    # -----------------------------
    def _get_flat_state(self) -> np.ndarray:
        net_state = self._get_network_state_dict()

        cpu_norm = net_state['cpu'] / max(1.0, self.C_cap)
        mem_norm = net_state['mem'] / max(1.0, self.M_cap)
        bw_norm = net_state['bw'] / max(1.0, self.B_cap)

        hvt_norm = np.clip(net_state['hvt'].flatten() / 10.0, 0, 1)

        req_vec = np.zeros(self.dim_request)
        if self.current_request:
            req_vec[0] = self.current_request['bw_origin'] / max(1.0, self.B_cap)
            if self.current_request.get('cpu_origin'):
                req_vec[1] = np.mean(self.current_request['cpu_origin']) / max(1.0, self.C_cap)
            if self.current_request.get('memory_origin'):
                req_vec[2] = np.mean(self.current_request['memory_origin']) / max(1.0, self.M_cap)
            req_vec[3] = len(self.current_request['vnf']) / 8.0
            req_vec[4] = len(self.current_request['dest']) / 10.0
            if len(self.current_request['dest']) > 0:
                completed = len(self.current_request['dest']) - len(self.unadded_dest_indices)
                req_vec[5] = completed / len(self.current_request['dest'])
            req_vec[6] = 1.0 if self.current_request['source'] in self.expert.DC else 0.0
            if self.nodes_on_tree:
                req_vec[7] = len(self.nodes_on_tree) / self.n
            if self.current_tree:
                req_vec[8] = np.sum(self.current_tree['tree'] > 0) / max(1.0, self.L)
            req_vec[9] = len(self.unadded_dest_indices) / max(1, len(self.current_request['dest']))

        flat_net = np.concatenate([cpu_norm, mem_norm, bw_norm, hvt_norm])
        final_state = np.zeros(self.STATE_VECTOR_SIZE, dtype=np.float32)
        final_state[:self.dim_network] = flat_net[:self.dim_network]
        final_state[self.dim_network:] = req_vec
        return final_state

    # -----------------------------
    # handle leave events
    # -----------------------------
    def _handle_leave_events(self, t: int):
        if t >= len(self.events):
            return
        leave_ids = self.events[t]['leave']
        if leave_ids.size == 0:
            return
        leave_set = set(leave_ids)
        remaining = []
        for req, tree in self.served_requests:
            if req['id'] in leave_set:
                # release bw
                for link_idx in np.where(tree['tree'] > 0)[0]:
                    if self.link_ref_count[link_idx] > 0:
                        self.link_ref_count[link_idx] -= 1
                        if self.link_ref_count[link_idx] == 0:
                            self.B[link_idx] += req['bw_origin']
                # release vnfs
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
                remaining.append((req, tree))
        self.served_requests = remaining

    # -----------------------------
    # reset_request
    # -----------------------------
    def reset_request(self) -> Tuple[Optional[Dict], np.ndarray]:
        print(f"DEBUG reset_request: t={self.t}, events_len={len(self.events)}")
        if self.t < len(self.events):
            try:
                print("  arrive sample:", self.events[self.t]['arrive'][:10])
            except Exception:
                pass

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
                req_id = arrive_ids[0]
                if req_id in self.req_map:
                    self.current_request = self.req_map[req_id]

        if self.current_request is None:
            return None, self._get_flat_state()

        # update metrics
        self.total_requests_seen += 1
        self.total_dest_seen += len(self.current_request.get('dest', []))

        # init current tree etc.
        self.unadded_dest_indices = set(range(len(self.current_request['dest'])))
        self.current_tree = {
            'id': self.current_request['id'],
            'tree': np.zeros(self.L),
            'hvt': np.zeros((self.n, self.K_vnf)),
            'paths_map': {}
        }
        self.nodes_on_tree = set([self.current_request['source']])
        self.path_manager.reset()
        return self.current_request, self._get_flat_state()

    # -----------------------------
    # expert high level goal (oracle)
    # -----------------------------
    def get_expert_high_level_goal(self, state_vec: np.ndarray) -> int:
        if not self.current_request or not self.unadded_dest_indices:
            return 0

        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request

        # stage1: choose best S->d
        if not self.current_tree['paths_map']:
            tree_set = []
            for d_idx in self.unadded_dest_indices:
                best_eval = -1.0
                for k in range(1, self.K_path + 1):
                    # expert._calc_eval returns 8 values
                    eval_val, paths, tree, hvt_new, feasible, dest, cost, placement = \
                        self.expert._calc_eval(self.current_request, d_idx, k, network_state)
                    if feasible and eval_val > best_eval:
                        best_eval = eval_val
                tree_set.append((d_idx, best_eval))

            if not tree_set or max(tree_set, key=lambda item: item[1])[1] <= 0:
                return list(self.unadded_dest_indices)[0]
            best_d_idx, _ = max(tree_set, key=lambda item: item[1])
            return best_d_idx

        # stage2: choose best Tree->d
        else:
            best_eval, best_d = -1, -1
            for d_idx in self.unadded_dest_indices:
                for conn_path in self.current_tree['paths_map'].values():
                    # _calc_atnp returns (plan, best_eval, best_action, best_cost)
                    plan, m, action, cost = self.expert._calc_atnp(
                        self.current_tree, conn_path, d_idx, network_state, self.nodes_on_tree
                    )
                    if m > best_eval:
                        best_eval, best_d = m, d_idx
            return best_d if best_d != -1 else list(self.unadded_dest_indices)[0]

    # -----------------------------
    # Path helpers
    # -----------------------------
    def _get_path_for_i_idx(self, i_idx: int) -> List[int]:
        if not self.current_tree or not self.current_tree['paths_map']:
            return [self.current_request['source']]
        path = self.path_manager.get_path(i_idx)
        if path is None:
            if self.path_manager.paths:
                return self.path_manager.get_path(0)
            return [self.current_request['source']]
        return path

    def get_valid_low_level_actions(self) -> List[int]:
        valid_actions = []
        if not self.current_tree or not self.current_tree['paths_map']:
            for k in range(self.K_path):
                valid_actions.append(0 * self.K_path + k)
        else:
            num_paths = len(self.path_manager)
            if num_paths == 0:
                num_paths = len(self.current_tree['paths_map'])
            for i in range(num_paths):
                for k in range(self.K_path):
                    action_id = i * self.K_path + k
                    if action_id < self.NB_LOW_LEVEL_ACTIONS:
                        valid_actions.append(action_id)
        valid = valid_actions if valid_actions else [0]
        print(f"DEBUG get_valid_low_level_actions -> count={len(valid)} sample={valid[:10]}")
        return valid

    def _decode_low_level_action(self, action: int) -> Tuple[int, int]:
        k_idx = action % self.K_path
        i_idx = action // self.K_path
        num_paths = max(1, len(self.path_manager))
        i_idx = i_idx % min(num_paths, self.MAX_PATHS_IN_TREE)
        return i_idx, k_idx

    # -----------------------------
    # Main: step_low_level (FULLY FIXED & COMPATIBLE)
    # -----------------------------
    def step_low_level(self, goal_dest_idx: int, low_level_action) -> \
            Tuple[np.ndarray, float, bool, bool]:
        """
        支持两种低层动作格式：
        1) 旧格式：low_level_action = int
        2) 新格式：low_level_action = (i_idx, k_idx, placement_dict)
        函数返回 (next_state, cost, sub_task_done, request_done)
        """
        # parse action
        if isinstance(low_level_action, tuple) and len(low_level_action) == 3:
            i_idx, k_idx, placement = low_level_action
            # if flat-action was passed accidentally encoded as tuple of ints, still ok
        else:
            i_idx, k_idx = self._decode_low_level_action(low_level_action)
            placement = {}

        k = k_idx + 1  # expert uses 1-based k

        # invalid goal or already done
        if self.current_request is None or goal_dest_idx not in self.unadded_dest_indices:
            return self._get_flat_state(), 0.0, True, not (self.unadded_dest_indices or self.current_request)

        # build network state for expert calls
        network_state = self._get_network_state_dict()
        network_state['request'] = self.current_request

        plan = None
        cost = 0.0
        feasible = False
        eval_val = "NA"

        # Stage 1: S -> d (tree empty)
        if not self.current_tree['paths_map']:
            # expert._calc_eval returns 8 values
            eval_val, paths, tree, hvt, feasible, dest_or_flag, cost, placement_from_expert = \
                self.expert._calc_eval(self.current_request, goal_dest_idx, k, network_state)

            if feasible:
                plan = {'tree': tree, 'hvt': hvt, 'new_path_full': paths, 'feasible': True,
                        'placement': placement_from_expert}
        else:
            # Stage 2: Tree -> d
            conn_path = self._get_path_for_i_idx(i_idx)
            # _calc_atnp returns (plan_dict, best_eval, action, best_cost)
            plan, eval_val, action, cost = self.expert._calc_atnp(
                self.current_tree, conn_path, goal_dest_idx, network_state, self.nodes_on_tree
            )
            feasible = plan.get('feasible', False)

        print(f"[step_low_level] goal={goal_dest_idx}, i={i_idx}, k={k}, feasible={feasible}, "
              f"placement_keys={list(placement.keys()) if placement else []}, eval={eval_val}")

        # apply plan if feasible
        if feasible and plan is not None:
            # apply resources (plan is a dict with 'tree' and 'hvt')
            self._apply_deployment(self.current_request, plan)

            # remove goal
            self.unadded_dest_indices.remove(goal_dest_idx)
            dest_node = self.current_request['dest'][goal_dest_idx]

            # metrics
            self.total_dest_accepted += 1

            # register path via path_manager
            new_path = plan.get('new_path_full', [])
            if new_path:
                self.path_manager.add_path(new_path)
                self.current_tree['paths_map'][dest_node] = new_path
                self.nodes_on_tree.update(new_path)

            sub_task_done = True

            # if request finished
            if not self.unadded_dest_indices:
                if (self.current_request, self.current_tree) not in self.served_requests:
                    self.served_requests.append((self.current_request, self.current_tree))
                self.total_requests_accepted += 1
        else:
            cost = 10.0
            sub_task_done = False
            print(f"[step_low_level] plan infeasible for goal {goal_dest_idx}")

        request_done = not self.unadded_dest_indices
        return self._get_flat_state(), float(cost), sub_task_done, request_done

    # -----------------------------
    # Apply deployment: consume resources
    # -----------------------------
    def _apply_deployment(self, request: Dict, plan: Dict):
        """
        plan expected to contain:
            - 'tree': binary vector over links (len L)
            - 'hvt': n x K_vnf binary matrix (new deployments marked as 1)
        """
        tree_branch = plan.get('tree', np.zeros(self.L))
        hvt_branch = plan.get('hvt', np.zeros((self.n, self.K_vnf)))

        # merge tree
        self.current_tree['tree'] = np.logical_or(self.current_tree['tree'], tree_branch).astype(float)

        # consume bandwidth
        for link_idx in np.where(tree_branch > 0)[0]:
            if self.link_ref_count[link_idx] == 0:
                self.B[link_idx] -= request['bw_origin']
            self.link_ref_count[link_idx] += 1

        # consume node resources for newly deployed VNFs
        for node, vnf_t in np.argwhere(hvt_branch > 0):
            if self.hvt_all[node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(int(vnf_t + 1))
                    self.C[node] -= request['cpu_origin'][j]
                    self.M[node] -= request['memory_origin'][j]
                except ValueError:
                    pass
            self.hvt_all[node, vnf_t] += 1

        # merge hvt
        self.current_tree['hvt'] = np.maximum(self.current_tree['hvt'], hvt_branch)

