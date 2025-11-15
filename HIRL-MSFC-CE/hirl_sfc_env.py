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
# 修复 #5: (补丁 5) 添加 PathManager
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

        # 如果已满,返回 -1 表示失败 (或返回第一个,取决于策略)
        # return -1
        # 返回 0 (索引0) 作为安全兜底
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
        self.served_requests = []  # (req, plan)

        # 高层动作 = 选择哪个目的地 (假设最多10个目的地)
        self.NB_HIGH_LEVEL_GOALS = 10

        # 低层动作 = (连接到哪条路径 i, 使用哪条路径 k)
        self.MAX_PATHS_IN_TREE = 10  # 假设一棵树最多有10条路径
        self.NB_LOW_LEVEL_ACTIONS = self.MAX_PATHS_IN_TREE * self.K_path

        # ----------------------------------------------------
        # 修复 #5: (补丁 5) 使用 PathManager
        # ----------------------------------------------------
        self.path_manager = PathManager(max_paths=self.MAX_PATHS_IN_TREE)

        # ----------------------------------------------------
        # 修复 #2: (补丁 2) 明确计算状态维度
        # ----------------------------------------------------
        self.dim_cpu = self.n
        self.dim_mem = self.n
        self.dim_bw = self.L
        self.dim_hvt = self.n * self.K_vnf
        self.dim_network = self.dim_cpu + self.dim_mem + self.dim_bw + self.dim_hvt
        self.dim_request = 10  # 简化的请求特征向量
        self.STATE_VECTOR_SIZE = self.dim_network + self.dim_request

        print(f"状态维度: 网络={self.dim_network}, 请求={self.dim_request}, 总计={self.STATE_VECTOR_SIZE}")

        # 定义 Gym 空间
        self.action_space = spaces.Discrete(self.NB_HIGH_LEVEL_GOALS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32)

        # ----------------------------------------------------
        # ✅ 修复: (补丁 A) 添加你的新计数器
        # ----------------------------------------------------
        self.total_requests_seen = 0  # 新到达的请求数
        self.total_requests_accepted = 0  # 完整被接受的请求数（所有dest都部署完成）
        self.total_dest_seen = 0  # 累计到达的目的地数量（sum len(dest)）
        self.total_dest_accepted = 0  # 累计成功部署的目的地数量（每成功部署1个 +1）

    def _get_network_state_dict(self) -> Dict:
        """获取专家求解器所需格式的当前网络状态"""
        return {
            'bw': self.B, 'cpu': self.C, 'mem': self.M,
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }

    # ----------------------------------------------------
    # 修复 #2: 替换为您的新 _get_flat_state 方法
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
        req_vec = np.zeros(self.dim_request)  # 使用 self.dim_request
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
        # (使用 self.dim_network 和 self.dim_request 保证维度)
        final_state[:self.dim_network] = flat_state[:self.dim_network]

        # 请求特征部分 (最后10维)
        final_state[self.dim_network:] = req_vec

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

        # 修复: (补丁 A) 添加你的 Debug 探针
        print(f"DEBUG reset_request: t={self.t}, events_len={len(self.events)}")
        if self.t < len(self.events):
            try:
                print("  arrive sample:", self.events[self.t]['arrive'][:10])
            except Exception as e:
                print("  cannot print arrive, err:", e)

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

        # ----------------------------------------------------
        # ✅ 修复: (补丁 A) 更新计数器
        # ----------------------------------------------------
        if self.current_request is not None:
            self.total_requests_seen += 1
            self.total_dest_seen += len(self.current_request.get('dest', []))

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
        # 修复 #5: (补丁 5) 重置路径管理器
        # ----------------------------------------------------
        self.path_manager.reset()

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
    # 修复 #5: (补丁 5) 替换为 PathManager 版本
    # ----------------------------------------------------

    def _get_path_for_i_idx(self, i_idx: int) -> List[int]:
        """修复: 根据 i_idx 获取树上的特定连接路径"""
        if not self.current_tree['paths_map']:
            # 对于 S->d 阶段, i_idx 总是0，返回源节点
            return [self.current_request['source']]

        path = self.path_manager.get_path(i_idx)
        if path is None:
            # 兜底: 尝试从 paths_map 获取或返回源节点
            if self.path_manager.paths:
                return self.path_manager.get_path(0)
            return [self.current_request['source']]
        return path

    def get_valid_low_level_actions(self) -> List[int]:
        """修复: 返回当前状态下有效的低层动作ID列表"""
        valid_actions = []

        if not self.current_tree['paths_map']:
            # S->d 阶段: 只有 (i=0, k=0-4) 有效
            for k in range(self.K_path):
                valid_actions.append(0 * self.K_path + k)
        else:
            # Tree->d 阶段: 遍历所有树上的路径
            num_paths = len(self.path_manager)
            if num_paths == 0:
                # 兜底，如果 path_manager 为空但 paths_map 不为空
                num_paths = len(self.current_tree['paths_map'])

            for i in range(num_paths):
                for k in range(self.K_path):
                    action_id = i * self.K_path + k
                    # 确保动作ID不超过最大值
                    if action_id < self.NB_LOW_LEVEL_ACTIONS:
                        valid_actions.append(action_id)

        # 确保总有至少一个动作可选
        valid = valid_actions if valid_actions else [0]
        # 修复: (补丁 B) 添加你的 Debug 探针
        print(f"DEBUG get_valid_low_level_actions -> count={len(valid)} sample={valid[:10]}")
        return valid

    def _decode_low_level_action(self, action: int) -> Tuple[int, int]:
        """将扁平化的低层动作ID (0-N) 解码为 (i_idx, k_idx)"""
        k_idx = action % self.K_path
        i_idx = action // self.K_path

        # 限制 i_idx 在当前树的路径数内 (或最大动作空间)
        num_paths = max(1, len(self.path_manager))
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
        eval_val = "NA"  # (用于 Debug 探针)

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

        # 修复: (补丁 C) 添加你的 Debug 探针
        print(
            f"DEBUG step_low_level try: goal={goal_dest_idx}, i_idx={i_idx}, k={k}, feasible_guess={feasible}, eval_val={eval_val}")

        if feasible:
            # 修复: (补丁 D) 添加你的 Debug 探针
            print("  APPLIED deployment. unadded before removal:", self.unadded_dest_indices)

            # 成功！应用资源
            self._apply_deployment(self.current_request, plan)
            self.unadded_dest_indices.remove(goal_dest_idx)
            dest_node = self.current_request['dest'][goal_dest_idx]

            # ----------------------------------------------------
            # ✅ 修复: (补丁 B) 更新子目的地计数
            # ----------------------------------------------------
            self.total_dest_accepted += 1
            print(f"METRIC: total_dest_seen={self.total_dest_seen}, total_dest_accepted={self.total_dest_accepted}")

            # ----------------------------------------------------
            # 修复 #5: (补丁 5) 更新路径管理器
            # ----------------------------------------------------
            new_path = plan['new_path_full']
            self.path_manager.add_path(new_path)  # 添加到稳定列表
            self.current_tree['paths_map'][dest_node] = new_path
            self.nodes_on_tree.update(new_path)

            # 修复: (补丁 D) 添加你的 Debug 探针
            print("  unadded after:", self.unadded_dest_indices, "served_requests_len:", len(self.served_requests))

            sub_task_done = True
            if not self.unadded_dest_indices:
                # 整个请求完成了
                if (self.current_request, self.current_tree) not in self.served_requests:
                    self.served_requests.append((self.current_request, self.current_tree))

                # ----------------------------------------------------
                # ✅ 修复: (补丁 B) 更新完整请求计数
                # ----------------------------------------------------
                self.total_requests_accepted += 1
                print(
                    f"METRIC: Request {self.current_request['id']} COMPLETED. total_requests_accepted={self.total_requests_accepted}")
        else:
            # 失败！
            cost = 10.0  # 惩罚
            sub_task_done = False  # 修复: 任务未完成, Agent 必须重试

            # 修复: (补丁 C) 添加你的 Debug 探针
            print(f"  plan infeasible for goal {goal_dest_idx}")

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