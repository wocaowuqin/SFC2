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
        # ----------------------------------------------------
        # ✅ 建议: 添加文件加载异常处理
        # ----------------------------------------------------
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

        # 定义状态和动作空间
        # TODO: 这是一个关键且困难的定义。这里使用一个简化的占位符。
        # 状态 = (CPU 负载, Mem 负载, BW 负载, HVT 状态) + (请求信息)
        state_size = self.n + self.n + self.L + (self.n * self.K_vnf)
        req_size = 10  # 简化的请求特征向量
        self.STATE_VECTOR_SIZE = state_size + req_size

        # 高层动作 = 选择哪个目的地 (假设最多10个目的地)
        self.NB_HIGH_LEVEL_GOALS = 10

        # 低层动作 = (连接到哪个节点 i, 使用哪条路径 k)
        # 修复建议：i_idx 应该是树上路径的索引，而不是节点索引
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
        # ✅ 修复：防止越界
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

            # ✅ 修复：防止越界
            if self.t >= len(self.events):
                self.t += 1
                continue

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

    # ----------------------------------------------------
    # ✅ 修复 #3 和 #5: 替换为您的新 get_expert_high_level_goal 方法
    # ----------------------------------------------------
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
                    # ✅ [修复 #3] 完整解包
                    eval_val, paths, tree, hvt_new, feasible, dest, cost = \
                        self.expert._calc_eval(self.current_request, d_idx, k, network_state)
                    if feasible and eval_val > best_eval:
                        best_eval = eval_val
                tree_set.append((d_idx, best_eval))

            if not tree_set or max(tree_set, key=lambda item: item[1])[1] <= 0:
                return list(self.unadded_dest_indices)[0]  # 随机选一个

            best_d_idx, _ = max(tree_set, key=lambda item: item[1])
            return best_d_idx  # ✅ [修复 #5] 返回原始索引

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
            # ✅ [修复 #5] 返回原始索引
            return best_d if best_d != -1 else list(self.unadded_dest_indices)[0]

    # ----------------------------------------------------
    # ✅ 修复 #6: 添加动作掩码函数
    # ----------------------------------------------------
    def get_valid_low_level_actions(self) -> List[int]:
        """返回当前状态下有效的低层动作ID列表"""
        valid_actions = []
        if not self.current_tree['paths_map']:
            # S->d 阶段: 只有 (i=0, k=0-4) 有效
            for k in range(self.K_path):
                valid_actions.append(0 * self.K_path + k)
        else:
            # Tree->d 阶段: 遍历所有树上的路径
            # TODO: 修复 #6.c, 这里的 i_idx 映射仍然是暂时的
            num_paths = len(self.current_tree['paths_map'])
            for i in range(num_paths):
                for k in range(self.K_path):
                    # 确保动作ID不超过最大值
                    action_id = i * self.K_path + k
                    if action_id < self.NB_LOW_LEVEL_ACTIONS:
                        valid_actions.append(action_id)

        if not valid_actions:
            # 确保总有至少一个动作可选，防止掩码为空
            return [0]

        return valid_actions

    def _decode_low_level_action(self, action: int) -> Tuple[int, int]:
        """将扁平化的低层动作ID (0-N) 解码为 (i_idx, k_idx)"""
        k_idx = action % self.K_path
        i_idx = action // self.K_path
        # ----------------------------------------------------
        # ⚠️ 修复 #6.c (建议): 这里的映射仍然是不稳定的!
        # ----------------------------------------------------
        # 确保 i_idx 在当前树的节点范围内
        i_idx = i_idx % max(1, len(self.current_tree['paths_map']))
        return i_idx, k_idx

    def _get_path_for_i_idx(self, i_idx: int) -> List[int]:
        """(TODO) 根据 i_idx 获取树上的特定连接路径"""
        if not self.current_tree['paths_map']:
            return [self.current_request['source']]  # 对于 S->d, i_idx 总是0

        # ----------------------------------------------------
        # ⚠️ 修复 #6.c (建议): 这里的映射仍然是不稳定的!
        # ----------------------------------------------------
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