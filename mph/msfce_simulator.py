#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : msfce_simulator_fixed.py
# @Date    : 2025-11-06
# @Author  : gxr (Fixed by Claude)
# @Desc    : 完全修复的 MSFC-CE 仿真器 + DRL 环境

import numpy as np
import scipy.io as sio
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# === DRL 库导入 (可选) ===
try:
    import gym
    from gym import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    HAS_DRL = True
except ImportError:
    HAS_DRL = False
    print("⚠️ 警告: 未安装 DRL 库 (gym, stable-baselines3)")
    print("   将只运行基础仿真 (不使用强化学习)")


# ==============================================================
# ✅ .mat 文件解析
# ==============================================================
def parse_mat_request(req_obj) -> Dict:
    """
    将 MATLAB 请求结构 (来自 sorted_requests.mat) 解析为 Python 字典
    """
    req = req_obj

    # 兼容两种您提供的 .mat 结构
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


# ==============================================================
# ✅ MSFCE 静态求解器
# ==============================================================
class MSFCE_Solver:
    """MSFC-CE 启发式算法求解器 (无状态)"""

    def __init__(self, path_db_file: Path, topology_matrix: np.ndarray,
                 dc_nodes: List[int], capacities: Dict):
        print("初始化 MSFC-CE 求解器...")
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
        print(f"✅ 求解器加载: {self.node_num}节点, {self.link_num}链路, {self.dc_num}DC")

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
        """✅ 修复: 添加输入验证"""
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

    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """评估 S->d 的第k条路径"""
        bw, cpu, mem, hvt = state['bw'], state['cpu'], state['mem'], state['hvt']
        src, dest = request['source'], request['dest'][d_idx]

        path, dist, links = self._get_path_from_db(src, dest, k)
        if not path:
            return 0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dest

        tree = np.zeros(self.link_num)
        hvt_new = np.zeros((self.node_num, self.type_num))
        usable = [n for n in path if n in self.DC]

        # 检查资源
        if len(usable) < len(request['vnf']):
            return 0, path, tree, hvt_new, False, dest
        for lid in links:
            if bw[lid - 1] < request['bw_origin']:
                return 0, path, tree, hvt_new, False, dest

        # VNF 放置
        j, i = 0, 0
        while j < len(request['vnf']):
            if i >= len(usable):
                return 0, path, tree, hvt_new, False, dest
            node, vnf_t = usable[i] - 1, request['vnf'][j] - 1
            if hvt[node, vnf_t] == 0:
                if cpu[node] < request['cpu_origin'][j] or mem[node] < request['memory_origin'][j]:
                    i += 1
                    continue
            hvt_new[node, vnf_t] = 1
            j, i = j + 1, i + 1

        if np.sum(hvt_new) != len(request['vnf']):
            return 0, path, tree, hvt_new, False, dest

        for lid in links:
            tree[lid - 1] = 1

        # 计算得分
        max_dist = self._get_kth_path_max_distance(src, dest, self.k_path_count)
        score = (
                (1 - dist / max_dist if max_dist > 0 else 1) +
                len(usable) / self.dc_num +
                np.sum(cpu[np.array(path) - 1]) / (self.cpu_capacity * self.dc_num) +
                np.sum(mem[np.array(path) - 1]) / (self.memory_capacity * self.dc_num) +
                np.sum(bw[np.array(links) - 1]) / (self.bandwidth_capacity * self.link_num)
        )
        return score, path, tree, hvt_new, True, dest

    def _calc_eval1(self, d_idx: int, k: int, i_idx: int, tree1_path: List[int],
                    request: Dict, tree1_hvt: np.ndarray, state: Dict, nodes_on_tree: Set[int]):
        """
        评估从树上第 i_idx 个节点到目的节点 d_idx 的第 k 条路径
        ✅ 修复版本
        """
        hvt = tree1_hvt.copy()
        tree = np.zeros(self.link_num)
        tree_paths = tree1_path[:i_idx + 1]
        feasible = True
        infeasible_dest = 0

        # 获取连接点和目的节点
        connect_node = tree1_path[i_idx]
        dest_node = request['dest'][d_idx]

        # ===== 步骤 1: 获取连接点 -> 目的节点的第 k 条路径 =====
        paths, dist, links = self._get_path_from_db(connect_node, dest_node, k)

        # ✅ 修复: 检查路径长度
        if not paths or len(paths) < 2:
            return 0, [], tree, hvt, False, dest_node

        # ===== 步骤 2: 检测环路 (三种情况) =====
        arr1 = set(paths[1:])
        arr2 = set(tree_paths)
        if arr1 & arr2:
            return 0, paths, tree, hvt, False, dest_node

        arr4 = nodes_on_tree - set(tree_paths)
        if arr1 & arr4:
            return 0, paths, tree, hvt, False, dest_node

        if i_idx + 1 < len(tree1_path):
            arr6 = set(tree1_path[i_idx + 1:])
            if arr1 & arr6:
                return 0, paths, tree, hvt, False, dest_node

        # ===== 步骤 3: 统计路径上的可用 DC 节点 =====
        usable_on_path = [n for n in paths[1:] if n in self.DC]
        deployed_on_path = [n for n in tree_paths if n in self.DC]

        # ===== 步骤 4: 检查带宽资源 =====
        for lid in links:
            if state['bw'][lid - 1] < request['bw_origin']:
                return 0, paths, tree, hvt, False, dest_node

        # ===== 步骤 5: 计算路径资源 (用于评分) =====
        CPU_status = sum(state['cpu'][n - 1] for n in paths[1:] if n in self.DC)
        Memory_status = sum(state['mem'][n - 1] for n in paths[1:] if n in self.DC)
        Bandwidth_status = sum(state['bw'][lid - 1] for lid in links)

        # ===== 步骤 6: 计算已部署的 VNF 数量 =====
        # ✅ 修复: 只统计请求需要的 VNF 类型
        shared_path_deployed = sum(
            1 for vnf_type in request['vnf']
            if any(hvt[n - 1, vnf_type - 1] > 0 for n in deployed_on_path)
        )
        undeployed_vnf = len(request['vnf']) - shared_path_deployed

        # ===== 步骤 7: VNF 部署检查 =====
        if undeployed_vnf == 0:
            # 所有 VNF 已在共享路径上部署
            eval_score = self._calc_score(
                connect_node, dest_node, dist,  # ✅ 修复: 传入正确的 src/dst
                len(deployed_on_path), CPU_status, Memory_status, Bandwidth_status
            )
            for lid in links:
                tree[lid - 1] = 1
            return eval_score, paths, tree, hvt, True, 0

        else:
            # 需要在新路径上部署 VNF
            if len(usable_on_path) < undeployed_vnf:
                return 0, paths, tree, hvt, False, dest_node

            # 尝试部署未部署的 VNF
            j = shared_path_deployed
            g = 0

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

            # ✅ 修复: 检查是否所有 VNF 都部署成功
            usable_path_deployed = sum(
                1 for vnf_type in request['vnf']
                if any(hvt[n - 1, vnf_type - 1] > 0 for n in usable_on_path)
            )

            # ✅ 修复: 应该与总需求数比较
            total_deployed = sum(
                1 for vnf_type in request['vnf']
                if any(hvt[n - 1, vnf_type - 1] > 0 for n in (deployed_on_path + usable_on_path))
            )

            if total_deployed != len(request['vnf']):
                return 0, paths, tree, hvt, False, dest_node

            eval_score = self._calc_score(
                connect_node, dest_node, dist,  # ✅ 修复: 传入正确的 src/dst
                len(usable_on_path), CPU_status, Memory_status, Bandwidth_status
            )
            for lid in links:
                tree[lid - 1] = 1

            return eval_score, paths, tree, hvt, True, 0

    def _calc_score(self, src: int, dst: int, dist: int, dc_count: int,
                    cpu_sum: float, mem_sum: float, bw_sum: float) -> float:
        """✅ 修复: 计算评分函数，传入正确的 src/dst"""
        max_dist = self._get_kth_path_max_distance(src, dst, self.k_path_count) or 1

        score = (
                (1 - dist / max_dist) +
                dc_count / self.dc_num +
                cpu_sum / (self.cpu_capacity * self.dc_num) +
                mem_sum / (self.memory_capacity * self.dc_num) +
                bw_sum / (self.bandwidth_capacity * self.link_num)
        )
        return score

    def _calc_atnp(self, tree1: Dict, tree1_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """
        找到将目的节点 d 连接到树 tree1 的最佳方案
        """
        request = state['request']

        # 如果上一个树不可行，直接返回
        if tree1.get('eval', 0) == 0:
            return {
                'tree': tree1['tree'].copy(),
                'hvt': tree1['hvt'].copy(),
                'feasible': tree1.get('feasible', False),
                'infeasible_dest': tree1.get('infeasible_dest', 0)
            }, 0

        # ===== 遍历树上的所有可能连接点 =====
        best_eval = -1
        best_plan = None

        for i_idx in range(len(tree1_path)):
            for k in range(1, self.k_path_count + 1):
                eval_val, paths, tree_new, hvt_new, feasible, infeasible_dest = \
                    self._calc_eval1(
                        d_idx, k, i_idx, tree1_path, request,
                        tree1['hvt'], state, nodes_on_tree
                    )

                if feasible and eval_val > best_eval:
                    best_eval = eval_val
                    best_plan = {
                        'tree': tree_new,
                        'hvt': hvt_new,
                        'new_path_full': paths,
                        'connect_idx': i_idx,
                        'feasible': True,
                        'infeasible_dest': 0
                    }

        if best_plan is None:
            return {
                'tree': tree1['tree'].copy(),
                'hvt': tree1['hvt'].copy(),
                'feasible': False,
                'infeasible_dest': request['dest'][d_idx]
            }, 0

        return best_plan, best_eval

    def solve_request(self, request: Dict, network_state: Dict) -> Optional[Dict]:
        """部署请求 (返回部署方案或 None)"""
        dest_num = len(request['dest'])
        network_state['request'] = request

        # 阶段1: 找到所有 S->d 的最佳路径
        tree_set = []
        for d_idx in range(dest_num):
            best_eval, best_result = -1, None
            for k in range(1, self.k_path_count + 1):
                eval_val, paths, tree, hvt, feasible, _ = self._calc_eval(
                    request, d_idx, k, network_state)
                if feasible and eval_val > best_eval:
                    best_eval, best_result = eval_val, {
                        'eval': eval_val, 'paths': paths, 'tree': tree, 'hvt': hvt
                    }
            tree_set.append(best_result if best_result else {'eval': -1})

        # 贪心构建多播树
        best_d_idx = np.argmax([t.get('eval', -1) for t in tree_set])
        if tree_set[best_d_idx]['eval'] <= 0:
            return None

        current_tree = {
            'id': request['id'],
            'tree': tree_set[best_d_idx]['tree'],
            'hvt': tree_set[best_d_idx]['hvt'],
            'paths_map': {request['dest'][best_d_idx]: tree_set[best_d_idx]['paths']}
        }
        nodes_on_tree = set(tree_set[best_d_idx]['paths'])
        unadded = set(range(dest_num)) - {best_d_idx}

        while unadded:
            best_eval, best_plan, best_d = -1, None, -1
            for d_idx in unadded:
                for conn_path in current_tree['paths_map'].values():
                    t, m = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, network_state, nodes_on_tree
                    )
                    if t.get('feasible') and m > best_eval:
                        best_eval, best_plan, best_d = m, t, d_idx

            if best_d == -1:
                return None

            # ✅ 修复: 树结构合并
            current_tree['tree'] = np.logical_or(
                current_tree['tree'],
                best_plan['tree']
            ).astype(float)

            # ✅ 修复: hvt 使用 maximum 而非 logical_or (保留引用计数信息)
            current_tree['hvt'] = np.maximum(
                current_tree['hvt'],
                best_plan['hvt']
            )

            current_tree['paths_map'][request['dest'][best_d]] = best_plan['new_path_full']
            nodes_on_tree.update(best_plan['new_path_full'])
            unadded.remove(best_d)

        # 计算资源成本
        used_links = np.where(current_tree['tree'] > 0)[0]
        new_links_mask = (network_state['bw_ref_count'][used_links] == 0)
        bw_cost = np.sum(new_links_mask) * request['bw_origin']

        cpu_cost, mem_cost = 0, 0
        for node, vnf_t in np.argwhere(current_tree['hvt'] > 0):
            if network_state['hvt'][node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    cpu_cost += request['cpu_origin'][j]
                    mem_cost += request['memory_origin'][j]
                except ValueError:
                    pass

        return {
            'id': request['id'],
            'tree': current_tree['tree'],
            'hvt': current_tree['hvt'],
            'cost_bw': bw_cost,
            'cost_cpu': cpu_cost,
            'cost_mem': mem_cost
        }


# ==============================================================
# ✅ 动态仿真器
# ==============================================================
class DynamicSimulator:
    """动态网络仿真环境 (管理时间、事件和资源)"""

    def __init__(self, input_dir: Path, output_dir: Path, topo: np.ndarray,
                 dc_nodes: List[int], capacities: Dict):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.solver = MSFCE_Solver(input_dir / "US_Backbone_path.mat", topo, dc_nodes, capacities)

        self.T = 400
        n, L, K = self.solver.node_num, self.solver.link_num, self.solver.type_num
        self.dc_nodes_0based = np.array(dc_nodes) - 1

        # 资源状态
        self.B = np.full((self.T, L), capacities['bandwidth'])
        self.C = np.full((self.T, n), capacities['cpu'])
        self.M = np.full((self.T, n), capacities['memory'])
        self.hvt_all = np.zeros((n, K), dtype=int)
        self.link_ref_count = np.zeros(L, dtype=int)

        # 加载数据
        print(f"加载数据从 {input_dir}...")
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
        print(f"✅ 数据加载: {len(self.requests)}请求, {len(self.events)}时间步")

        # 仿真状态
        self.served_requests, self.served_trees = [], []
        self.block, self.arrived = 0, 0

        # 日志
        self.logs = {
            'cpu_load_var': np.zeros(self.T), 'memory_load_var': np.zeros(self.T),
            'bandwidth_load_var': np.zeros(self.T), 'cpu_resource_comp': np.zeros(self.T),
            'memory_resource_comp': np.zeros(self.T), 'bandwidth_resource_comp': np.zeros(self.T),
            'blocking_rate': np.zeros(self.T), 'duration': np.zeros(self.T)
        }
        self.temp_step_costs = {'bw': 0, 'cpu': 0, 'mem': 0}

    def get_state(self, t: int) -> Dict:
        """获取当前时刻的网络状态"""
        return {
            'bw': self.B[t], 'cpu': self.C[t], 'mem': self.M[t],
            'hvt': self.hvt_all, 'bw_ref_count': self.link_ref_count
        }

    def _handle_leave_event(self, t: int, leave_ids: np.ndarray):
        """✅ 修复: 处理离开事件"""
        if leave_ids.size == 0:
            return

        leave_set = set(leave_ids)
        remaining_reqs = []

        for req in self.served_requests:
            if req['id'] in leave_set:
                try:
                    tree = next(tr for tr in self.served_trees if tr['id'] == req['id'])

                    # 释放带宽
                    for link_idx in np.where(tree['tree'] > 0)[0]:
                        if self.link_ref_count[link_idx] > 0:
                            self.link_ref_count[link_idx] -= 1
                            if self.link_ref_count[link_idx] == 0:
                                self.B[t, link_idx] += req['bw_origin']

                    # ✅ 修复: 释放 VNF (处理可能的重复类型)
                    for node, vnf_t in np.argwhere(tree['hvt'] > 0):
                        if self.hvt_all[node, vnf_t] > 0:
                            self.hvt_all[node, vnf_t] -= 1
                            if self.hvt_all[node, vnf_t] == 0:
                                # 找到对应的资源量 (只释放一次)
                                vnf_type_to_find = vnf_t + 1
                                if vnf_type_to_find in req['vnf']:
                                    j = req['vnf'].index(vnf_type_to_find)
                                    self.C[t, node] += req['cpu_origin'][j]
                                    self.M[t, node] += req['memory_origin'][j]
                except StopIteration:
                    print(f"⚠️ 找不到 ID={req['id']} 的树")
            else:
                remaining_reqs.append(req)

        self.served_requests = remaining_reqs
        self.served_trees = [t for t in self.served_trees if t['id'] in {r['id'] for r in remaining_reqs}]

    def _apply_deployment(self, t: int, request: Dict, plan: Dict):
        """应用部署方案 (占用资源,使用引用计数)"""
        # 占用带宽
        for link_idx in np.where(plan['tree'] > 0)[0]:
            if self.link_ref_count[link_idx] == 0:
                self.B[t, link_idx] -= request['bw_origin']
            self.link_ref_count[link_idx] += 1

        # 占用 VNF
        for node, vnf_t in np.argwhere(plan['hvt'] > 0):
            if self.hvt_all[node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    self.C[t, node] -= request['cpu_origin'][j]
                    self.M[t, node] -= request['memory_origin'][j]
                except ValueError:
                    pass
            self.hvt_all[node, vnf_t] += 1

    def _handle_arrive_event(self, t: int, arrive_ids: np.ndarray):
        """处理到达事件"""
        self.temp_step_costs = {'bw': 0, 'cpu': 0, 'mem': 0}
        if arrive_ids.size == 0:
            return

        for rid in arrive_ids:
            if rid not in self.req_map:
                continue

            self.arrived += 1
            req = self.req_map[rid]
            plan = self.solver.solve_request(req, self.get_state(t))

            if plan is None:
                self.block += 1
            else:
                self._apply_deployment(t, req, plan)
                self.served_requests.append(req)
                self.served_trees.append(plan)
                self.temp_step_costs['bw'] += plan['cost_bw']
                self.temp_step_costs['cpu'] += plan['cost_cpu']
                self.temp_step_costs['mem'] += plan['cost_mem']

    def _log_metrics(self, t: int, t1: float):
        """记录性能指标"""
        self.logs['duration'][t] = time.time() - t1

        # 累积资源消耗
        if t > 0:
            self.logs['cpu_resource_comp'][t] = self.logs['cpu_resource_comp'][t - 1] + self.temp_step_costs['cpu']
            self.logs['memory_resource_comp'][t] = self.logs['memory_resource_comp'][t - 1] + self.temp_step_costs[
                'mem']
            self.logs['bandwidth_resource_comp'][t] = self.logs['bandwidth_resource_comp'][t - 1] + \
                                                      self.temp_step_costs['bw']
        else:
            self.logs['cpu_resource_comp'][t] = self.temp_step_costs['cpu']
            self.logs['memory_resource_comp'][t] = self.temp_step_costs['mem']
            self.logs['bandwidth_resource_comp'][t] = self.temp_step_costs['bw']

        cpu_load_t = 1 - (self.C[t] / self.solver.cpu_capacity)
        self.logs['cpu_load_var'][t] = np.var(cpu_load_t[self.dc_nodes_0based])

        mem_load_t = 1 - (self.M[t] / self.solver.memory_capacity)
        self.logs['memory_load_var'][t] = np.var(mem_load_t[self.dc_nodes_0based])

        bw_load_t = 1 - (self.B[t] / self.solver.bandwidth_capacity)
        self.logs['bandwidth_load_var'][t] = np.var(bw_load_t)

        if self.arrived > 0:
            self.logs['blocking_rate'][t] = self.block / self.arrived

    def _save_results(self):
        """保存结果到 .mat 文件"""
        print(f"保存结果到 {self.output_dir}")
        try:
            if not self.output_dir.exists():
                self.output_dir.mkdir(parents=True)

            sio.savemat(self.output_dir / 'cpu_load_var.mat', {'cpu_load_var': self.logs['cpu_load_var']})
            sio.savemat(self.output_dir / 'memory_load_var.mat', {'memory_load_var': self.logs['memory_load_var']})
            sio.savemat(self.output_dir / 'bandwidth_load_var.mat',
                        {'bandwidth_load_var': self.logs['bandwidth_load_var']})
            sio.savemat(self.output_dir / 'blocking_rate.mat', {'blocking_rate': self.logs['blocking_rate']})
            sio.savemat(self.output_dir / 'cpu_resource_comp.mat',
                        {'cpu_resource_comp': self.logs['cpu_resource_comp']})
            sio.savemat(self.output_dir / 'memory_resource_comp.mat',
                        {'memory_resource_comp': self.logs['memory_resource_comp']})
            sio.savemat(self.output_dir / 'bandwidth_resource_comp.mat',
                        {'bandwidth_resource_comp': self.logs['bandwidth_resource_comp']})
            sio.savemat(self.output_dir / 'CPU_status.mat', {'CPU_status': self.C})
            sio.savemat(self.output_dir / 'Bandwidth_status.mat', {'Bandwidth_status': self.B})
            sio.savemat(self.output_dir / 'Memory_status.mat', {'Memory_status': self.M})

            print("✅ 结果保存完成")
        except Exception as e:
            print(f"❌ 保存失败: {e}")

    def run(self):
        """运行基础仿真 (不使用DRL)"""
        print("--- 开始基础仿真 ---")

        for t in range(self.T):
            t1 = time.time()
            if (t + 1) % 10 == 0:
                print(f"时间步: {t + 1} / {self.T}")

            arrive_ids = self.events[t]['arrive']
            leave_ids = self.events[t]['leave']

            # 复制上一时刻状态
            if t > 0:
                self.B[t] = self.B[t - 1].copy()
                self.C[t] = self.C[t - 1].copy()
                self.M[t] = self.M[t - 1].copy()

            # 处理事件
            if leave_ids is not None and leave_ids.size > 0:
                self._handle_leave_event(t, leave_ids)

            if arrive_ids is not None and arrive_ids.size > 0:
                self._handle_arrive_event(t, arrive_ids)

            self._log_metrics(t, t1)

        # 打印结果
        blocking_rate = self.block / self.arrived if self.arrived > 0 else 0
        acceptance_rate = 1 - blocking_rate

        print(f"\n--- 仿真完成 ---")
        print(f"总执行时间: {np.sum(self.logs['duration']):.2f} 秒")
        print(f"总到达请求: {self.arrived}")
        print(f"总阻塞请求: {self.block}")
        print(f"阻塞率: {blocking_rate:.4f}")
        print(f"接受率: {acceptance_rate:.4f}")

        self._save_results()


# ==============================================================
# ✅ DRL Gym 环境 (仅在有 gym 库时定义)
# ==============================================================
if HAS_DRL:
    class SFCEnv(gym.Env):
        """DRL 高层调度环境: 学习部署顺序"""
        metadata = {'render.modes': ['human']}

        def __init__(self, simulator: DynamicSimulator):
            super(SFCEnv, self).__init__()
            self.sim = simulator
            self.t = 0
            self.pending_requests = []
            self.max_pending = 10

            self.action_space = spaces.Discrete(self.max_pending)

            n, L, K = self.sim.solver.node_num, self.sim.solver.link_num, self.sim.solver.type_num
            obs_shape = (n + n + L + (n * K) + L + self.max_pending,)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

            self.MAX_COST = 15000.0

        def _get_state(self):
            """编码当前状态"""
            state = self.sim.get_state(self.t)
            cpu_norm = state['cpu'] / self.sim.solver.cpu_capacity
            mem_norm = state['mem'] / self.sim.solver.memory_capacity
            bw_norm = state['bw'] / self.sim.solver.bandwidth_capacity
            hvt_norm = state['hvt'].flatten() / 10.0
            bw_ref_norm = state['bw_ref_count'] / 10.0
            pending_vec = np.zeros(self.max_pending, dtype=np.float32)
            pending_vec[:len(self.pending_requests)] = 1.0

            return np.concatenate([cpu_norm, mem_norm, bw_norm, hvt_norm, bw_ref_norm, pending_vec]).astype(np.float32)

        def reset(self):
            """重置环境"""
            print("--- Episode 重置 ---")
            self.t = 0
            self.sim.B[:] = self.sim.solver.bandwidth_capacity
            self.sim.C[:] = self.sim.solver.cpu_capacity
            self.sim.M[:] = self.sim.solver.memory_capacity
            self.sim.hvt_all[:] = 0
            self.sim.link_ref_count[:] = 0
            self.sim.served_requests, self.sim.served_trees = [], []
            self.sim.block, self.sim.arrived = 0, 0

            self.pending_requests = list(self.sim.events[0]['arrive'])
            self.sim.arrived += len(self.pending_requests)

            return self._get_state()

        def step(self, action: int):
            """✅ 修复: 执行一个动作"""
            # 自动跳过空时间步
            while len(self.pending_requests) == 0:
                self.t += 1

                if self.t >= self.sim.T:
                    final_br = self.sim.block / self.sim.arrived if self.sim.arrived > 0 else 0
                    print(f"--- Episode 结束 --- 阻塞率: {final_br:.4f}")
                    return self._get_state(), 0.0, True, {"episode_blocked": self.sim.block}

                # 处理离开事件
                leave_ids = self.sim.events[self.t]['leave']
                if leave_ids is not None and leave_ids.size > 0:
                    self.sim._handle_leave_event(self.t, leave_ids)

                # 加载新请求
                self.pending_requests = list(self.sim.events[self.t]['arrive'])
                self.sim.arrived += len(self.pending_requests)

            # 检查非法动作
            if action >= len(self.pending_requests):
                return self._get_state(), -10.0, False, {"error": "Invalid action"}

            req_id = self.pending_requests.pop(action)
            if req_id not in self.sim.req_map:
                return self._get_state(), -10.0, False, {"error": "Request not found"}

            req = self.sim.req_map[req_id]
            plan = self.sim.solver.solve_request(req, self.sim.get_state(self.t))

            # 归一化奖励
            if plan is None:
                reward = -100.0
                self.sim.block += 1
            else:
                self.sim._apply_deployment(self.t, req, plan)
                self.sim.served_requests.append(req)
                self.sim.served_trees.append(plan)

                cost = plan['cost_bw'] + plan['cost_cpu'] + plan['cost_mem']
                norm_cost = cost / self.MAX_COST
                reward = 1.0 - norm_cost

            # ✅ 修复: 正确设置 done 标志
            done = (self.t >= self.sim.T - 1 and len(self.pending_requests) == 0)

            return self._get_state(), reward, done, {}

        def render(self, mode='human'):
            print(
                f"[T={self.t}] 待办:{len(self.pending_requests)}, 已服务:{len(self.sim.served_requests)}, 阻塞:{self.sim.block}")

        def close(self):
            print("环境关闭")


# ==============================================================
# ✅ 主程序入口
# ==============================================================
def run_baseline(input_dir: Path, output_dir: Path, topo: np.ndarray,
                 dc_nodes: List[int], capacities: Dict):
    """运行基础仿真 (不使用 DRL)"""
    print("\n=== 运行基础仿真模式 ===")
    sim = DynamicSimulator(input_dir, output_dir, topo, dc_nodes, capacities)
    sim.run()


def train_drl(input_dir: Path, output_dir: Path, topo: np.ndarray,
              dc_nodes: List[int], capacities: Dict):
    """运行 DRL 训练"""
    if not HAS_DRL:
        print("❌ 错误: 缺少 DRL 库 (gym, stable-baselines3)")
        print("   请运行: pip install gym stable-baselines3")
        return

    print("\n=== 运行 DRL 训练模式 ===")

    # 初始化仿真器和环境
    sim = DynamicSimulator(input_dir, output_dir, topo, dc_nodes, capacities)
    env = SFCEnv(sim)
    vec_env = DummyVecEnv([lambda: env])

    # 初始化 PPO 智能体
    log_path = output_dir / "ppo_logs"
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(log_path))

    # 添加检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="ppo_sfc"
    )

    # 训练
    print("\n--- 开始训练 ---")
    model.learn(total_timesteps=50000, callback=checkpoint_callback)
    print("--- 训练完成 ---")

    # 保存模型
    model_path = output_dir / "ppo_sfc_final"
    model.save(str(model_path))
    print(f"模型已保存: {model_path}.zip")

    # 评估
    print("\n--- 评估模型 ---")
    obs = vec_env.reset()
    episode_reward = 0
    done = False

    # --- 新增: 导入 tqdm ---
    # (请确保您在文件顶部导入了: from tqdm import tqdm)
    from tqdm import tqdm

    # --- 新增: 从环境中获取总时间步长 T ---
    # (我们假设 T 在您的环境中是 721)
    try:
        sim_T = vec_env.envs[0].sim.T
    except AttributeError:
        print("警告：无法自动获取 sim.T，进度条最大值将设为默认值 (721)")
        sim_T = 721  # 备用值

    # --- 新增: 初始化 tqdm 进度条 ---
    pbar = tqdm(total=sim_T, desc="评估进度")
    last_t = 0  # 跟踪上一个时间步

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        episode_reward += reward[0]

        # --- 新增: 更新进度条 ---
        # 1. 从环境中获取当前模拟时间
        current_t = vec_env.envs[0].t

        # 2. 检查时间是否推进
        if current_t > last_t:
            pbar.update(current_t - last_t)  # 按实际跳过的时间步更新
            last_t = current_t

    # --- 新增: 确保进度条在循环结束时关闭 ---
    if pbar.n < sim_T:  # 补全剩余进度
        pbar.update(sim_T - pbar.n)
    pbar.close()

    # --- 原始代码继续 ---
    final_sim = vec_env.envs[0].sim
    final_br = final_sim.block / final_sim.arrived if final_sim.arrived > 0 else 0

    print(f"评估完成 - 总奖励: {episode_reward:.2f}")
    print(f"最终阻塞率: {final_br:.4f}")

    # 保存仿真日志
    final_sim._save_results()

def main():
    """主函数"""
    import sys

    # 配置
    INPUT_DIR = Path('E:/pycharmworkspace/SFC-master/mph/mat')
    OUTPUT_DIR = Path('E:/pycharmworkspace/SFC-master/mph/out')

    CAPACITIES = {
        'cpu': 2000.0,
        'memory': 1100.0,
        'bandwidth': 500.0
    }

    DC_NODES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 17, 18, 19, 20, 21, 23, 24, 27, 28]

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

    # 解析命令行参数
    try:
        if len(sys.argv) > 1 and sys.argv[1] == '--baseline':
            run_baseline(INPUT_DIR, OUTPUT_DIR, TOPOLOGY_MATRIX, DC_NODES, CAPACITIES)
        else:
            train_drl(INPUT_DIR, OUTPUT_DIR, TOPOLOGY_MATRIX, DC_NODES, CAPACITIES)
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}")
        print(f"   请确保以下文件存在于 {INPUT_DIR}:")
        print("   - US_Backbone_path.mat")
        print("   - sorted_requests.mat")
        print("   - event_list.mat")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()