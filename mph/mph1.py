#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : msfce_simulator_fixed.py
# @Date    : 2025-11-06
# @Author  : gxr (Fixed by Gemini)
# @Desc    : 完全修复的 MSFC-CE 仿真器 + DRL 环境
#            (修复了 'arrival_time' 和 Gymnasium 警告)

import numpy as np
import scipy.io as sio
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import sys

# === DRL 库导入 (可选) ===
try:
    # ✅ 修正 (建议): 升级到 Gymnasium
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    HAS_DRL = True
except ImportError:
    HAS_DRL = False
    print("⚠️ 警告: 未安装 DRL 库 (gymnasium, stable-baselines3)")
    print("   将只运行基础仿真 (不使用强化学习)")


# ==============================================================
# ✅ .mat 文件解析
# ==============================================================
def parse_mat_request(req_obj) -> Dict:
    """将 MATLAB 请求结构解析为 Python 字典"""

    # 修正: req_obj 已经是 1D 数组中的单个对象
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
            # ✅ 修正 (建议 1):
            # 'arrival_time' 和 'leave_time' 不在该文件中
            # 'arrival_time': int(req['arrival_time'][0, 0]),
            # 'leave_time': int(req['leave_time'][0, 0]),
        }
    except (IndexError, TypeError, KeyError):
        # 兼容备用格式
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

    def _calc_eval(self, request: Dict, d_idx: int, k: int, network_state: Dict):
        """评估 S->d 的第k条路径"""
        bw, cpu, mem, hvt = network_state['bw'], network_state['cpu'], network_state['mem'], network_state['hvt']
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

    def _calc_eval1(self, d_idx: int, k: int, i_idx: int, tree1_node_path: List[int],
                    request: Dict, tree1_hvt: np.ndarray,
                    network_state: Dict, node_on_tree: Set[int]):
        """评估新树枝 (连接点->d)"""
        bw, cpu, mem, hvt_all = network_state['bw'], network_state['cpu'], network_state['mem'], network_state['hvt']

        hvt = tree1_hvt.copy()
        tree = np.zeros(self.link_num)  # 只评估新树枝

        conn_node = tree1_node_path[i_idx]
        dest = request['dest'][d_idx]

        path, dist, links = self._get_path_from_db(conn_node, dest, k)
        tree_paths_to_connector = tree1_node_path[:i_idx + 1]

        if not path:
            return 0, [], tree, hvt, tree_paths_to_connector, False, dest

        new_path_nodes = set(path[1:])
        connector_path_nodes = set(tree_paths_to_connector)

        if new_path_nodes.intersection(connector_path_nodes) or \
                new_path_nodes.intersection(node_on_tree - connector_path_nodes):
            return 0, path, tree, hvt, tree_paths_to_connector, False, dest  # 环路

        for link_id in links:
            if bw[link_id - 1] < request['bw_origin']:
                return 0, path, tree, hvt, tree_paths_to_connector, False, dest  # 带宽不足

        usable_on_new_path = [n for n in path[1:] if n in self.DC]

        # VNF 检查
        shared_path_nodes = [n for n in tree_paths_to_connector if n in self.DC]
        shared_vnf_count = 0
        vnf_list_0based = np.array(request['vnf']) - 1
        for node_1based in shared_path_nodes:
            if node_1based - 1 < hvt.shape[0]:
                shared_vnf_count += np.sum(tree1_hvt[node_1based - 1, vnf_list_0based])

        undeployed_vnf_count = len(request['vnf']) - shared_vnf_count

        if undeployed_vnf_count > 0:
            if len(usable_on_new_path) < undeployed_vnf_count:
                return 0, path, tree, hvt, tree_paths_to_connector, False, dest  # DC不足

            j = int(shared_vnf_count)
            i = 0
            while j < len(request['vnf']):
                if i >= len(usable_on_new_path):
                    return 0, path, tree, hvt, tree_paths_to_connector, False, dest  # DC不足

                dc_node_1based = usable_on_new_path[i]
                dc_node_0based = dc_node_1based - 1
                vnf_type_1based = request['vnf'][j]
                vnf_type_0based = vnf_type_1based - 1

                if hvt_all[dc_node_0based, vnf_type_0based] == 0:
                    cpu_req = request['cpu_origin'][j]
                    mem_req = request['memory_origin'][j]
                    if (cpu[dc_node_0based] < cpu_req) or (mem[dc_node_0based] < mem_req):
                        i += 1
                        continue

                hvt[dc_node_0based, vnf_type_0based] = 1
                j += 1
                i += 1

            deployed_count_on_new_path = np.sum(hvt) - np.sum(tree1_hvt)
            if deployed_count_on_new_path != undeployed_vnf_count:
                return 0, path, tree, hvt, tree_paths_to_connector, False, dest  # 逻辑错误

        # 可行
        for l_id in links:
            tree[l_id - 1] = 1

        path_indices_0based = np.array(path[1:]) - 1 if len(path) > 1 else np.array([])
        CPU_status_sum = np.sum(cpu[path_indices_0based])
        Memory_status_sum = np.sum(mem[path_indices_0based])
        Bandwidth_status_sum = np.sum(bw[np.array(links) - 1]) if links else 0
        max_k_dist = self._get_kth_path_max_distance(conn_node, dest, self.k_path_count)

        score = (
                (1 - dist / max_k_dist if max_k_dist > 0 else 1) +
                (len(usable_on_new_path) / self.dc_num) +
                (CPU_status_sum / (self.cpu_capacity * self.dc_num)) +
                (Memory_status_sum / (self.memory_capacity * self.dc_num)) +
                (Bandwidth_status_sum / (self.bandwidth_capacity * self.link_num))
        )

        return score, path, tree, hvt, tree_paths_to_connector, True, dest

    def _calc_atnp(self, tree1: Dict, tree1_node_path: List[int], d_idx: int,
                   network_state: Dict, node_on_tree: Set[int]):
        """找到最佳连接点"""
        eval_set = []
        temp_feasible_set = []
        k_path_results = {}
        dest_node_1based = network_state['request']['dest'][d_idx]

        for i_idx, conn_node_1based in enumerate(tree1_node_path):
            temp_eval_k = []
            temp_feasible_k = []
            k_path_results[i_idx] = []

            for k in range(1, self.k_path_count + 1):
                eval_val, paths, tree, hvt, tree_paths, feasible, infeasible_dest = self._calc_eval1(
                    d_idx, k, i_idx, tree1_node_path, network_state['request'], tree1['hvt'],
                    network_state, node_on_tree
                )
                temp_eval_k.append(eval_val)
                temp_feasible_k.append(feasible)
                k_path_results[i_idx].append({
                    'eval': eval_val, 'paths': paths, 'tree': tree, 'hvt': hvt,
                    'tree_paths': tree_paths, 'feasible': feasible
                })

            eval_set.append(np.max(temp_eval_k))
            temp_feasible_set.append(np.min(temp_feasible_k))

        if not eval_set or np.max(eval_set) <= 0:
            return {'feasible': False}, -1

        best_conn_node_idx = np.argmax(eval_set)

        r = 1.0  # 可行
        if best_conn_node_idx == 0 and len(temp_feasible_set) > 0:
            r = np.min(temp_feasible_set)
            if r == 0:
                best_conn_node_idx = np.argmin(temp_feasible_set)

        best_k_idx = np.argmax([res['eval'] for res in k_path_results[best_conn_node_idx]])
        best_result = k_path_results[best_conn_node_idx][best_k_idx]
        m = best_result['eval']

        t = {
            'tree': best_result['tree'],
            'hvt': best_result['hvt'],
            'feasible': best_result['feasible'],
            'new_path_full': best_result['tree_paths'] + best_result['paths'][1:]
        }

        return t, m

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
                    request, d_idx, k, network_state
                )
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
                best_eval_for_d, best_tree_for_d = -1, None
                for conn_path in current_tree['paths_map'].values():
                    t, m = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, network_state, nodes_on_tree
                    )
                    if t.get('feasible') and m > best_eval_for_d:
                        best_eval_for_d, best_tree_for_d = m, t

                if best_eval_for_d > best_eval:
                    best_eval, best_plan, best_d = best_eval_for_d, best_tree_for_d, d_idx

            if best_d == -1:
                return None

            current_tree['tree'] = np.logical_or(current_tree['tree'], best_plan['tree']).astype(float)
            current_tree['hvt'] = np.logical_or(current_tree['hvt'], best_plan['hvt']).astype(float)
            current_tree['paths_map'][request['dest'][best_d]] = best_plan['new_path_full']
            nodes_on_tree.update(best_plan['new_path_full'])
            unadded.remove(best_d)

        # 修复: 计算带宽成本 (只计算新占用的链路)
        used_links = np.where(current_tree['tree'] > 0)[0]
        new_links_mask = (network_state['bw_ref_count'][used_links] == 0)
        bw_cost = np.sum(new_links_mask) * request['bw_origin']

        # 计算 VNF 成本 (只计算新部署的)
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
        print(f"加载数据从 {self.input_dir}...")
        reqs_mat = sio.loadmat(input_dir / "sorted_requests.mat")['sorted_requests']
        # ✅ 修正 (针对):
        # 'reqs_mat' 是 (N, 1) 数组, 必须 .flatten()
        self.requests = [parse_mat_request(r) for r in reqs_mat.flatten()]
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
        """处理离开事件 (使用引用计数释放资源)"""
        if leave_ids.size == 0:
            return

        leave_set = set(leave_ids)
        remaining_reqs = []

        for req in self.served_requests:
            if req['id'] in leave_set:
                try:
                    tree = next(tr for tr in self.served_trees if tr['id'] == req['id'])
                    # 1. 释放带宽 (引用计数)
                    for link_idx in np.where(tree['tree'] > 0)[0]:
                        if self.link_ref_count[link_idx] > 0:
                            self.link_ref_count[link_idx] -= 1
                            if self.link_ref_count[link_idx] == 0:
                                self.B[t, link_idx] += req['bw_origin']

                    # 2. 释放 VNF (引用计数)
                    for node, vnf_t in np.argwhere(tree['hvt'] > 0):
                        if self.hvt_all[node, vnf_t] > 0:
                            self.hvt_all[node, vnf_t] -= 1
                            if self.hvt_all[node, vnf_t] == 0:
                                try:
                                    j = req['vnf'].index(vnf_t + 1)
                                    self.C[t, node] += req['cpu_origin'][j]
                                    self.M[t, node] += req['memory_origin'][j]
                                except ValueError:
                                    pass
                except StopIteration:
                    print(f"⚠️ 找不到 ID={req['id']} 的树")
            else:
                remaining_reqs.append(req)

        self.served_requests = remaining_reqs
        self.served_trees = [t for t in self.served_trees if t['id'] in {r['id'] for r in remaining_reqs}]

    def _apply_deployment(self, t: int, request: Dict, plan: Dict):
        """应用部署方案 (占用资源,使用引用计数)"""
        # 1. 占用带宽 (引用计数)
        for link_idx in np.where(plan['tree'] > 0)[0]:
            if self.link_ref_count[link_idx] == 0:
                self.B[t, link_idx] -= request['bw_origin']
            self.link_ref_count[link_idx] += 1

        # 2. 占用 VNF (引用计数)
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
        """处理到达事件 (基础启发式版本)"""
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

    def run_baseline(self):
        """运行基础仿真 (不使用DRL)"""
        print("--- 开始基础仿真 (MSFC-CE 启发式) ---")
        for t in range(self.T):
            t1 = time.time()
            if (t + 1) % 10 == 0:
                print(f"时间步: {t + 1} / {self.T}")

            # 复制状态
            if t > 0:
                self.B[t], self.C[t], self.M[t] = self.B[t - 1].copy(), self.C[t - 1].copy(), self.M[t - 1].copy()

            # 处理事件
            leave_ids = self.events[t]['leave']
            if leave_ids is not None and leave_ids.size > 0:
                self._handle_leave_event(t, leave_ids)

            arrive_ids = self.events[t]['arrive']
            if arrive_ids is not None and arrive_ids.size > 0:
                self._handle_arrive_event(t, arrive_ids)

            self._log_metrics(t, t1)

        # 打印结果
        br = self.block / self.arrived if self.arrived > 0 else 0
        print(f"\n--- 仿真完成 ---")
        print(f"总时间: {np.sum(self.logs['duration']):.2f}s, 总请求: {self.arrived}, 总阻塞: {self.block}")
        print(f"阻塞率: {br:.4f}, 接受率: {1 - br:.4f}")
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
            # 状态: [CPU, Mem, BW, HVT(计数), BW(计数), 待办Mask]
            obs_shape = (n + n + L + (n * K) + L + self.max_pending,)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs_shape, dtype=np.float32)

            # ✅ (建议 3) 奖励归一化
            self.MAX_COST = 15000.0
            self.REWARD_BLOCK = -100.0
            self.REWARD_INVALID = -10.0

        def _get_state(self) -> np.ndarray:
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

        def reset(self, *, seed=None, options=None):  # ✅ 修正: 兼容新版 gym
            """重置环境"""
            super().reset(seed=seed)  # ✅ 修正: 兼容新版 gym
            print("--- DRL Episode 重置 (T=0) ---")
            self.t = 0
            self.sim.B[:] = self.sim.solver.bandwidth_capacity
            self.sim.C[:] = self.sim.solver.cpu_capacity
            self.sim.M[:] = self.sim.solver.memory_capacity
            self.sim.hvt_all[:] = 0
            self.sim.link_ref_count[:] = 0
            self.sim.served_requests, self.sim.served_trees = [], []
            self.sim.block, self.sim.arrived = 0, 0

            for key in self.sim.logs:
                self.sim.logs[key][:] = 0

            self.pending_requests = list(self.sim.events[0]['arrive'])
            self.sim.arrived += len(self.pending_requests)

            return self._get_state(), {}  # ✅ 修正: 兼容新版 gym (返回 state, info)

        def step(self, action: int):
            """执行一个动作"""

            # ✅ (建议 1) 修正 Episode 终止逻辑
            while len(self.pending_requests) == 0:
                self.t += 1
                if self.t >= self.sim.T:
                    br = self.sim.block / self.sim.arrived if self.sim.arrived > 0 else 0
                    print(f"--- DRL Episode 结束 --- 阻塞率: {br:.4f}")
                    # ✅ 修正: 兼容新版 gym (返回 state, reward, terminated, truncated, info)
                    return self._get_state(), 0.0, True, False, {"episode_blocked": self.sim.block}

                leave_ids = self.sim.events[self.t]['leave']
                if leave_ids is not None and leave_ids.size > 0:
                    self.sim._handle_leave_event(self.t, leave_ids)

                self.pending_requests = list(self.sim.events[self.t]['arrive'])
                self.sim.arrived += len(self.pending_requests)

            # ✅ (建议 2) 处理非法动作
            if action >= len(self.pending_requests):
                return self._get_state(), self.REWARD_INVALID, False, False, {"error": "Invalid action"}

            req_id = self.pending_requests.pop(action)
            if req_id not in self.sim.req_map:
                return self._get_state(), self.REWARD_INVALID, False, False, {"error": "Request not found"}

            req = self.sim.req_map[req_id]
            plan = self.sim.solver.solve_request(req, self.sim.get_state(self.t))

            # ✅ (建议 3) 修正奖励函数
            if plan is None:
                reward = self.REWARD_BLOCK
                self.sim.block += 1
            else:
                self.sim._apply_deployment(self.t, req, plan)
                self.sim.served_requests.append(req)
                self.sim.served_trees.append(plan)

                cost = plan['cost_bw'] + plan['cost_cpu'] + plan['cost_mem']
                norm_cost = cost / self.MAX_COST
                reward = 1.0 - norm_cost

            return self._get_state(), reward, False, False, {}  # ✅ 修正: 兼容新版 gym

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
    try:
        sim = DynamicSimulator(input_dir, output_dir, topo, dc_nodes, capacities)
        sim.run_baseline()
    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}\n   请检查路径: {input_dir}")
    except Exception as e:
        print(f"❌ 仿真时发生未知错误: {e}")
        import traceback
        traceback.print_exc()


def train_drl(input_dir: Path, output_dir: Path, topo: np.ndarray,
              dc_nodes: List[int], capacities: Dict):
    """运行 DRL 训练"""
    if not HAS_DRL:
        print("❌ 错误: 缺少 DRL 库 (gymnasium, stable-baselines3)")
        print("   请运行: pip install gymnasium stable-baselines3")
        return

    print("\n=== 运行 DRL 训练模式 ===")
    try:
        # 1. 初始化仿真器和环境
        sim = DynamicSimulator(input_dir, output_dir, topo, dc_nodes, capacities)
        env = SFCEnv(sim)
        vec_env = DummyVecEnv([lambda: env])  # SB3 需要一个 VecEnv

        # 2. 初始化 PPO 智能体
        log_path = output_dir / "ppo_logs"
        model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=str(log_path))

        # 3. 添加检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(output_dir / "checkpoints"),
            name_prefix="ppo_sfc"
        )

        # 4. 训练
        print("\n--- 开始训练 (50000 步) ---")
        model.learn(total_timesteps=50000, callback=checkpoint_callback)
        print("--- 训练完成 ---")

        # 5. 保存模型
        model_path = output_dir / "ppo_sfc_final"
        model.save(str(model_path))
        print(f"模型已保存: {model_path}.zip")

        # 6. 评估
        print("\n--- 评估模型 ---")
        obs, info = vec_env.reset()  # ✅ 修正: 兼容新版 gym
        episode_reward = 0

        terminated, truncated = False, False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = vec_env.step(action)
            episode_reward += reward[0]
            terminated = terminated[0]
            truncated = truncated[0]

        final_sim = vec_env.envs[0].sim
        final_br = final_sim.block / final_sim.arrived if final_sim.arrived > 0 else 0

        print(f"评估完成 - 总奖励: {episode_reward:.2f}")
        print(f"最终阻塞率: {final_br:.4f} (阻塞 {final_sim.block} / 到达 {final_sim.arrived})")

        # 7. 保存 DRL 运行的仿真日志
        final_sim._save_results()

    except FileNotFoundError as e:
        print(f"❌ 文件未找到: {e}\n   请检查路径: {input_dir}")
    except Exception as e:
        print(f"❌ DRL 训练时发生未知错误: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数 - 允许选择 'baseline' 或 'drl' 模式"""

    # --- 1. 配置 ---
    # ✅ (建议 4) 修正文件路径
    try:
        # 尝试使用 Linux/macOS 路径
        BASE_DIR = Path('/home/user1/PycharmProjects/SFC-master/mph')
        INPUT_DIR = BASE_DIR / 'mat'
        OUTPUT_DIR = BASE_DIR / 'out'
        if not INPUT_DIR.exists():
            raise FileNotFoundError
    except FileNotFoundError:
        # 回退到 Windows 路径
        BASE_DIR = Path('E:/pycharmworkspace/SFC-master/mph')
        INPUT_DIR = BASE_DIR / 'mat'
        OUTPUT_DIR = BASE_DIR / 'out'

    print(f"--- 使用输入路径: {INPUT_DIR} ---")

    # ✅ (建议 2) 修正硬编码容量
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

    # --- 启动仿真 ---
    try:
        # 解析命令行参数
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