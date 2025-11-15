#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : expert_msfce.py

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any


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
            print(f"致命错误: 找不到 MAT 文件 {path_db_file}")
            raise
        except KeyError:
            print(f"致命错误: MAT 文件 {path_db_file} 中缺少 'Paths' 键")
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
        print(f"专家加载: {self.node_num}节点, {self.link_num}链路, {self.dc_num}DC")

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
            # ✅ 修复: 添加你的 Debug 探针
            print(
                f"DEBUG_EVAL FAIL (goal={dest}): Not enough DC nodes on path. Found {len(usable)}, Need {len(request['vnf'])}")
            return 0, path, tree, hvt_new, False, dest, 0
        for lid in links:
            if lid - 1 >= len(bw) or bw[lid - 1] < request['bw_origin']:
                # ✅ 修复: 添加你的 Debug 探针
                print(f"DEBUG_EVAL FAIL (goal={dest}): link {lid} bw={bw[lid - 1]} < req_bw={request['bw_origin']}")
                return 0, path, tree, hvt_new, False, dest, 0

        # VNF 放置
        j, i = 0, 0
        while j < len(request['vnf']):
            if i >= len(usable):
                print(f"DEBUG_EVAL FAIL (goal={dest}): Ran out of usable DC nodes while placing VNF {j}")
                return 0, path, tree, hvt_new, False, dest, 0
            node, vnf_t = usable[i] - 1, request['vnf'][j] - 1
            if hvt[node, vnf_t] == 0:
                if cpu[node] < request['cpu_origin'][j] or mem[node] < request['memory_origin'][j]:
                    # ✅ 修复: 添加你的 Debug 探针
                    print(
                        f"DEBUG_EVAL FAIL (goal={dest}): node {node + 1} cpu/mem not enough. Has C={cpu[node]}, M={mem[node]}. Need C={request['cpu_origin'][j]}, M={request['memory_origin'][j]}")
                    i += 1
                    continue
            hvt_new[node, vnf_t] = 1
            j, i = j + 1, i + 1

        if np.sum(hvt_new) != len(request['vnf']):
            print(f"DEBUG_EVAL FAIL (goal={dest}): VNF placement count mismatch.")
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
            print(f"DEBUG_EVAL1 FAIL (goal={dest_node}): Loop detected (arr1 & arr2)")
            return 0, paths, tree, hvt, False, dest_node, 0
        arr4 = nodes_on_tree - set(tree_paths)
        if arr1 & arr4:
            print(f"DEBUG_EVAL1 FAIL (goal={dest_node}): Loop detected (arr1 & arr4)")
            return 0, paths, tree, hvt, False, dest_node, 0
        if i_idx + 1 < len(tree1_path):
            arr6 = set(tree1_path[i_idx + 1:])
            if arr1 & arr6:
                print(f"DEBUG_EVAL1 FAIL (goal={dest_node}): Loop detected (arr1 & arr6)")
                return 0, paths, tree, hvt, False, dest_node, 0

        usable_on_path = [n for n in paths[1:] if n in self.DC]
        deployed_on_path = [n for n in tree_paths if n in self.DC]

        for lid in links:
            if lid - 1 >= len(state['bw']) or state['bw'][lid - 1] < request['bw_origin']:
                # ✅ 修复: 添加你的 Debug 探针
                print(
                    f"DEBUG_EVAL1 FAIL (goal={dest_node}): link {lid} bw={state['bw'][lid - 1]} < req_bw={request['bw_origin']}")
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
                print(
                    f"DEBUG_EVAL1 FAIL (goal={dest_node}): Not enough DC nodes on branch. Found {len(usable_on_path)}, Need {undeployed_vnf}")
                return 0, paths, tree, hvt, False, dest_node, 0

            j, g = shared_path_deployed, 0
            while j < len(request['vnf']) and g < len(usable_on_path):
                node_idx = usable_on_path[g] - 1
                vnf_type = request['vnf'][j] - 1
                if hvt[node_idx, vnf_type] == 0:
                    if (state['cpu'][node_idx] < request['cpu_origin'][j] or
                            state['mem'][node_idx] < request['memory_origin'][j]):
                        # ✅ 修复: 添加你的 Debug 探针
                        print(
                            f"DEBUG_EVAL1 FAIL (goal={dest_node}): node {node_idx + 1} cpu/mem not enough. Has C={state['cpu'][node_idx]}, M={state['mem'][node_idx]}. Need C={request['cpu_origin'][j]}, M={request['memory_origin'][j]}")
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
                print(
                    f"DEBUG_EVAL1 FAIL (goal={dest_node}): VNF placement count mismatch. Deployed {total_deployed}, Need {len(request['vnf'])}")
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
    # 修复 #1: 替换为您的新 _calculate_cost 方法
    # ----------------------------------------------------
    def _calculate_cost(self, request: Dict, state: Dict, tree: np.ndarray, hvt: np.ndarray) -> float:
        """计算部署此方案的资源成本 (用于RL奖励)"""
        bw_cost, cpu_cost, mem_cost = 0, 0, 0

        used_links = np.where(tree > 0)[0]
        if used_links.size > 0:
            bw_ref_count = state.get('bw_ref_count', np.zeros(self.link_num))  # 安全获取
            new_links_mask = (bw_ref_count[used_links] == 0)
            bw_cost = np.sum(new_links_mask) * request['bw_origin']

        for node, vnf_t in np.argwhere(hvt > 0):
            if state['hvt'][node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    cpu_cost += request['cpu_origin'][j]
                    mem_cost += request['memory_origin'][j]
                except ValueError:
                    pass

        # 修复: 定义归一化权重
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

        # ✅ 新增: 跟踪可行的目的地
        feasible_dests = []
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
        # 新增: 标记可行的目的地
            if best_result and best_result.get('eval', -1) > 0:
                feasible_dests.append(d_idx)
        # 修复: 如果没有可行的目的地,返回空解
        if not feasible_dests:
            print(f"⚠️ 请求 {request['id']} 的所有目的地都不可行 (DC节点不足)")
            return None, []
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
        # 修复 #3: (补丁 3)
        # ----------------------------------------------------

        # 创建局部状态副本 (已应用补丁 3)
        local_network_state = {
            'bw': network_state['bw'].copy(),
            'cpu': network_state['cpu'].copy(),
            'mem': network_state['mem'].copy(),
            'hvt': network_state['hvt'].copy(),
            'bw_ref_count': network_state.get('bw_ref_count', np.zeros(len(network_state['bw']))).copy(),
            'request': request
        }

        while unadded:
            best_eval, best_plan, best_d, best_action, best_cost = -1, None, -1, (0, 0), 0

            for d_idx in unadded:
                # 遍历所有已在树上的路径，看从哪里连接
                for conn_path in current_tree['paths_map'].values():
                    t, m, action, cost = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, local_network_state, nodes_on_tree  # 使用本地状态
                    )
                    if t.get('feasible') and m > best_eval:
                        best_eval, best_plan, best_d = m, t, d_idx
                        best_action, best_cost = action, cost

            if best_d == -1:
                # 修复: (方案 2) 停止尝试分支，但返回已经成功的主干轨迹
                break  # 阻塞

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

            # 修复 #4: 临时在 local_network_state 上应用资源变化
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
                # 修复 #3 (补丁 3): 确保 HVT 是累加的
                local_network_state['hvt'][node, vnf_t] += 1

        return current_tree, expert_trajectory