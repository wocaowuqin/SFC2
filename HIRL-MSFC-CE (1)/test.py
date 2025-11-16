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
    MSFC-CE 启发式算法求解器 (对应论文第三章)
    这是专家预言机，用于生成模仿学习的标签
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

        # 论文3.3.4节的权重系数 (α, β, γ)
        self.alpha = 0.3  # 路径跳数权重
        self.beta = 0.3  # DC节点数权重
        self.gamma = 0.4  # 剩余资源权重

        print(f"专家加载: {self.node_num}节点, {self.link_num}链路, {self.dc_num}DC")

    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        """创建链路映射"""
        link_map, lid = {}, 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def _get_path_from_db(self, src: int, dst: int, k: int):
        """从路径数据库获取第k条最短路径"""
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
        """获取第k条路径的最大距离"""
        try:
            return int(self.path_db[src - 1, dst - 1]['pathsdistance'][kpath - 1][0])
        except Exception:
            return 1

    def _calc_score(self, src: int, dst: int, dist: int, dc_count: int,
                    cpu_sum: float, mem_sum: float, bw_sum: float) -> float:
        """
        计算评分函数 (论文公式未明确给出，这里使用综合评分)
        """
        max_dist = self._get_kth_path_max_distance(src, dst, self.k_path_count) or 1
        score = (
                (1 - dist / max_dist) +
                dc_count / self.dc_num +
                cpu_sum / (self.cpu_capacity * self.dc_num) +
                mem_sum / (self.memory_capacity * self.dc_num) +
                bw_sum / (self.bandwidth_capacity * self.link_num)
        )
        return score

    def _calculate_cost(self, request: Dict, state: Dict, tree: np.ndarray, hvt: np.ndarray) -> float:
        """
        计算部署此方案的资源成本 (用于RL奖励)
        对应论文公式(3-1)
        """
        bw_cost, cpu_cost, mem_cost = 0, 0, 0

        # 计算带宽成本
        used_links = np.where(tree > 0)[0]
        if used_links.size > 0:
            bw_ref_count = state.get('bw_ref_count', np.zeros(self.link_num))
            new_links_mask = (bw_ref_count[used_links] == 0)
            bw_cost = np.sum(new_links_mask) * request['bw_origin']

        # 计算节点CPU和内存成本
        for node, vnf_t in np.argwhere(hvt > 0):
            if state['hvt'][node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    cpu_cost += request['cpu_origin'][j]
                    mem_cost += request['memory_origin'][j]
                except ValueError:
                    pass

        # 归一化权重设计
        bw_weight = 1.0 / self.bandwidth_capacity
        cpu_weight = 10.0 / self.cpu_capacity
        mem_weight = 10.0 / self.memory_capacity

        # 总成本
        total_cost = (bw_cost * bw_weight) + (cpu_cost * cpu_weight) + (mem_cost * mem_weight)

        # 归一化到 [0, 10]
        max_possible_cost = (
                self.link_num * self.bandwidth_capacity * bw_weight +
                self.dc_num * self.cpu_capacity * cpu_weight +
                self.dc_num * self.memory_capacity * mem_weight
        )

        if max_possible_cost == 0:
            return 0.0

        normalized_cost = (total_cost / max_possible_cost) * 10.0
        return np.clip(normalized_cost, 0, 10)

    def _calc_path_eval_first(self, src: int, dst: int, k: int, state: Dict, request: Dict) -> float:
        """
        计算第一个目的节点路径评价指标
        对应论文公式(3-11)
        """
        path, dist, links = self._get_path_from_db(src, dst, k)
        if not path:
            return -1

        # 路径跳数归一化
        max_dist = self._get_kth_path_max_distance(src, dst, self.k_path_count) or 1
        hop_score = 1 - (dist / max_dist)

        # DC节点数归一化
        dc_nodes_on_path = [n for n in path if n in self.DC]
        dc_score = len(dc_nodes_on_path) / self.dc_num

        # 剩余资源归一化
        cpu_sum = sum(state['cpu'][n - 1] for n in dc_nodes_on_path)
        mem_sum = sum(state['mem'][n - 1] for n in dc_nodes_on_path)
        bw_sum = sum(state['bw'][lid - 1] for lid in links if lid - 1 < len(state['bw']))

        resource_score = (
                                 cpu_sum / (self.cpu_capacity * self.dc_num) +
                                 mem_sum / (self.memory_capacity * self.dc_num) +
                                 bw_sum / (self.bandwidth_capacity * self.link_num)
                         ) / 3.0

        # 综合评价 (论文公式3-11)
        eval_score = self.alpha * hop_score + self.beta * dc_score + self.gamma * resource_score
        return eval_score

    def _calc_path_eval_subsequent(self, src: int, connect_node: int, dst: int,
                                   k: int, state: Dict, request: Dict,
                                   tree_path: List[int]) -> float:
        """
        计算后续目的节点路径评价指标
        对应论文公式(3-12)
        """
        path, dist, links = self._get_path_from_db(connect_node, dst, k)
        if not path or len(path) < 2:
            return -1

        # 计算源到接入点的距离
        src_to_connect_dist = 0
        for i in range(len(tree_path) - 1):
            src_to_connect_dist += 1

        # 总跳数归一化
        max_dist_src = self._get_kth_path_max_distance(src, dst, self.k_path_count) or 1
        total_hops = src_to_connect_dist + dist
        hop_score = 1 - (total_hops / max_dist_src)

        # 源到接入点的DC节点数
        dc_on_shared = [n for n in tree_path if n in self.DC]
        dc_score = len(dc_on_shared) / self.dc_num

        # 接入点到目的节点的剩余资源
        dc_on_branch = [n for n in path[1:] if n in self.DC]
        cpu_sum = sum(state['cpu'][n - 1] for n in dc_on_branch)
        mem_sum = sum(state['mem'][n - 1] for n in dc_on_branch)
        bw_sum = sum(state['bw'][lid - 1] for lid in links if lid - 1 < len(state['bw']))

        resource_score = (
                                 cpu_sum / (self.cpu_capacity * self.dc_num) +
                                 mem_sum / (self.memory_capacity * self.dc_num) +
                                 bw_sum / (self.bandwidth_capacity * self.link_num)
                         ) / 3.0

        # 综合评价 (论文公式3-12)
        eval_score = self.alpha * hop_score + self.beta * dc_score + self.gamma * resource_score
        return eval_score

    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """
        评估第一个目的节点 S->d 的第k条路径
        对应论文3.4.1节的路径选择和VNF部署策略
        """
        bw, cpu, mem, hvt = state['bw'], state['cpu'], state['mem'], state['hvt']
        src, dest = request['source'], request['dest'][d_idx]

        path, dist, links = self._get_path_from_db(src, dest, k)
        if not path:
            return 0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dest, 0

        tree = np.zeros(self.link_num)
        hvt_new = np.zeros((self.node_num, self.type_num))
        usable = [n for n in path if n in self.DC]

        # 检查DC节点数量
        if len(usable) < len(request['vnf']):
            return 0, path, tree, hvt_new, False, dest, 0

        # 检查带宽资源
        for lid in links:
            if lid - 1 >= len(bw) or bw[lid - 1] < request['bw_origin']:
                return 0, path, tree, hvt_new, False, dest, 0

        # VNF 按序部署 (论文要求从距离源节点最近的DC开始)
        j, i = 0, 0
        while j < len(request['vnf']):
            if i >= len(usable):
                return 0, path, tree, hvt_new, False, dest, 0

            node, vnf_t = usable[i] - 1, request['vnf'][j] - 1

            # 如果该节点已部署该VNF，直接复用
            if hvt[node, vnf_t] == 0:
                # 检查资源
                if cpu[node] < request['cpu_origin'][j] or mem[node] < request['memory_origin'][j]:
                    i += 1
                    continue

            hvt_new[node, vnf_t] = 1
            j, i = j + 1, i + 1

        # 验证所有VNF已部署
        if np.sum(hvt_new) != len(request['vnf']):
            return 0, path, tree, hvt_new, False, dest, 0

        # 标记使用的链路
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
        """
        评估从树上第 i_idx 个节点到目的节点 d_idx 的第 k 条路径
        对应论文3.4.1节的后续节点接入策略
        """
        hvt = tree1_hvt.copy()
        tree = np.zeros(self.link_num)
        tree_paths = tree1_path[:i_idx + 1]

        connect_node = tree1_path[i_idx]
        dest_node = request['dest'][d_idx]

        paths, dist, links = self._get_path_from_db(connect_node, dest_node, k)

        if not paths or len(paths) < 2:
            return 0, [], tree, hvt, False, dest_node, 0

        # 破环策略: 检测环路 (论文3.4.1节策略4)
        arr1 = set(paths[1:])
        arr2 = set(tree_paths)
        if arr1 & arr2:  # 与已建树路径有重叠
            return 0, paths, tree, hvt, False, dest_node, 0

        arr4 = nodes_on_tree - set(tree_paths)
        if arr1 & arr4:  # 与树上其他路径有重叠
            return 0, paths, tree, hvt, False, dest_node, 0

        if i_idx + 1 < len(tree1_path):
            arr6 = set(tree1_path[i_idx + 1:])
            if arr1 & arr6:
                return 0, paths, tree, hvt, False, dest_node, 0

        # 可用DC节点
        usable_on_path = [n for n in paths[1:] if n in self.DC]
        deployed_on_path = [n for n in tree_paths if n in self.DC]

        # 检查带宽资源
        for lid in links:
            if lid - 1 >= len(state['bw']) or state['bw'][lid - 1] < request['bw_origin']:
                return 0, paths, tree, hvt, False, dest_node, 0

        # 计算已部署的VNF数量 (共享路径上)
        shared_path_deployed = sum(
            1 for vnf_type in request['vnf']
            if any(hvt[n - 1, vnf_type - 1] > 0 for n in deployed_on_path)
        )
        undeployed_vnf = len(request['vnf']) - shared_path_deployed

        # 如果所有VNF已在共享路径上部署
        if undeployed_vnf == 0:
            for lid in links:
                tree[lid - 1] = 1
            cost = self._calculate_cost(request, state, tree, hvt)

            # 计算评分
            CPU_status = sum(state['cpu'][n - 1] for n in paths[1:] if n in self.DC)
            Memory_status = sum(state['mem'][n - 1] for n in paths[1:] if n in self.DC)
            Bandwidth_status = sum(state['bw'][lid - 1] for lid in links)
            eval_score = self._calc_score(
                connect_node, dest_node, dist,
                len(deployed_on_path), CPU_status, Memory_status, Bandwidth_status
            )
            return eval_score, paths, tree, hvt, True, 0, cost

        # 需要在分支路径上部署VNF
        if len(usable_on_path) < undeployed_vnf:
            return 0, paths, tree, hvt, False, dest_node, 0

        # 部署剩余VNF (从已部署的数量开始)
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

        # 验证所有VNF已部署
        total_deployed = sum(
            1 for vnf_type in request['vnf']
            if any(hvt[n - 1, vnf_type - 1] > 0 for n in (deployed_on_path + usable_on_path))
        )
        if total_deployed != len(request['vnf']):
            return 0, paths, tree, hvt, False, dest_node, 0

        # 标记链路
        for lid in links:
            tree[lid - 1] = 1

        cost = self._calculate_cost(request, state, tree, hvt)

        # 计算评分
        CPU_status = sum(state['cpu'][n - 1] for n in paths[1:] if n in self.DC)
        Memory_status = sum(state['mem'][n - 1] for n in paths[1:] if n in self.DC)
        Bandwidth_status = sum(state['bw'][lid - 1] for lid in links)
        eval_score = self._calc_score(
            connect_node, dest_node, dist,
            len(usable_on_path), CPU_status, Memory_status, Bandwidth_status
        )

        return eval_score, paths, tree, hvt, True, 0, cost

    def _calc_atnp(self, tree1: Dict, tree1_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """
        找到将目的节点 d 连接到树 tree1 的最佳方案
        对应论文3.4.1节: 从候选目的节点集合中确定最优目的节点策略

        返回: (最佳方案, 最佳评估值, 最佳动作(i_idx, k_idx), 成本)
        """
        request = state['request']

        if tree1.get('eval', 0) == 0:
            return {
                'tree': tree1['tree'].copy(), 'hvt': tree1['hvt'].copy(),
                'feasible': tree1.get('feasible', False),
                'infeasible_dest': tree1.get('infeasible_dest', 0)
            }, 0, (0, 0), 0

        best_eval = -1
        best_plan = None
        best_action = (0, 0)
        best_cost = 0

        # 遍历树上的所有可能连接点
        for i_idx in range(len(tree1_path)):
            # 遍历 K 条路径
            for k in range(1, self.k_path_count + 1):
                eval_val, paths, tree_new, hvt_new, feasible, infeasible_dest, cost = \
                    self._calc_eval1(
                        d_idx, k, i_idx, tree1_path, request,
                        tree1['hvt'], state, nodes_on_tree
                    )

                if feasible and eval_val > best_eval:
                    best_eval = eval_val
                    best_action = (i_idx, k - 1)  # k-1 转为0-indexed
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

    def _evaluate_subsequent_tree(self, current_tree: Dict, candidate_d_idx: int,
                                  remaining_dests: Set[int], local_state: Dict,
                                  nodes_on_tree: Set[int], n_depth: int = 2) -> float:
        """
        评估候选节点加入后的后续树质量
        对应论文3.4.1节策略3: 评估后续树收益

        Args:
            current_tree: 当前树状态
            candidate_d_idx: 候选目的节点索引
            remaining_dests: 剩余未加入的目的节点集合
            local_state: 当前网络状态
            nodes_on_tree: 已在树上的节点集合
            n_depth: 后续树深度 (默认评估后续2个节点)

        Returns:
            后续树的平均评估值
        """
        request = local_state['request']

        # 找到候选节点加入树的最佳方案
        best_plan_for_candidate = None
        best_eval_for_candidate = -1
        best_path_for_candidate = None

        for conn_path in current_tree['paths_map'].values():
            plan, eval_val, action, cost = self._calc_atnp(
                {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                conn_path, candidate_d_idx, local_state, nodes_on_tree
            )
            if plan.get('feasible') and eval_val > best_eval_for_candidate:
                best_eval_for_candidate = eval_val
                best_plan_for_candidate = plan
                best_path_for_candidate = plan['new_path_full']

        if best_plan_for_candidate is None:
            return -1  # 候选节点不可行

        # 创建临时树状态 (加入候选节点后)
        temp_tree = {
            'tree': np.logical_or(current_tree['tree'], best_plan_for_candidate['tree']).astype(float),
            'hvt': np.maximum(current_tree['hvt'], best_plan_for_candidate['hvt']),
            'paths_map': current_tree['paths_map'].copy()
        }
        temp_tree['paths_map'][request['dest'][candidate_d_idx]] = best_path_for_candidate
        temp_nodes = nodes_on_tree.union(set(best_path_for_candidate))

        # 创建临时网络状态
        temp_state = {
            'bw': local_state['bw'].copy(),
            'cpu': local_state['cpu'].copy(),
            'mem': local_state['mem'].copy(),
            'hvt': local_state['hvt'].copy(),
            'bw_ref_count': local_state['bw_ref_count'].copy(),
            'request': request
        }

        # 应用候选节点的资源变化
        self._apply_resources_to_state(best_plan_for_candidate, request, temp_state)

        # 评估后续 n_depth 个节点
        remaining = remaining_dests - {candidate_d_idx}
        subsequent_evals = []

        for next_d_idx in list(remaining)[:n_depth]:
            best_eval_for_next = -1

            # 遍历临时树上的所有路径
            for conn_path in temp_tree['paths_map'].values():
                for i_idx in range(len(conn_path)):
                    for k in range(1, self.k_path_count + 1):
                        eval_val, _, _, _, feasible, _, _ = \
                            self._calc_eval1(
                                next_d_idx, k, i_idx, conn_path,
                                request, temp_tree['hvt'], temp_state, temp_nodes
                            )
                        if feasible and eval_val > best_eval_for_next:
                            best_eval_for_next = eval_val

            if best_eval_for_next > 0:
                subsequent_evals.append(best_eval_for_next)

        # 综合评估: 当前评估值 + 后续平均评估值
        if len(subsequent_evals) > 0:
            subsequent_avg = np.mean(subsequent_evals)
            # 综合得分 = 当前得分 + 0.5 * 后续平均得分
            overall_eval = best_eval_for_candidate + 0.5 * subsequent_avg
        else:
            overall_eval = best_eval_for_candidate

        return overall_eval

    def _apply_resources_to_state(self, plan: Dict, request: Dict, state: Dict):
        """应用资源变化到状态"""
        # 应用链路资源
        for link_idx in np.where(plan['tree'] > 0)[0]:
            if state['bw_ref_count'][link_idx] == 0:
                state['bw'][link_idx] -= request['bw_origin']
            state['bw_ref_count'][link_idx] += 1

        # 应用节点资源
        for node, vnf_t in np.argwhere(plan['hvt'] > 0):
            if state['hvt'][node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    state['cpu'][node] -= request['cpu_origin'][j]
                    state['mem'][node] -= request['memory_origin'][j]
                except ValueError:
                    pass
            state['hvt'][node, vnf_t] += 1

    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> \
            Tuple[Optional[Dict], List[Tuple[int, Tuple[int, int], float]]]:
        """
        运行 MSFC-CE 算法并记录专家决策
        对应论文3.4节完整流程

        返回: (最终方案, 轨迹)
        轨迹 = [(high_level_goal, low_level_action, cost), ...]
        其中:
          - high_level_goal: 目的节点在dest列表中的索引 (0-based)
          - low_level_action: (i_idx, k_idx) 表示从树的第i_idx个节点,用第k_idx条路径连接
          - cost: 该决策的成本
        """
        dest_num = len(request['dest'])
        network_state['request'] = request
        expert_trajectory = []

        # 阶段1: 找到所有 S->d 的最佳路径 (论文步骤四)
        tree_set = []
        best_k_set = []
        best_cost_set = []

        print(f"[Expert] 开始处理请求 {request['id']}, 源={request['source']}, 目的节点={request['dest']}")

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

        # 选择第一个加入树的目的节点 (评估值最大)
        best_d_idx = np.argmax([t.get('eval', -1) for t in tree_set])
        if tree_set[best_d_idx]['eval'] <= 0:
            print(f"[Expert] 请求 {request['id']} 阻塞: 第一个目的节点无可行解")
            return None, []

        # 记录第一个决策 (论文步骤七-九)
        # 第一个节点从源节点连接,所以 i_idx=0
        high_level_goal = best_d_idx
        low_level_action = (0, best_k_set[best_d_idx])  # (i_idx, k_idx)
        cost = best_cost_set[best_d_idx]
        expert_trajectory.append((high_level_goal, low_level_action, cost))

        print(f"[Expert] 第1步: 选择目的节点 d{best_d_idx}(节点{request['dest'][best_d_idx]}), "
              f"动作=(0, {best_k_set[best_d_idx]}), cost={cost:.4f}")

        # 构建初始树
        current_tree = {
            'id': request['id'],
            'tree': tree_set[best_d_idx]['tree'],
            'hvt': tree_set[best_d_idx]['hvt'],
            'paths_map': {request['dest'][best_d_idx]: tree_set[best_d_idx]['paths']}
        }
        nodes_on_tree = set(tree_set[best_d_idx]['paths'])
        unadded = set(range(dest_num)) - {best_d_idx}

        # 创建局部状态副本 (论文"修复3"部分)
        local_network_state = {
            'bw': network_state['bw'].copy(),
            'cpu': network_state['cpu'].copy(),
            'mem': network_state['mem'].copy(),
            'hvt': network_state['hvt'].copy(),
            'bw_ref_count': network_state.get('bw_ref_count', np.zeros(len(network_state['bw']))).copy(),
            'request': request
        }

        # 临时应用第一个目的节点的资源变化
        first_plan = {'tree': current_tree['tree'], 'hvt': current_tree['hvt']}
        self._apply_resources_to_state(first_plan, request, local_network_state)

        step_count = 1
        # 逐步添加剩余目的节点 (论文步骤十-十一)
        while unadded:
            step_count += 1

            # ========== 步骤1: 构建候选目的节点集合 (论文3.4.1节策略2) ==========
            candidate_evaluations = []

            for d_idx in unadded:
                best_eval_for_d = -1
                best_action_for_d = (0, 0)

                # 找到该目的节点接入树的最佳方式
                for conn_path in current_tree['paths_map'].values():
                    plan, eval_val, action, cost = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, local_network_state, nodes_on_tree
                    )

                    if plan.get('feasible') and eval_val > best_eval_for_d:
                        best_eval_for_d = eval_val
                        best_action_for_d = action

                # 记录该目的节点的评估值
                candidate_evaluations.append({
                    'd_idx': d_idx,
                    'eval': best_eval_for_d,
                    'action': best_action_for_d
                })

            # 按评估值排序,选择前m个作为候选集合
            candidate_evaluations.sort(key=lambda x: x['eval'], reverse=True)
            m = min(3, len(candidate_evaluations))  # 候选集合大小 (论文建议2-3)
            candidates = [c for c in candidate_evaluations[:m] if c['eval'] > 0]

            if not candidates:
                print(f"[Expert] 步骤{step_count}: 无可行候选节点,停止构建")
                break

            print(f"[Expert] 步骤{step_count}: 候选集合 = {[c['d_idx'] for c in candidates]}, "
                  f"评估值 = {[c['eval'] for c in candidates]}")

            # ========== 步骤2: 评估每个候选节点的后续树 (论文3.4.1节策略3) ==========
            best_overall_score = -1
            best_overall_d = -1
            best_overall_plan = None
            best_overall_action = (0, 0)
            best_overall_cost = 0

            for candidate in candidates:
                d_idx = candidate['d_idx']

                # 找到该候选节点接入树的最佳方案
                best_plan_for_candidate = None
                best_eval_for_candidate = -1
                best_action_for_candidate = (0, 0)
                best_cost_for_candidate = 0

                for conn_path in current_tree['paths_map'].values():
                    plan, eval_val, action, cost = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, local_network_state, nodes_on_tree
                    )
                    if plan.get('feasible') and eval_val > best_eval_for_candidate:
                        best_eval_for_candidate = eval_val
                        best_plan_for_candidate = plan
                        best_action_for_candidate = action
                        best_cost_for_candidate = cost

                if best_plan_for_candidate is None:
                    continue

                # 评估后续树 (论文中的"后续树收益")
                subsequent_score = self._evaluate_subsequent_tree(
                    current_tree=current_tree,
                    candidate_d_idx=d_idx,
                    remaining_dests=unadded,
                    local_state=local_network_state,
                    nodes_on_tree=nodes_on_tree,
                    n_depth=2  # 评估后续2个节点
                )

                # 综合评分 = 当前评估值 + 后续树评估值
                overall_score = subsequent_score

                print(f"[Expert]   候选d{d_idx}: 当前eval={best_eval_for_candidate:.4f}, "
                      f"后续eval={subsequent_score:.4f}")

                # 更新全局最优
                if overall_score > best_overall_score:
                    best_overall_score = overall_score
                    best_overall_d = d_idx
                    best_overall_plan = best_plan_for_candidate
                    best_overall_action = best_action_for_candidate
                    best_overall_cost = best_cost_for_candidate

            # 如果没有找到可行方案,停止 (论文"修复2")
            if best_overall_d == -1:
                print(f"[Expert] 步骤{step_count}: 无可行方案,停止构建 (已成功: {step_count-1}/{dest_num})")
                break

            # 记录决策
            high_level_goal = best_overall_d
            low_level_action = best_overall_action
            cost = best_overall_cost
            expert_trajectory.append((high_level_goal, low_level_action, cost))

            print(f"[Expert] 步骤{step_count}: 选择目的节点 d{best_overall_d}(节点{request['dest'][best_overall_d]}), "
                  f"动作={best_overall_action}, cost={cost:.4f}, overall_score={best_overall_score:.4f}")

            # 合并树 (论文步骤九)
            current_tree['tree'] = np.logical_or(current_tree['tree'], best_overall_plan['tree']).astype(float)
            current_tree['hvt'] = np.maximum(current_tree['hvt'], best_overall_plan['hvt'])
            current_tree['paths_map'][request['dest'][best_overall_d]] = best_overall_plan['new_path_full']
            nodes_on_tree.update(best_overall_plan['new_path_full'])
            unadded.remove(best_overall_d)

            # 临时应用资源变化 (论文"修复4")
            self._apply_resources_to_state(best_overall_plan, request, local_network_state)

        # 最终验证
        if len(expert_trajectory) < dest_num:
            print(f"[Expert] 请求 {request['id']} 部分成功: {len(expert_trajectory)}/{dest_num} 目的节点已连接")
        else:
            print(f"[Expert] 请求 {request['id']} 完全成功: 所有 {dest_num} 个目的节点已连接")

        print(f"[Expert] 轨迹长度: {len(expert_trajectory)}, 轨迹: {expert_trajectory}\n")

        return current_tree, expert_trajectory


# ============ 辅助函数 ============

def test_expert_solver():
    """测试专家求解器"""
    import os

    # 创建测试数据
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # 创建简单的拓扑矩阵 (5个节点)
    topology = np.array([
        [0, 1, np.inf, np.inf, 1],
        [1, 0, 1, np.inf, np.inf],
        [np.inf, 1, 0, 1, 1],
        [np.inf, np.inf, 1, 0, 1],
        [1, np.inf, 1, 1, 0]
    ])

    # DC节点: 2, 3, 4
    dc_nodes = [2, 3, 4]

    # 容量配置
    capacities = {
        'cpu': 100,
        'memory': 100,
        'bandwidth': 100
    }

    # 创建测试请求
    test_request = {
        'id': 1,
        'source': 1,
        'dest': [3, 5],  # 两个目的节点
        'vnf': [1, 2],   # 需要2个VNF
        'bw_origin': 10,
        'cpu_origin': [20, 20],
        'memory_origin': [15, 15],
        'arrival_time': 0,
        'leave_time': 100
    }

    # 初始化网络状态
    network_state = {
        'bw': np.full(10, 100.0),  # 假设10条链路
        'cpu': np.full(5, 100.0),
        'mem': np.full(5, 100.0),
        'hvt': np.zeros((5, 8)),
        'bw_ref_count': np.zeros(10)
    }

    # 注意: 需要预先生成路径数据库
    # 这里假设已有 Paths.mat 文件
    path_db_file = data_dir / "Paths.mat"

    if not path_db_file.exists():
        print("错误: 需要先生成 Paths.mat 文件")
        print("请使用 MATLAB 或其他工具生成前k条最短路径数据")
        return

    # 创建求解器
    solver = MSFCE_Solver(
        path_db_file=path_db_file,
        topology_matrix=topology,
        dc_nodes=dc_nodes,
        capacities=capacities
    )

    # 求解
    solution, trajectory = solver.solve_request_for_expert(test_request, network_state)

    if solution:
        print("=" * 60)
        print("求解成功!")
        print(f"轨迹: {trajectory}")
        print(f"使用的链路数: {np.sum(solution['tree'])}")
        print(f"部署的VNF总数: {np.sum(solution['hvt'])}")
    else:
        print("求解失败: 请求被阻塞")


if __name__ == "__main__":
    # 运行测试
    test_expert_solver()