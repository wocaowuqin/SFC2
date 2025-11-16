#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce_fixed_full.py
"""
修正并完整实现 MSFC-CE 专家求解器（对应论文第三章）
主要修正点：
 - 路径评价（公式3-11/3-12）按 DC 的剩余资源比率计算 resource score
 - VNF 按路径顺序从源到目的部署（优先复用已有部署），并返回 placement 映射
 - 共享路径判定使用路径重叠段（精确），避免误判
 - 后续树评估采用累积式（sum）后续收益
 - 轨迹 low-level action 扩展为 (i_idx, k_idx, placement_dict)
"""
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import scipy.io as sio


def parse_mat_request(req_obj) -> Dict:
    """
    将 MATLAB 请求结构 (来自 sorted_requests.mat) 解析为 Python 字典
    兼容两种常见 mat 解析形式
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
    except Exception:
        # 备选解析（不同 mat 结构）
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

        # 权重系数 α, β, γ (可调或在初始化随机化)
        self.alpha = 0.3  # hop weight
        self.beta = 0.3   # dc count weight
        self.gamma = 0.4  # resource weight

        print(f"专家加载: {self.node_num}节点, {self.link_num}链路, {self.dc_num}DC")

    # ---------- 基本工具函数 ----------
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
        """
        从路径数据库获取第k条最短路径 (1-based k)
        返回 (nodes_list, dist, links_list)
        """
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

    # ---------- 评分与成本函数 ----------

    def _calc_score(self, src: int, dst: int, dist: int, dc_count: int,
                    cpu_sum: float, mem_sum: float, bw_sum: float) -> float:
        """
        备用的总评分（保留），不过主要评分使用 _calc_path_eval_first / _calc_eval1
        """
        max_dist = max(1, self._get_kth_path_max_distance(src, dst, self.k_path_count))
        score = (
            (1 - dist / max_dist) +
            (dc_count / max(1, self.dc_num)) +
            (cpu_sum / (self.cpu_capacity * max(1, self.dc_num))) +
            (mem_sum / (self.memory_capacity * max(1, self.dc_num))) +
            (bw_sum / (self.bandwidth_capacity * max(1, self.link_num)))
        )
        return score

    def _calculate_cost(self, request: Dict, state: Dict, tree: np.ndarray, hvt: np.ndarray) -> float:
        """
        计算部署此方案的资源成本 (用于 RL 奖励/记录)
        基于论文公式(3-1) 风格实现：带宽成本 + 节点 cpu/mem 成本，归一化到 [0,10]
        """
        bw_cost = 0.0
        cpu_cost = 0.0
        mem_cost = 0.0

        # 带宽成本：新使用链路的数量 * bw_origin
        used_links = np.where(tree > 0)[0]
        if used_links.size > 0:
            bw_ref_count = state.get('bw_ref_count', np.zeros(self.link_num))
            new_links_mask = (bw_ref_count[used_links] == 0)
            bw_cost = float(np.sum(new_links_mask)) * float(request['bw_origin'])

        # 节点成本：对 hvt 中新部署的 VNF 统计对应 request 的 cpu/mem
        for node_idx, vnf_t in np.argwhere(hvt > 0):
            # 如果 state 已有该 vnf 则跳过新增成本（复用）
            if state['hvt'][node_idx, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    cpu_cost += float(request['cpu_origin'][j])
                    mem_cost += float(request['memory_origin'][j])
                except ValueError:
                    pass

        # 权重归一化
        bw_weight = 1.0 / max(1.0, self.bandwidth_capacity)
        cpu_weight = 10.0 / max(1.0, self.cpu_capacity)
        mem_weight = 10.0 / max(1.0, self.memory_capacity)

        total_cost = bw_cost * bw_weight + cpu_cost * cpu_weight + mem_cost * mem_weight

        max_possible_cost = (
            self.link_num * self.bandwidth_capacity * bw_weight +
            self.dc_num * self.cpu_capacity * cpu_weight +
            self.dc_num * self.memory_capacity * mem_weight
        )

        if max_possible_cost == 0:
            return 0.0

        normalized_cost = (total_cost / max_possible_cost) * 10.0
        return float(np.clip(normalized_cost, 0, 10))

    # ---------- 路径评价（论文 3.3.4 / 3.4.1） ----------
    def _calc_path_eval_first(self, src: int, dst: int, k: int, state: Dict, request: Dict) -> float:
        """
        第一目的节点路径评价（对应公式 3-11）
        hop_score, dc_score, resource_score (resource_score 用 DC 上的剩余比例与链路剩余比例合成)
        返回 eval_score（>= -1, -1 表示路径不可用）
        """
        path, dist, links = self._get_path_from_db(src, dst, k)
        if not path:
            return -1.0

        max_dist = max(1, self._get_kth_path_max_distance(src, dst, self.k_path_count))
        hop_score = 1.0 - (dist / max_dist)

        dc_nodes_on_path = [n for n in path if n in self.DC]
        dc_score = len(dc_nodes_on_path) / max(1, self.dc_num)

        # 资源比例：对每个 DC 取 cpu_remain / cpu_capacity, mem_remain / mem_capacity
        cpu_ratios = []
        mem_ratios = []
        for n in dc_nodes_on_path:
            idx = n - 1
            cpu_ratios.append(float(state['cpu'][idx]) / max(1.0, self.cpu_capacity))
            mem_ratios.append(float(state['mem'][idx]) / max(1.0, self.memory_capacity))

        link_ratios = []
        for lid in links:
            li = lid - 1
            if li < len(state['bw']):
                link_ratios.append(float(state['bw'][li]) / max(1.0, self.bandwidth_capacity))

        cpu_mean = float(np.mean(cpu_ratios)) if cpu_ratios else 0.0
        mem_mean = float(np.mean(mem_ratios)) if mem_ratios else 0.0
        bw_mean = float(np.mean(link_ratios)) if link_ratios else 0.0

        resource_score = (cpu_mean + mem_mean + bw_mean) / 3.0

        eval_score = self.alpha * hop_score + self.beta * dc_score + self.gamma * resource_score
        return float(eval_score)

    # ---------- 评估并部署第一个目的节点（严格按路径顺序部署 VNF） ----------
    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """
        评估第一个目的节点 S->d 的第 k 条路径，并尝试按路径顺序部署 VNF。
        返回:
          score, path, tree (link vec), hvt_new (only new deployments marked as 1),
          feasible (bool), dest, cost, placement_map
        placement_map: {vnf_index (0-based): node_id (1-based)}
        """
        src = request['source']
        dest = request['dest'][d_idx]
        path, dist, links = self._get_path_from_db(src, dest, k)
        if not path:
            return 0.0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dest, 0.0, {}

        # 带宽检查
        for lid in links:
            if lid - 1 >= len(state['bw']) or state['bw'][lid - 1] < request['bw_origin']:
                return 0.0, path, np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dest, 0.0, {}

        # 按路径顺序部署 VNF：从 path 的第一个节点开始（包括源自身或源之后的节点）
        placement: Dict[int, int] = {}
        hvt_new = np.zeros((self.node_num, self.type_num))
        vnf_list = request['vnf']
        vnf_count = len(vnf_list)
        assigned = 0

        for node in path:
            if assigned >= vnf_count:
                break
            if node not in self.DC:
                continue
            node_idx = node - 1
            # 试图放置当前还未分配的 vnf (preserving order)
            t = vnf_list[assigned] - 1
            # 如果已经存在，复用
            if state['hvt'][node_idx, t] > 0:
                placement[assigned] = node
                # hvt_new 不记录复用（0），因为不是新增
                assigned += 1
                continue
            # 否则检查资源充足性
            if state['cpu'][node_idx] >= request['cpu_origin'][assigned] and state['mem'][node_idx] >= request['memory_origin'][assigned]:
                hvt_new[node_idx, t] = 1
                placement[assigned] = node
                assigned += 1
                continue
            # 否则跳到 path 上的下一个 DC

        if assigned != vnf_count:
            # 无法在这条路径上按顺序部署完全部 VNF
            return 0.0, path, np.zeros(self.link_num), hvt_new, False, dest, 0.0, {}

        # 标记链接
        tree_vec = np.zeros(self.link_num)
        for lid in links:
            tree_vec[lid - 1] = 1

        cost = self._calculate_cost(request, state, tree_vec, hvt_new)

        # 计算评分（可选，基于路径与资源）
        cpu_sum = sum(state['cpu'][n - 1] for n in path if n in self.DC)
        mem_sum = sum(state['mem'][n - 1] for n in path if n in self.DC)
        bw_sum = sum(state['bw'][lid - 1] for lid in links if lid - 1 < len(state['bw']))
        score = self._calc_score(src, dest, dist, len([n for n in path if n in self.DC]), cpu_sum, mem_sum, bw_sum)

        return float(score), path, tree_vec, hvt_new, True, dest, float(cost), placement

    # ---------- 评估从树上某节点接入目的节点的第 k 条路径（用于后续节点接入） ----------
    def _calc_eval1(self, d_idx: int, k: int, i_idx: int, tree1_path: List[int],
                    request: Dict, tree1_hvt: np.ndarray, state: Dict, nodes_on_tree: Set[int]):
        """
        评估从 tree1_path 的第 i_idx 个节点连接到 request.dest[d_idx] 的第 k 条路径
        返回: eval_score, full_path, tree_links_vec, hvt_after_new, feasible, dest_or_0, cost, placement
        """
        connect_node = tree1_path[i_idx]
        dest_node = request['dest'][d_idx]
        paths, dist, links = self._get_path_from_db(connect_node, dest_node, k)
        if not paths or len(paths) < 2:
            return 0.0, [], np.zeros(self.link_num), tree1_hvt.copy(), False, dest_node, 0.0, {}

        # 破环检测：如果路径与 tree1_path 中 connect_node 之后的节点重叠 -> infeasible
        new_nodes = set(paths[1:])
        if new_nodes & set(tree1_path[i_idx + 1:]):
            return 0.0, paths, np.zeros(self.link_num), tree1_hvt.copy(), False, dest_node, 0.0, {}

        # 若与树上其他路径重叠（除自己外） -> infeasible（避免复杂环）
        other_nodes = nodes_on_tree - set(tree1_path)
        if new_nodes & other_nodes:
            return 0.0, paths, np.zeros(self.link_num), tree1_hvt.copy(), False, dest_node, 0.0, {}

        # 带宽检查
        for lid in links:
            if lid - 1 >= len(state['bw']) or state['bw'][lid - 1] < request['bw_origin']:
                return 0.0, paths, np.zeros(self.link_num), tree1_hvt.copy(), False, dest_node, 0.0, {}

        # 共享路径（重叠段）上的 DC
        shared_nodes = set(tree1_path[:i_idx + 1]) & set(paths)
        deployed_on_shared = [n for n in shared_nodes if n in self.DC]

        # 统计 shared 上已为 request 部署的 VNF 类型数量
        shared_deployed_count = 0
        for v in request['vnf']:
            if any(tree1_hvt[n - 1, v - 1] > 0 for n in deployed_on_shared):
                shared_deployed_count += 1
        undeployed_count = len(request['vnf']) - shared_deployed_count

        # copy hvt
        hvt = tree1_hvt.copy()
        placement: Dict[int, int] = {}

        # 若 shared 已覆盖全部 vnf，则仅使用该路径（无需新部署）
        if undeployed_count == 0:
            tree_vec = np.zeros(self.link_num)
            for lid in links:
                tree_vec[lid - 1] = 1
            cost = self._calculate_cost(request, state, tree_vec, hvt)
            CPU_status = sum(state['cpu'][n - 1] for n in paths[1:] if n in self.DC)
            Memory_status = sum(state['mem'][n - 1] for n in paths[1:] if n in self.DC)
            Bandwidth_status = sum(state['bw'][lid - 1] for lid in links)
            eval_score = self._calc_score(connect_node, dest_node, dist, len(deployed_on_shared),
                                         CPU_status, Memory_status, Bandwidth_status)
            return float(eval_score), paths, tree_vec, hvt, True, 0, float(cost), {}

        # 需要在 branch 上部署剩余 VNF: 在 paths[1:]（从接入点到目的）按顺序部署
        usable_on_branch = [n for n in paths[1:] if n in self.DC]
        if len(usable_on_branch) < undeployed_count:
            return 0.0, paths, np.zeros(self.link_num), tree1_hvt.copy(), False, dest_node, 0.0, {}

        # 构建未被 shared 覆盖的 vnf types（保序）
        needed_types = []
        for t in request['vnf']:
            if not any(hvt[n - 1, t - 1] > 0 for n in deployed_on_shared):
                needed_types.append(t)

        j = 0
        for node in usable_on_branch:
            if j >= len(needed_types):
                break
            node_idx = node - 1
            t = needed_types[j] - 1
            if hvt[node_idx, t] == 0:
                # 检查资源（注意：request.cpu_origin 与 needed_types 的下标映射需要考虑原顺序）
                try:
                    # find the index in original vnf list to get resource demands
                    orig_index = request['vnf'].index(needed_types[j])
                except ValueError:
                    orig_index = j
                if state['cpu'][node_idx] < request['cpu_origin'][orig_index] or state['mem'][node_idx] < request['memory_origin'][orig_index]:
                    # 不足，跳到下一个 DC
                    continue
                hvt[node_idx, t] = 1
                placement[orig_index] = node
                j += 1
            else:
                # already deployed on branch
                try:
                    orig_index = request['vnf'].index(needed_types[j])
                except ValueError:
                    orig_index = j
                placement[orig_index] = node
                j += 1

        # 检查是否全部部署
        total_deployed = sum(1 for t in request['vnf'] if any(hvt[n - 1, t - 1] > 0 for n in (deployed_on_shared + usable_on_branch)))
        if total_deployed != len(request['vnf']):
            return 0.0, paths, np.zeros(self.link_num), tree1_hvt.copy(), False, dest_node, 0.0, {}

        # 标记链路并返回
        tree_vec = np.zeros(self.link_num)
        for lid in links:
            tree_vec[lid - 1] = 1
        cost = self._calculate_cost(request, state, tree_vec, hvt)
        CPU_status = sum(state['cpu'][n - 1] for n in paths[1:] if n in self.DC)
        Memory_status = sum(state['mem'][n - 1] for n in paths[1:] if n in self.DC)
        Bandwidth_status = sum(state['bw'][lid - 1] for lid in links)
        eval_score = self._calc_score(connect_node, dest_node, dist, len(usable_on_branch),
                                     CPU_status, Memory_status, Bandwidth_status)

        return float(eval_score), paths, tree_vec, hvt, True, 0, float(cost), placement

    # ---------- 查找将目的节点 d 加入 tree1 的最佳方案（遍历 i_idx, k） ----------
    def _calc_atnp(self, tree1: Dict, tree1_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """
        返回 best_plan_dict, best_eval, best_action (i_idx, k_idx), best_cost
        如果无 feasible，则返回一个不可行 plan（feasible False）和 infeasible_dest
        """
        request = state['request']

        if tree1.get('eval', 0) == 0:
            return {
                'tree': tree1['tree'].copy(), 'hvt': tree1['hvt'].copy(),
                'feasible': tree1.get('feasible', False),
                'infeasible_dest': tree1.get('infeasible_dest', 0)
            }, 0.0, (0, 0), 0.0

        best_eval = -1.0
        best_plan = None
        best_action = (0, 0)
        best_cost = 0.0

        for i_idx in range(len(tree1_path)):
            for k in range(1, self.k_path_count + 1):
                eval_val, paths, tree_new, hvt_new, feasible, infeasible_dest, cost, placement = \
                    self._calc_eval1(
                        d_idx, k, i_idx, tree1_path, request,
                        tree1['hvt'], state, nodes_on_tree
                    )
                if feasible and eval_val > best_eval:
                    best_eval = eval_val
                    best_action = (i_idx, k - 1)  # return 0-based k_idx for consistency
                    best_cost = cost
                    best_plan = {
                        'tree': tree_new, 'hvt': hvt_new, 'new_path_full': paths,
                        'connect_idx': i_idx, 'feasible': True, 'infeasible_dest': 0,
                        'placement': placement
                    }

        if best_plan is None:
            return {
                'tree': tree1['tree'].copy(), 'hvt': tree1['hvt'].copy(),
                'feasible': False, 'infeasible_dest': request['dest'][d_idx]
            }, 0.0, (0, 0), 0.0

        return best_plan, best_eval, best_action, best_cost

    # ---------- 后续树评估（采用累积式收益） ----------
    def _evaluate_subsequent_tree(self, current_tree: Dict, candidate_d_idx: int,
                                  remaining_dests: Set[int], local_state: Dict,
                                  nodes_on_tree: Set[int], n_depth: int = 2) -> float:
        request = local_state['request']

        # 找到候选节点加入树的最佳方案
        best_plan_for_candidate = None
        best_eval_for_candidate = -1.0
        best_path_for_candidate = None

        for conn_path in current_tree['paths_map'].values():
            plan, eval_val, action, cost = self._calc_atnp(
                {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                conn_path, candidate_d_idx, local_state, nodes_on_tree
            )
            if plan.get('feasible') and eval_val > best_eval_for_candidate:
                best_eval_for_candidate = eval_val
                best_plan_for_candidate = plan
                best_path_for_candidate = plan.get('new_path_full', [])

        if best_plan_for_candidate is None:
            return -1.0

        # 临时树与状态
        temp_tree = {
            'tree': np.logical_or(current_tree['tree'], best_plan_for_candidate['tree']).astype(float),
            'hvt': np.maximum(current_tree['hvt'], best_plan_for_candidate['hvt']),
            'paths_map': current_tree['paths_map'].copy()
        }
        temp_tree['paths_map'][request['dest'][candidate_d_idx]] = best_path_for_candidate
        temp_nodes = nodes_on_tree.union(set(best_path_for_candidate))

        temp_state = {
            'bw': local_state['bw'].copy(),
            'cpu': local_state['cpu'].copy(),
            'mem': local_state['mem'].copy(),
            'hvt': local_state['hvt'].copy(),
            'bw_ref_count': local_state['bw_ref_count'].copy(),
            'request': request
        }

        # 应用 candidate 的资源占用
        self._apply_resources_to_state(best_plan_for_candidate, request, temp_state)

        remaining = list(remaining_dests - {candidate_d_idx})
        subsequent_sum = 0.0
        count = 0
        for next_d_idx in remaining[:n_depth]:
            best_eval_for_next = -1.0
            for conn_path in temp_tree['paths_map'].values():
                for i_idx in range(len(conn_path)):
                    for k in range(1, self.k_path_count + 1):
                        eval_val, _, _, _, feasible, _, _, _ = \
                            self._calc_eval1(
                                next_d_idx, k, i_idx, conn_path,
                                request, temp_tree['hvt'], temp_state, temp_nodes
                            )
                        if feasible and eval_val > best_eval_for_next:
                            best_eval_for_next = eval_val
            if best_eval_for_next > 0:
                subsequent_sum += best_eval_for_next
                count += 1

        if count > 0:
            overall_eval = best_eval_for_candidate + 0.5 * subsequent_sum
        else:
            overall_eval = best_eval_for_candidate

        return float(overall_eval)

    # ---------- 应用资源变化到状态 ----------
    def _apply_resources_to_state(self, plan: Dict, request: Dict, state: Dict):
        # 链路资源
        for link_idx in np.where(plan['tree'] > 0)[0]:
            if state['bw_ref_count'][link_idx] == 0:
                state['bw'][link_idx] -= request['bw_origin']
            state['bw_ref_count'][link_idx] += 1

        # 节点资源 (只扣除新增部署)
        for node, vnf_t in np.argwhere(plan['hvt'] > 0):
            if state['hvt'][node, vnf_t] == 0:
                try:
                    j = request['vnf'].index(vnf_t + 1)
                    state['cpu'][node] -= request['cpu_origin'][j]
                    state['mem'][node] -= request['memory_origin'][j]
                except ValueError:
                    pass
            state['hvt'][node, vnf_t] += 1

    # ---------- 主流程：基于专家策略生成轨迹 ----------
    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> \
            Tuple[Optional[Dict], List[Tuple[int, Tuple[int, int, Dict[int,int]], float]]]:
        """
        运行 MSFC-CE 算法并记录专家决策轨迹
        返回: final_tree (或 None if blocked), expert_trajectory
        expert_trajectory = [(high_level_goal, low_level_action, cost), ...]
          low_level_action = (i_idx, k_idx, placement_dict)
        """
        dest_num = len(request['dest'])
        network_state['request'] = request
        expert_trajectory = []

        # 阶段1: 为每个目的节点找到 best S->d 路径及其评估
        tree_set = []
        best_k_set = []
        best_cost_set = []

        print(f"[Expert] 开始处理请求 {request['id']}, 源={request['source']}, 目的节点={request['dest']}")

        for d_idx in range(dest_num):
            best_eval, best_result, best_k, best_cost = -1.0, None, 0, 0.0
            for k in range(1, self.k_path_count + 1):
                eval_val = self._calc_path_eval_first(request['source'], request['dest'][d_idx], k, network_state, request)
                if eval_val <= 0:
                    continue
                # 尝试构建并部署（评估能否按该路径部署）
                score, paths, tree, hvt_new, feasible, dest, cost, placement = self._calc_eval(request, d_idx, k, network_state)
                if feasible and score > best_eval:
                    best_eval = score
                    best_k = k - 1
                    best_cost = cost
                    best_result = {'eval': score, 'paths': paths, 'tree': tree, 'hvt': hvt_new, 'placement': placement}
            tree_set.append(best_result if best_result else {'eval': -1})
            best_k_set.append(best_k)
            best_cost_set.append(best_cost)

        # 选择第一个加入树的目的节点 (max eval)
        evals = [t.get('eval', -1) for t in tree_set]
        best_d_idx = int(np.argmax(evals))
        if tree_set[best_d_idx].get('eval', -1) <= 0:
            print(f"[Expert] 请求 {request['id']} 阻塞: 第一个目的节点无可行解")
            return None, []

        # 记录第一个决策 (i_idx=0, 从源直接)
        high_level_goal = best_d_idx
        low_level_action = (0, best_k_set[best_d_idx], tree_set[best_d_idx].get('placement', {}))
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

        # 局部状态副本
        local_network_state = {
            'bw': network_state['bw'].copy(),
            'cpu': network_state['cpu'].copy(),
            'mem': network_state['mem'].copy(),
            'hvt': network_state['hvt'].copy(),
            'bw_ref_count': network_state.get('bw_ref_count', np.zeros(len(network_state['bw']))).copy(),
            'request': request
        }

        # 应用第一个目的节点的资源变化
        first_plan = {'tree': current_tree['tree'], 'hvt': current_tree['hvt']}
        self._apply_resources_to_state(first_plan, request, local_network_state)

        step_count = 1
        while unadded:
            step_count += 1

            # ========== 构建候选集合 ==========
            candidate_evaluations = []
            for d_idx in unadded:
                best_eval_for_d = -1.0
                best_action_for_d = (0, 0)
                for conn_path in current_tree['paths_map'].values():
                    plan, eval_val, action, cost = self._calc_atnp(
                        {'tree': current_tree['tree'].copy(), 'hvt': current_tree['hvt'].copy()},
                        conn_path, d_idx, local_network_state, nodes_on_tree
                    )
                    if plan.get('feasible') and eval_val > best_eval_for_d:
                        best_eval_for_d = eval_val
                        best_action_for_d = action
                candidate_evaluations.append({'d_idx': d_idx, 'eval': best_eval_for_d, 'action': best_action_for_d})

            candidate_evaluations.sort(key=lambda x: x['eval'], reverse=True)
            m = min(3, len(candidate_evaluations))
            candidates = [c for c in candidate_evaluations[:m] if c['eval'] > 0]

            if not candidates:
                print(f"[Expert] 步骤{step_count}: 无可行候选节点,停止构建")
                break

            print(f"[Expert] 步骤{step_count}: 候选集合 = {[c['d_idx'] for c in candidates]}, "
                  f"评估值 = {[c['eval'] for c in candidates]}")

            # ========== 对候选评估其后续树 ==========
            best_overall_score = -1.0
            best_overall_d = -1
            best_overall_plan = None
            best_overall_action = (0, 0)
            best_overall_cost = 0.0

            for candidate in candidates:
                d_idx = candidate['d_idx']
                # 找到该候选节点接入的best方案
                best_plan_for_candidate = None
                best_eval_for_candidate = -1.0
                best_action_for_candidate = (0, 0)
                best_cost_for_candidate = 0.0

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

                subsequent_score = self._evaluate_subsequent_tree(
                    current_tree=current_tree,
                    candidate_d_idx=d_idx,
                    remaining_dests=unadded,
                    local_state=local_network_state,
                    nodes_on_tree=nodes_on_tree,
                    n_depth=2
                )

                overall_score = subsequent_score

                print(f"[Expert]   候选d{d_idx}: 当前eval={best_eval_for_candidate:.4f}, "
                      f"后续eval={subsequent_score:.4f}")

                if overall_score > best_overall_score:
                    best_overall_score = overall_score
                    best_overall_d = d_idx
                    best_overall_plan = best_plan_for_candidate
                    best_overall_action = best_action_for_candidate
                    best_overall_cost = best_cost_for_candidate

            if best_overall_d == -1:
                print(f"[Expert] 步骤{step_count}: 无可行方案,停止构建 (已成功: {step_count-1}/{dest_num})")
                break

            # 记录决策（注意扩展 low_level_action 包含 placement）
            high_level_goal = best_overall_d
            # best_overall_action 是 (i_idx, k_idx)；plan 包含 placement
            low_level_action = (best_overall_action[0], best_overall_action[1], best_overall_plan.get('placement', {}))
            cost = best_overall_cost
            expert_trajectory.append((high_level_goal, low_level_action, cost))

            print(f"[Expert] 步骤{step_count}: 选择目的节点 d{best_overall_d}(节点{request['dest'][best_overall_d]}), "
                  f"动作={low_level_action}, cost={cost:.4f}, overall_score={best_overall_score:.4f}")

            # 合并树
            current_tree['tree'] = np.logical_or(current_tree['tree'], best_overall_plan['tree']).astype(float)
            current_tree['hvt'] = np.maximum(current_tree['hvt'], best_overall_plan['hvt'])
            current_tree['paths_map'][request['dest'][best_overall_d]] = best_overall_plan['new_path_full']
            nodes_on_tree.update(best_overall_plan['new_path_full'])
            unadded.remove(best_overall_d)

            # 应用资源变化到 local network state
            self._apply_resources_to_state(best_overall_plan, request, local_network_state)

        # 最终验证
        if len(expert_trajectory) < dest_num:
            print(f"[Expert] 请求 {request['id']} 部分成功: {len(expert_trajectory)}/{dest_num} 目的节点已连接")
        else:
            print(f"[Expert] 请求 {request['id']} 完全成功: 所有 {dest_num} 个目的节点已连接")

        print(f"[Expert] 轨迹长度: {len(expert_trajectory)}, 轨迹: {expert_trajectory}\n")

        return current_tree, expert_trajectory


# =================== 测试函数 ===================
def test_expert_solver():
    """
    运行一个本地测试（需要预先生成 Paths.mat）
    用以验证基本功能：构造拓扑、请求；若缺 Paths.mat 则会提示。
    """
    import os

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    topology = np.array([
        [0, 1, np.inf, np.inf, 1],
        [1, 0, 1, np.inf, np.inf],
        [np.inf, 1, 0, 1, 1],
        [np.inf, np.inf, 1, 0, 1],
        [1, np.inf, 1, 1, 0]
    ])

    dc_nodes = [2, 3, 4]

    capacities = {'cpu': 100, 'memory': 100, 'bandwidth': 100}

    test_request = {
        'id': 1,
        'source': 1,
        'dest': [3, 5],
        'vnf': [1, 2],
        'bw_origin': 10,
        'cpu_origin': [20, 20],
        'memory_origin': [15, 15],
        'arrival_time': 0,
        'leave_time': 100
    }

    network_state = {
        'bw': np.full(10, 100.0),
        'cpu': np.full(5, 100.0),
        'mem': np.full(5, 100.0),
        'hvt': np.zeros((5, 8)),
        'bw_ref_count': np.zeros(10)
    }

    path_db_file = data_dir / "Paths.mat"

    if not path_db_file.exists():
        print("错误: 需要先生成 Paths.mat 文件（包含 Paths 数据结构）。")
        print("建议：用 Matlab/脚本预计算每对节点的前 k 条最短路径并保存为 Paths 结构。")
        return

    solver = MSFCE_Solver(path_db_file=path_db_file, topology_matrix=topology, dc_nodes=dc_nodes, capacities=capacities)
    sol, traj = solver.solve_request_for_expert(test_request, network_state)
    if sol:
        print("=" * 60)
        print("求解成功!")
        print(f"轨迹: {traj}")
        print(f"使用的链路数: {np.sum(sol['tree'])}")
        print(f"部署的VNF总数: {np.sum(sol['hvt'])}")
    else:
        print("求解失败: 请求被阻塞")


if __name__ == "__main__":
    test_expert_solver()
