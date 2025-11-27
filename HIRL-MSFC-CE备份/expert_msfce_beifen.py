#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce.py
# ✅ 完全修复版本: 解决所有 KeyError 问题

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import copy

# --- 配置参数 ---
ALPHA = 0.3  # 跳数权重
BETA = 0.3  # DC节点数权重
GAMMA = 0.4  # 剩余资源权重
CANDIDATE_SET_SIZE = 4  # 候选集合大小


def parse_mat_request(req_obj) -> Dict:
    """解析 MATLAB 请求结构"""
    try:
        parsed = {
            'id': int(req_obj['id'][0, 0]),
            'source': int(req_obj['source'][0, 0]),
            'dest': [int(d) for d in req_obj['dest'].flatten()],
            'vnf': [int(v) for v in req_obj['vnf'].flatten()],
            'bw_origin': float(req_obj['bw_origin'][0, 0]),
            'cpu_origin': [float(c) for c in req_obj['cpu_origin'].flatten()],
            'memory_origin': [float(m) for m in req_obj['memory_origin'].flatten()],
            'arrival_time': int(req_obj['arrival_time'][0, 0]),
            'leave_time': int(req_obj['leave_time'][0, 0]),
        }
    except Exception:
        parsed = {
            'id': int(req_obj[0][0][0]),
            'source': int(req_obj[0][1][0]),
            'dest': [int(x) for x in req_obj[0][2].flatten()],
            'vnf': [int(x) for x in req_obj[0][3].flatten()],
            'cpu_origin': [float(x) for x in req_obj[0][4].flatten()],
            'memory_origin': [float(x) for x in req_obj[0][5].flatten()],
            'bw_origin': float(req_obj[0][6][0][0])
        }
    return parsed


class MSFCE_Solver:
    def __init__(self, path_db_file: Path, topology_matrix: np.ndarray,
                 dc_nodes: List[int], capacities: Dict):
        self.path_db = None
        try:
            if path_db_file.exists():
                self.path_db = sio.loadmat(path_db_file)['Paths']
                print(f"✅ 成功加载路径数据库: {path_db_file}")
        except Exception as e:
            print(f"⚠️ 无法加载路径数据库: {e}")

        self.node_num = topology_matrix.shape[0]
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        self.type_num = 8
        self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        # ✅ 修复: 统一容量属性命名
        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        # 保留旧名称以兼容
        self.cpu_capacity = self.cap_cpu
        self.memory_capacity = self.cap_mem
        self.bandwidth_capacity = self.cap_bw

        self.k_path_count = 5
        self.k_path = 5

        print(f"✅ 专家初始化: {self.node_num}节点, {self.link_num}链路, {self.dc_num}DC")

    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        link_map = {}
        lid = 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def _get_path_info(self, src: int, dst: int, k: int):
        """获取第k条最短路径"""
        if self.path_db is None:
            return [], 0, []
        try:
            cell = self.path_db[src - 1, dst - 1]
            dist = int(cell['pathsdistance'][k - 1][0])
            nodes = cell['paths'][k - 1, :dist + 1].astype(int).tolist()
            links = []
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                if (u, v) in self.link_map:
                    links.append(self.link_map[(u, v)])
            return nodes, dist, links
        except Exception:
            return [], 0, []

    def _get_max_hops(self, src: int, dst: int) -> int:
        """获取最大跳数"""
        try:
            cell = self.path_db[src - 1, dst - 1]
            return int(cell['pathsdistance'][self.k_path - 1][0])
        except:
            return 10

    # ============================================
    # ✅ 关键修复: 标准化状态访问方法
    # ============================================
    def _normalize_state(self, state: Dict) -> Dict:
        """
        将环境状态转换为专家内部格式
        环境使用: 'bw', 'cpu', 'mem', 'hvt', 'bw_ref_count'
        """
        normalized = {}

        # 带宽
        normalized['bw'] = state.get('bw', state.get('bandwidth', np.full(self.link_num, self.cap_bw)))

        # CPU
        normalized['cpu'] = state.get('cpu', np.full(self.node_num, self.cap_cpu))

        # 内存
        normalized['mem'] = state.get('mem', state.get('memory', np.full(self.node_num, self.cap_mem)))

        # HVT (最关键!)
        normalized['hvt'] = state.get('hvt', state.get('hvt_all', np.zeros((self.node_num, self.type_num))))

        # 引用计数
        normalized['bw_ref_count'] = state.get('bw_ref_count', np.zeros(self.link_num))

        # 请求
        if 'request' in state:
            normalized['request'] = state['request']

        return normalized

    # --- 核心评分逻辑 ---
    def _calc_path_eval(self, nodes: List[int], links: List[int],
                        state: Dict, src_node: int, dst_node: int) -> float:
        """计算路径评分"""
        if not nodes:
            return 0.0

        # ✅ 标准化状态
        state = self._normalize_state(state)

        max_hops = self._get_max_hops(src_node, dst_node)
        current_hops = len(nodes) - 1
        term1 = 1.0 - (current_hops / max(1, max_hops))

        dc_count = sum(1 for n in nodes if n in self.DC)
        term2 = dc_count / max(1, self.dc_num)

        sr_val = 0.0
        for n in nodes:
            if n in self.DC:
                idx = n - 1
                if idx < len(state['cpu']):
                    cpu_remain = state['cpu'][idx]
                    mem_remain = state['mem'][idx]
                    sr_val += (cpu_remain + mem_remain) / (self.cap_cpu + self.cap_mem)

        for lid in links:
            idx = lid - 1
            if idx < len(state['bw']):
                bw_remain = state['bw'][idx]
                sr_val += bw_remain / self.cap_bw

        norm_factor = max(1, len(nodes) + len(links))
        term3 = sr_val / norm_factor

        return float(ALPHA * term1 + BETA * term2 + GAMMA * term3)

    def _try_deploy_vnf(self, request: Dict, path_nodes: List[int],
                        state: Dict, existing_hvt: np.ndarray) -> Tuple[bool, np.ndarray, Dict]:
        """尝试在路径上部署VNF"""
        # ✅ 标准化状态
        state = self._normalize_state(state)

        req_vnfs = request['vnf']
        hvt = existing_hvt.copy()
        placement = {}
        path_dcs = [n for n in path_nodes if n in self.DC]

        if len(path_dcs) < len(req_vnfs):
            return False, existing_hvt, {}

        current_dc_idx = 0

        for v_idx, v_type in enumerate(req_vnfs):
            deployed = False

            # 1. 尝试复用现有VNF
            for node in path_dcs:
                node_idx = node - 1
                if node_idx < hvt.shape[0] and (v_type - 1) < hvt.shape[1]:
                    if hvt[node_idx, v_type - 1] > 0:
                        placement[v_idx] = node
                        deployed = True
                        break

            if deployed:
                continue

            # 2. 尝试新部署
            start_search = current_dc_idx
            while start_search < len(path_dcs):
                node = path_dcs[start_search]
                node_idx = node - 1

                if node_idx >= len(state['cpu']):
                    start_search += 1
                    continue

                cpu_req = request['cpu_origin'][v_idx]
                mem_req = request['memory_origin'][v_idx]

                if state['cpu'][node_idx] >= cpu_req and state['mem'][node_idx] >= mem_req:
                    # ✅ 临时占用资源 (不修改原状态)
                    hvt[node_idx, v_type - 1] = 1
                    placement[v_idx] = node
                    deployed = True
                    current_dc_idx = start_search + 1
                    break
                else:
                    start_search += 1

            if not deployed:
                return False, existing_hvt, {}

        return True, hvt, placement

    # ============================================
    # ✅ 修复: 环境调用接口 (_calc_eval)
    # ============================================
    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """
        Stage 1: Source -> Dest
        返回值与原版兼容: (score, nodes, tree_vec, hvt, feasible, dst, cost, placement)
        """
        # ✅ 标准化状态
        state = self._normalize_state(state)

        src = request['source']
        dst = request['dest'][d_idx]
        nodes, dist, links = self._get_path_info(src, dst, k)

        if not nodes:
            return (0.0, [], np.zeros(self.link_num),
                    np.zeros((self.node_num, self.type_num)),
                    False, dst, 0.0, {})

        # 计算评分
        score = self._calc_path_eval(nodes, links, state, src, dst)

        # 尝试部署
        temp_state = copy.deepcopy(state)
        initial_hvt = np.zeros((self.node_num, self.type_num))
        feasible, new_hvt, placement = self._try_deploy_vnf(
            request, nodes, temp_state, initial_hvt
        )

        # 构建树向量
        tree_vec = np.zeros(self.link_num)
        if feasible:
            for lid in links:
                if lid - 1 < len(tree_vec):
                    tree_vec[lid - 1] = 1

        # 简单成本估算
        cost = np.sum(tree_vec) * 0.2 + np.sum(new_hvt) * 0.8 if feasible else 0.0

        return score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement

    # ============================================
    # ✅ 修复: _calc_atnp (用于环境的查询接口)
    # ============================================
    def _calc_atnp(self, current_tree: Dict, conn_path: List[int],
                   d_idx: int, state: Dict, nodes_on_tree: Set[int]):
        """
        Stage 2: Tree Path -> Dest
        返回: (result_dict, eval, action_tuple, cost)
        """
        # ✅ 标准化状态
        state = self._normalize_state(state)

        # 获取请求
        request = state.get('request')
        if request is None:
            return {'feasible': False}, 0.0, (0, 0), 0.0

        best_eval = -1.0
        best_res = None
        best_action = (0, 0)

        for i_idx, conn_node in enumerate(conn_path):
            for k in range(1, self.k_path + 1):
                nodes, dist, links = self._get_path_info(
                    conn_node, request['dest'][d_idx], k
                )

                if not nodes or len(nodes) < 2:
                    continue

                # 检查是否形成环路
                if set(nodes[1:]) & nodes_on_tree:
                    continue

                # 计算评分
                score = self._calc_path_eval(nodes, links, state, conn_node, request['dest'][d_idx])

                if score > best_eval:
                    temp_state = copy.deepcopy(state)
                    temp_state['request'] = request

                    # 完整路径用于VNF部署检查
                    full_nodes = conn_path[:i_idx + 1] + nodes[1:]

                    # 获取现有HVT
                    existing_hvt = current_tree.get('hvt', current_tree.get('hvt_vec',
                                                                            np.zeros((self.node_num, self.type_num))))

                    feasible, new_hvt, placement = self._try_deploy_vnf(
                        request, full_nodes, temp_state, existing_hvt
                    )

                    if feasible:
                        best_eval = score
                        tree_vec = np.zeros(self.link_num)
                        for lid in links:
                            if lid - 1 < len(tree_vec):
                                tree_vec[lid - 1] = 1

                        best_res = {
                            'tree': tree_vec,
                            'hvt': new_hvt,
                            'new_path_full': nodes,
                            'feasible': True,
                            'placement': placement
                        }
                        best_action = (i_idx, k - 1)

        if best_res:
            cost = np.sum(best_res['tree']) * 0.2 + np.sum(best_res['hvt']) * 0.8
            return best_res, best_eval, best_action, cost
        else:
            return {'feasible': False}, 0.0, (0, 0), 0.0

    # ============================================
    # ✅ 修复: 专家主流程
    # ============================================
    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> Tuple[Optional[Dict], List]:
        """
        执行专家算法并生成训练轨迹
        返回: (solution_tree, trajectory)
        trajectory = [(goal_idx, action_tuple, cost), ...]
        """
        # ✅ 标准化状态
        network_state = self._normalize_state(network_state)
        network_state['request'] = request

        temp_state = copy.deepcopy(network_state)

        # 初始化树结构
        current_tree = {
            'id': request['id'],
            'tree': np.zeros(self.link_num),  # ✅ 使用 'tree' 而非 'tree_vec'
            'hvt': np.zeros((self.node_num, self.type_num)),  # ✅ 使用 'hvt'
            'paths_map': {},
            'added_dest_indices': []
        }

        nodes_on_tree = {request['source']}
        trajectory = []

        dest_indices = list(range(len(request['dest'])))
        unadded = set(dest_indices)

        # ============================================
        # 阶段 1: 选择最佳初始路径 (S -> D)
        # ============================================
        if unadded:
            best_score = -1
            best_d_idx = -1
            best_k = -1
            best_data = None

            for d_idx in unadded:
                for k in range(1, self.k_path + 1):
                    score, nodes, tree_vec, hvt, feasible, _, cost, placement = \
                        self._calc_eval(request, d_idx, k, temp_state)

                    if feasible and score > best_score:
                        best_score = score
                        best_d_idx = d_idx
                        best_k = k
                        best_data = (nodes, tree_vec, hvt, cost, placement)

            if best_d_idx == -1:
                # 无法找到任何可行路径
                return None, []

            # 应用最佳初始路径
            nodes, tree_vec, hvt, cost, placement = best_data
            current_tree['tree'] = tree_vec
            current_tree['hvt'] = hvt
            current_tree['paths_map'][request['dest'][best_d_idx]] = nodes
            current_tree['added_dest_indices'].append(best_d_idx)
            nodes_on_tree.update(nodes)

            # 更新临时状态
            self._apply_deployment_to_state(request, tree_vec, hvt, temp_state)

            # 记录轨迹
            action_tuple = (0, best_k - 1)  # (path_idx, k_idx)
            trajectory.append((best_d_idx, action_tuple, cost))

            unadded.remove(best_d_idx)

        # ============================================
        # 阶段 2: 迭代添加剩余目的地
        # ============================================
        while unadded:
            best_eval = -1
            best_d = -1
            best_plan = None
            best_action = (0, 0)
            best_cost = 0

            for d_idx in unadded:
                # 遍历所有已在树上的路径
                for conn_path in current_tree['paths_map'].values():
                    plan, eval_val, action, cost = self._calc_atnp(
                        current_tree, conn_path, d_idx, temp_state, nodes_on_tree
                    )

                    if plan.get('feasible') and eval_val > best_eval:
                        best_eval = eval_val
                        best_d = d_idx
                        best_plan = plan
                        best_action = action
                        best_cost = cost

            if best_d == -1:
                # 无法继续扩展
                break

            # 应用最佳分支
            current_tree['tree'] = np.logical_or(
                current_tree['tree'], best_plan['tree']
            ).astype(float)
            current_tree['hvt'] = np.maximum(
                current_tree['hvt'], best_plan['hvt']
            )
            current_tree['paths_map'][request['dest'][best_d]] = best_plan['new_path_full']
            current_tree['added_dest_indices'].append(best_d)
            nodes_on_tree.update(best_plan['new_path_full'])

            # 更新临时状态
            self._apply_deployment_to_state(request, best_plan['tree'], best_plan['hvt'], temp_state)

            # 记录轨迹
            trajectory.append((best_d, best_action, best_cost))

            unadded.remove(best_d)

        # 检查是否完成所有目的地
        if unadded:
            return None, []

        return current_tree, trajectory

    def _apply_deployment_to_state(self, request: Dict, tree_vec: np.ndarray,
                                   hvt: np.ndarray, state: Dict):
        """将部署应用到状态 (用于模拟)"""
        # 占用带宽
        for lid_idx, used in enumerate(tree_vec):
            if used > 0 and lid_idx < len(state['bw']):
                if state['bw_ref_count'][lid_idx] == 0:
                    state['bw'][lid_idx] -= request['bw_origin']
                state['bw_ref_count'][lid_idx] += 1

        # 占用CPU/内存
        for node_idx in range(hvt.shape[0]):
            for vnf_idx in range(hvt.shape[1]):
                if hvt[node_idx, vnf_idx] > 0 and state['hvt'][node_idx, vnf_idx] == 0:
                    vnf_type = vnf_idx + 1
                    if vnf_type in request['vnf']:
                        try:
                            req_idx = request['vnf'].index(vnf_type)
                            state['cpu'][node_idx] -= request['cpu_origin'][req_idx]
                            state['mem'][node_idx] -= request['memory_origin'][req_idx]
                        except:
                            pass
                    state['hvt'][node_idx, vnf_idx] += 1


if __name__ == "__main__":
    print("✅ 专家模块加载成功")