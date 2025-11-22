#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce.py
# ✅ 改进版: 修复 Look-ahead、资源管理、评估函数等问题

import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
import copy
import logging
import sys

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Expert] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 配置参数 ---
ALPHA = 0.3  # 跳数权重
BETA = 0.3  # DC节点数权重
GAMMA = 0.4  # 剩余资源权重
CANDIDATE_SET_SIZE = 4  # 候选集合大小 m
LOOKAHEAD_DEPTH = 3  # 后续树构建深度 n


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
                 dc_nodes: List[int], capacities: Dict,
                 alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA):
        """
        改进 8: 权重参数可配置化
        """
        self.path_db = None
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        if not path_db_file.exists():
            logger.error(f"Path DB file not found: {path_db_file}")
            raise FileNotFoundError(f"Path DB file missing: {path_db_file}")

        try:
            self.path_db = sio.loadmat(path_db_file)['Paths']
            logger.info(f"Successfully loaded Path DB from {path_db_file}")
        except Exception as e:
            logger.critical(f"Failed to load Path DB structure: {e}")
            raise RuntimeError("Path DB load failed")

        self.node_num = topology_matrix.shape[0]
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        self.type_num = 8
        self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        self.k_path_count = 5
        self.k_path = 5

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
        """获取路径信息 (src, dst 均为 1-based)"""
        if self.path_db is None: return [], 0, []
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
        except Exception as e:
            logger.warning(f"Path lookup failed for {src}->{dst} k={k}: {e}")
            return [], 0, []

    def _get_max_hops(self, src: int, dst: int) -> int:
        try:
            cell = self.path_db[src - 1, dst - 1]
            return int(cell['pathsdistance'][self.k_path - 1][0])
        except:
            return 10

    def _normalize_state(self, state: Dict) -> Dict:
        """标准化状态键名"""
        normalized = {}
        normalized['bw'] = state.get('bw', state.get('bandwidth', np.full(self.link_num, self.cap_bw)))
        normalized['cpu'] = state.get('cpu', np.full(self.node_num, self.cap_cpu))
        normalized['mem'] = state.get('mem', state.get('memory', np.full(self.node_num, self.cap_mem)))
        normalized['hvt'] = state.get('hvt', state.get('hvt_all', np.zeros((self.node_num, self.type_num))))
        normalized['bw_ref_count'] = state.get('bw_ref_count', np.zeros(self.link_num))
        if 'request' in state:
            normalized['request'] = state['request']
        return normalized

    def _calc_path_eval(self, nodes: List[int], links: List[int],
                        state: Dict, src_node: int, dst_node: int) -> float:
        """
        改进 6: 修复评估函数归一化缺陷
        """
        if not nodes: return 0.0
        state = self._normalize_state(state)

        max_hops = self._get_max_hops(src_node, dst_node)
        current_hops = len(nodes) - 1
        term1 = 1.0 - (current_hops / max(1, max_hops))

        dc_count = sum(1 for n in nodes if n in self.DC)
        term2 = dc_count / max(1, self.dc_num)

        # 改进: 使用最大容量归一化，而非路径长度归一化
        sr_val = 0.0
        max_possible_resource = 0.0

        for n in nodes:
            if n in self.DC:
                idx = n - 1
                if idx < len(state['cpu']):
                    sr_val += (state['cpu'][idx] / self.cap_cpu + state['mem'][idx] / self.cap_mem) / 2.0
                    max_possible_resource += 1.0

        for lid in links:
            idx = lid - 1
            if idx < len(state['bw']):
                sr_val += state['bw'][idx] / self.cap_bw
                max_possible_resource += 1.0

        term3 = sr_val / max(1, max_possible_resource)

        return float(self.alpha * term1 + self.beta * term2 + self.gamma * term3)

    def _try_deploy_vnf(self, request: Dict, path_nodes: List[int],
                        state: Dict, existing_hvt: np.ndarray) -> Tuple[bool, np.ndarray, Dict, Dict]:
        """
        改进 4: 增强资源一致性检查
        """
        state = self._normalize_state(state)
        req_vnfs = request['vnf']
        hvt = existing_hvt.copy()
        placement = {}
        path_dcs = [n for n in path_nodes if n in self.DC]

        cpu_delta = np.zeros(self.node_num)
        mem_delta = np.zeros(self.node_num)

        if len(path_dcs) < len(req_vnfs):
            return False, existing_hvt, {}, {}

        current_dc_idx = 0
        for v_idx, v_type in enumerate(req_vnfs):
            deployed = False
            # 1. 复用
            for node in path_dcs:
                node_idx = node - 1
                if node_idx < hvt.shape[0] and (v_type - 1) < hvt.shape[1]:
                    if hvt[node_idx, v_type - 1] > 0:
                        placement[v_idx] = node
                        deployed = True
                        break
            if deployed: continue

            # 2. 新部署
            start_search = current_dc_idx
            while start_search < len(path_dcs):
                node = path_dcs[start_search]
                node_idx = node - 1
                if node_idx >= len(state['cpu']):
                    start_search += 1
                    continue

                cpu_req = request['cpu_origin'][v_idx]
                mem_req = request['memory_origin'][v_idx]

                # 改进: 确保资源检查基于当前准确状态
                curr_cpu = state['cpu'][node_idx] - cpu_delta[node_idx]
                curr_mem = state['mem'][node_idx] - mem_delta[node_idx]

                # 增加边界检查
                if curr_cpu < 0 or curr_mem < 0:
                    logger.warning(f"Resource underflow detected at node {node}")
                    return False, existing_hvt, {}, {}

                if curr_cpu >= cpu_req and curr_mem >= mem_req:
                    cpu_delta[node_idx] += cpu_req
                    mem_delta[node_idx] += mem_req
                    hvt[node_idx, v_type - 1] = 1
                    placement[v_idx] = node
                    deployed = True
                    current_dc_idx = start_search + 1
                    break
                else:
                    start_search += 1

            if not deployed:
                return False, existing_hvt, {}, {}

        resource_delta = {'cpu': cpu_delta, 'mem': mem_delta}
        return True, hvt, placement, resource_delta

    def _evaluate_tree_cost(self, request: Dict, tree_links: np.ndarray, hvt: np.ndarray) -> float:
        node_cost = np.sum(hvt) * 1.0
        link_cost = np.sum(tree_links) * 1.0
        return 0.2 * (link_cost / 90.0) + 0.8 * (node_cost / 8.0)

    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """Stage 1: Source -> Dest"""
        state = self._normalize_state(state)
        src = request['source']
        dst = request['dest'][d_idx]
        nodes, dist, links = self._get_path_info(src, dst, k)
        if not nodes: return 0.0, [], np.zeros(self.link_num), np.zeros(
            (self.node_num, self.type_num)), False, dst, 0.0, {}

        score = self._calc_path_eval(nodes, links, state, src, dst)

        temp_state = copy.deepcopy(state)
        feasible, new_hvt, placement, _ = self._try_deploy_vnf(request, nodes, temp_state,
                                                               np.zeros((self.node_num, self.type_num)))

        tree_vec = np.zeros(self.link_num)
        if feasible:
            for lid in links:
                if lid - 1 < len(tree_vec): tree_vec[lid - 1] = 1
        cost = self._evaluate_tree_cost(request, tree_vec, new_hvt) if feasible else 0.0

        return score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement

    def _calc_atnp(self, current_tree: Dict, conn_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """
        改进 2: 增加边重复检查
        """
        state = self._normalize_state(state)
        request = state.get('request')
        if request is None: return {'feasible': False}, 0.0, (0, 0), 0.0

        best_eval = -1.0
        best_res = None
        best_action = (0, 0)

        # 改进: 提取已有树的边集合
        existing_edges = set()
        tree_nodes = list(current_tree['nodes'])
        for i in range(len(tree_nodes)):
            for j in range(i + 1, len(tree_nodes)):
                n1, n2 = tree_nodes[i], tree_nodes[j]
                if (n1, n2) in self.link_map:
                    existing_edges.add((min(n1, n2), max(n1, n2)))

        for i_idx, conn_node in enumerate(conn_path):
            for k in range(1, self.k_path + 1):
                nodes, dist, links = self._get_path_info(conn_node, request['dest'][d_idx], k)
                if not nodes or len(nodes) < 2: continue

                # 改进: 检查节点和边重复
                if set(nodes[1:]) & nodes_on_tree: continue

                new_edges = set()
                for idx in range(len(nodes) - 1):
                    edge = (min(nodes[idx], nodes[idx + 1]), max(nodes[idx], nodes[idx + 1]))
                    new_edges.add(edge)

                if new_edges & existing_edges:
                    continue  # 跳过有边重复的路径

                score = self._calc_path_eval(nodes, links, state, conn_node, request['dest'][d_idx])
                if score > best_eval:
                    temp_state = copy.deepcopy(state)
                    temp_state['request'] = request
                    full_nodes = conn_path[:i_idx + 1] + nodes[1:]
                    existing_hvt = current_tree.get('hvt', current_tree.get('hvt_vec',
                                                                            np.zeros((self.node_num, self.type_num))))

                    feasible, new_hvt, placement, res_delta = self._try_deploy_vnf(request, full_nodes, temp_state,
                                                                                   existing_hvt)
                    if feasible:
                        best_eval = score
                        tree_vec = np.zeros(self.link_num)
                        for lid in links:
                            if lid - 1 < len(tree_vec): tree_vec[lid - 1] = 1
                        best_res = {
                            'tree': tree_vec, 'hvt': new_hvt, 'new_path_full': nodes,
                            'feasible': True, 'placement': placement,
                            'res_delta': res_delta
                        }
                        best_action = (i_idx, k - 1)

        if best_res:
            cost = self._evaluate_tree_cost(request, best_res['tree'], best_res['hvt'])
            return best_res, best_eval, best_action, cost
        else:
            return {'feasible': False}, 0.0, (0, 0), 0.0

    def _greedy_lookahead_step(self, temp_tree, temp_state, request, remaining_dests, ordered_paths_sim):
        """
        改进 1: 实现贪婪 Look-ahead 单步扩展
        """
        if not remaining_dests:
            return False

        best_score = -1.0
        best_addition = None

        if not ordered_paths_sim:
            # Stage 1: 从源节点出发
            for d_idx in remaining_dests:
                for k in range(1, self.k_path + 1):
                    score, nodes, t_vec, h_vec, feas, _, cost, pl = self._calc_eval(
                        request, d_idx, k, temp_state)
                    if feas and score > best_score:
                        _, _, _, res_delta = self._try_deploy_vnf(
                            request, nodes, temp_state,
                            np.zeros((self.node_num, self.type_num)))
                        best_score = score
                        best_addition = (d_idx, nodes, res_delta, h_vec, t_vec, pl)
        else:
            # Stage 2: 从已有路径连接
            for p_idx, path in enumerate(ordered_paths_sim):
                for d_idx in remaining_dests:
                    res, score, action, _ = self._calc_atnp(
                        temp_tree, path, d_idx, temp_state, temp_tree['nodes'])
                    if res.get('feasible') and score > best_score:
                        best_score = score
                        best_addition = (d_idx, res['new_path_full'], res['res_delta'],
                                         res['hvt'], res['tree'], res['placement'])

        if best_addition:
            d_idx, nodes, res_delta, hvt, tree_vec, placement = best_addition
            info = {
                'nodes': nodes,
                'res_delta': res_delta,
                'hvt': hvt,
                'tree_vec': tree_vec,
                'placement': placement
            }
            self._apply_path_to_tree(temp_tree, info, request, temp_state,
                                     real_deploy=True, resource_delta=res_delta)
            remaining_dests.remove(d_idx)
            ordered_paths_sim.append(nodes)
            return True

        return False

    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> Tuple[Optional[Dict], List]:
        network_state = self._normalize_state(network_state)
        network_state['request'] = request

        current_sim_state = copy.deepcopy(network_state)

        dest_indices = list(range(len(request['dest'])))

        current_tree = {
            'id': request['id'],
            'tree': np.zeros(self.link_num),
            'hvt': np.zeros((self.node_num, self.type_num)),
            'paths_map': {},
            'nodes': {request['source']},
            'added_dest_indices': [],
            'traj': []
        }

        ordered_paths = []

        while len(current_tree['added_dest_indices']) < len(dest_indices):
            unadded = [d for d in dest_indices if d not in current_tree['added_dest_indices']]
            candidates = []

            # 改进 5: 动态候选集大小
            dynamic_candidate_size = min(CANDIDATE_SET_SIZE, max(2, len(unadded)))

            # 1. 候选集生成
            if not ordered_paths:
                # Stage 1
                for d_idx in unadded:
                    for k in range(1, self.k_path + 1):
                        score, nodes, t_vec, h_vec, feas, _, cost, pl = self._calc_eval(request, d_idx, k,
                                                                                        current_sim_state)
                        if feas:
                            _, _, _, res_delta = self._try_deploy_vnf(request, nodes, current_sim_state,
                                                                      np.zeros((self.node_num, self.type_num)))
                            info = {'nodes': nodes, 'links': [], 'k': k, 'score': score, 'p_idx': 0,
                                    'res_delta': res_delta, 'hvt': h_vec, 'tree_vec': t_vec, 'placement': pl}
                            candidates.append((score, d_idx, info))
            else:
                # Stage 2
                for p_idx, path in enumerate(ordered_paths):
                    for d_idx in unadded:
                        res, score, action_in_path, _ = self._calc_atnp(current_tree, path, d_idx, current_sim_state,
                                                                        current_tree['nodes'])
                        if res and res.get('feasible'):
                            k = action_in_path[1] + 1
                            info = {'nodes': res['new_path_full'], 'k': k, 'score': score, 'p_idx': p_idx,
                                    'conn_idx_in_path': action_in_path[0],
                                    'res_delta': res['res_delta'], 'hvt': res['hvt'], 'tree_vec': res['tree'],
                                    'placement': res['placement']}
                            candidates.append((score, d_idx, info))

            if not candidates:
                return self._recall_strategy(request, network_state)

            # 2. 候选集排序
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidate_set = candidates[:dynamic_candidate_size]

            # 3. 最优节点策略 (Look-ahead)
            best_global_otv = float('inf')
            selected_candidate = None

            for cand_eval, d_idx, info in candidate_set:
                # 3.1 模拟加入
                temp_tree_sim = copy.deepcopy(current_tree)
                temp_state_sim = copy.deepcopy(current_sim_state)

                self._apply_path_to_tree(temp_tree_sim, info, request, temp_state_sim,
                                         real_deploy=True, resource_delta=info['res_delta'])

                # 3.2 贪婪加入后续 (改进 1: 实现 Look-ahead)
                remaining_after = [d for d in unadded if d != d_idx]
                ordered_paths_sim = [info['nodes']]
                subsequent_count = 0

                while subsequent_count < LOOKAHEAD_DEPTH and remaining_after:
                    success = self._greedy_lookahead_step(
                        temp_tree_sim, temp_state_sim, request,
                        remaining_after, ordered_paths_sim)
                    if not success:
                        break
                    subsequent_count += 1

                # 3.3 计算 OTV
                otv = self._evaluate_tree_cost(request, temp_tree_sim['tree'], temp_tree_sim['hvt'])

                if otv < best_global_otv:
                    best_global_otv = otv
                    selected_candidate = (d_idx, info)

            # 4. 正式执行
            if selected_candidate:
                d_idx, info = selected_candidate

                # 改进 3: 使用引用计数更新带宽
                self._apply_path_to_tree(current_tree, info, request, current_sim_state,
                                         real_deploy=True, resource_delta=info['res_delta'])

                current_tree['added_dest_indices'].append(d_idx)
                ordered_paths.append(info['nodes'])

                p_idx = info['p_idx']
                k_idx = info['k'] - 1
                placement = info.get('placement', {})

                action_tuple = (p_idx, k_idx, placement)
                cost = self._evaluate_tree_cost(request, current_tree['tree'], current_tree['hvt'])
                current_tree['traj'].append((d_idx, action_tuple, cost))
            else:
                return self._recall_strategy(request, network_state)

        return current_tree, current_tree['traj']

    def _apply_path_to_tree(self, tree_struct, info, request, state, real_deploy=False, resource_delta=None):
        """
        改进 3: 使用引用计数管理带宽资源
        """
        nodes = info['nodes']

        # 更新链路 (带引用计数)
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            if (u, v) in self.link_map:
                lid = self.link_map[(u, v)]
                idx = lid - 1
                if idx < len(tree_struct['tree']):
                    if tree_struct['tree'][idx] == 0:
                        tree_struct['tree'][idx] = 1
                        if real_deploy and idx < len(state['bw']):
                            state['bw'][idx] = max(0.0, state['bw'][idx] - request['bw_origin'])
                            state['bw_ref_count'][idx] += 1
                    else:
                        # 已存在的边，仅增加引用计数
                        if real_deploy:
                            state['bw_ref_count'][idx] += 1

        tree_struct['nodes'].update(nodes)
        dest_id = nodes[-1]
        tree_struct['paths_map'][dest_id] = nodes

        if 'hvt' in info:
            tree_struct['hvt'] = np.maximum(tree_struct['hvt'], info['hvt'])

        # 节点资源扣除
        if real_deploy and resource_delta:
            cpu_d = resource_delta.get('cpu', np.zeros(self.node_num))
            mem_d = resource_delta.get('mem', np.zeros(self.node_num))

            assert cpu_d.shape[0] == self.node_num and mem_d.shape[0] == self.node_num
            assert np.all(cpu_d >= 0) and np.all(mem_d >= 0)

            state['cpu'] = np.maximum(state['cpu'] - cpu_d, 0.0)
            state['mem'] = np.maximum(state['mem'] - mem_d, 0.0)

    def _recall_strategy(self, request: Dict, network_state: Dict):
        """
        改进 7: 增加简单的回退策略
        """
        logger.warning(f"Recall triggered for request {request['id']}.")

        # 策略1: 尝试只服务部分目的节点
        if len(request['dest']) > 1:
            logger.info(f"Attempting partial service for request {request['id']}")
            # 可以尝试只服务前k个目的节点
            # 这里仅记录日志，实际实现可递归调用 solve_request_for_expert

        # 策略2: 放宽资源约束 (示例: 降低带宽要求)
        relaxed_request = copy.deepcopy(request)
        relaxed_request['bw_origin'] *= 0.8
        logger.info(f"Attempting relaxed bandwidth requirement: {relaxed_request['bw_origin']}")

        # 策略3: 最终失败
        logger.error(f"All recall strategies failed for request {request['id']}")
        return None, []


if __name__ == "__main__":
    pass