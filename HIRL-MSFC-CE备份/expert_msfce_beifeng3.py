#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce.py
# ✅ 最终修复版 (Vx) - 含高优先级补丁：
#  - 修复 self-loop (src==dst)
#  - _try_deploy_vnf 返回详细失败原因（便于记录与分析）
#  - _apply_path_to_tree 在扣减资源前做严格检查并抛错（提前发现异常）
#  - 路径评分缓存 LRU 限制（防止内存暴涨）
#  - _construct_tree 增加迭代/超时限制，避免单请求耗时过长
#  - 兼容原有接口与训练流程

import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import copy
import logging
import sys

import numpy as np
import scipy.io as sio

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Expert] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 配 ---
ALPHA = 0.3
BETA = 0.3
GAMMA = 0.4
CANDIDATE_SET_SIZE = 4
LOOKAHEAD_DEPTH = 3

# Cache config
DEFAULT_MAX_CACHE_SIZE = 5000

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
                 max_cache_size: int = DEFAULT_MAX_CACHE_SIZE):
        self.path_db = None

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

        # 保持与旧代码兼容
        self.k_path_count = 5
        self.k_path = 5

        # 路径评分缓存（仅缓存静态分量：跳数和DC计数）
        self._path_eval_cache = OrderedDict()
        self.MAX_CACHE_SIZE = int(max_cache_size)

        # metrics placeholder
        self.metrics = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'failure_reasons': {}
        }

    def clear_cache(self):
        self._path_eval_cache.clear()
        logger.info("Path evaluation cache cleared")

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
        """获取路径信息，处理自环和空条目 (src,dst 为 1-based)"""
        if self.path_db is None:
            return [], 0, []

        # 处理自环：把自环视为 trivial path（便于上层逻辑不反复报错）
        if src == dst:
            return [src], 0, []

        try:
            # 保护性访问，防止索引越界产生大量日志
            cell = self.path_db[src - 1, dst - 1]
            # 如果 cell 结构异常也捕获
            dist = int(cell['pathsdistance'][k - 1][0])
            nodes = cell['paths'][k - 1, :dist + 1].astype(int).tolist()
            links = []
            for i in range(len(nodes) - 1):
                u, v = nodes[i], nodes[i + 1]
                if (u, v) in self.link_map:
                    links.append(self.link_map[(u, v)])
            return nodes, dist, links
        except (IndexError, KeyError) as e:
            # 降低噪音为 WARNING，并包含 src==dst 已处理分支
            logger.warning(f"Path lookup failed for {src}->{dst} k={k}: {e}")
            return [], 0, []
        except Exception as e:
            logger.exception(f"Unexpected error in path lookup for {src}->{dst} k={k}: {e}")
            return [], 0, []

    def _get_max_hops(self, src: int, dst: int) -> int:
        try:
            cell = self.path_db[src - 1, dst - 1]
            return int(cell['pathsdistance'][self.k_path - 1][0])
        except:
            return 10

    def _normalize_state(self, state: Dict) -> Dict:
        normalized = {}
        normalized['bw'] = state.get('bw', state.get('bandwidth', np.full(self.link_num, self.cap_bw)))
        normalized['cpu'] = state.get('cpu', np.full(self.node_num, self.cap_cpu))
        normalized['mem'] = state.get('mem', state.get('memory', np.full(self.node_num, self.cap_mem)))
        normalized['hvt'] = state.get('hvt', state.get('hvt_all', np.zeros((self.node_num, self.type_num))))
        normalized['bw_ref_count'] = state.get('bw_ref_count', np.zeros(self.link_num))
        if 'request' in state:
            normalized['request'] = state['request']
        return normalized

    # 路径评分（含 LRU 缓存限制）
    def _calc_path_eval(self, nodes: List[int], links: List[int],
                        state: Dict, src_node: int, dst_node: int) -> float:
        if not nodes:
            return 0.0

        state = self._normalize_state(state)

        cache_key = (tuple(nodes), src_node, dst_node)

        # 静态部分缓存（term1, term2）
        if cache_key in self._path_eval_cache:
            # LRU move to end
            term1, term2 = self._path_eval_cache.pop(cache_key)
            self._path_eval_cache[cache_key] = (term1, term2)
        else:
            max_hops = self._get_max_hops(src_node, dst_node)
            current_hops = len(nodes) - 1
            term1 = 1.0 - (current_hops / max(1, max_hops))

            dc_count = sum(1 for n in nodes if n in self.DC)
            term2 = dc_count / max(1, self.dc_num)

            # 存入缓存并限制大小
            self._path_eval_cache[cache_key] = (term1, term2)
            if len(self._path_eval_cache) > self.MAX_CACHE_SIZE:
                # pop oldest
                self._path_eval_cache.popitem(last=False)

        # 动态部分 term3（资源相关）
        sr_val = 0.0
        for n in nodes:
            if n in self.DC:
                idx = n - 1
                if idx < len(state['cpu']):
                    sr_val += (state['cpu'][idx] + state['mem'][idx]) / (self.cap_cpu + self.cap_mem)
        for lid in links:
            idx = lid - 1
            if idx < len(state['bw']):
                sr_val += state['bw'][idx] / self.cap_bw

        norm_factor = max(1, len(nodes) + len(links))
        term3 = sr_val / norm_factor

        return float(ALPHA * term1 + BETA * term2 + GAMMA * term3)

    def _try_deploy_vnf(self, request: Dict, path_nodes: List[int],
                        state: Dict, existing_hvt: np.ndarray) -> Tuple[bool, np.ndarray, Dict, Dict]:
        """
        尝试部署 VNF 并返回资源扣减详情
        Returns:
           (feasible: bool, new_hvt: np.ndarray, placement: Dict, resource_delta_or_reason: Dict)
        如果不可行，resource_delta_or_reason 为失败 reason dict（包含 type 字段）
        """
        state = self._normalize_state(state)
        req_vnfs = request['vnf']
        hvt = existing_hvt.copy()
        placement = {}
        path_dcs = [n for n in path_nodes if n in self.DC]

        cpu_delta = np.zeros(self.node_num)
        mem_delta = np.zeros(self.node_num)

        # 详细失败原因：DC 数不足
        if len(path_dcs) < len(req_vnfs):
            reason = {
                'type': 'not_enough_dcs',
                'required_vnfs': len(req_vnfs),
                'available_dcs_on_path': len(path_dcs),
                'path_nodes': path_nodes
            }
            logger.debug(f"Req {request.get('id', '?')}: not_enough_dcs - need {reason['required_vnfs']}, have {reason['available_dcs_on_path']}")
            return False, existing_hvt, {}, reason

        current_dc_idx = 0
        for v_idx, v_type in enumerate(req_vnfs):
            deployed = False
            # 1. 复用已有实例
            for node in path_dcs:
                node_idx = node - 1
                if node_idx < hvt.shape[0] and (v_type - 1) < hvt.shape[1]:
                    if hvt[node_idx, v_type - 1] > 0:
                        placement[v_idx] = node
                        deployed = True
                        break
            if deployed:
                continue

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

                curr_cpu = state['cpu'][node_idx] - cpu_delta[node_idx]
                curr_mem = state['mem'][node_idx] - mem_delta[node_idx]

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

            # 资源不足 -> 记录详细失败原因并返回
            if not deployed:
                reason = {
                    'type': 'resource_shortage',
                    'vnf_idx': v_idx,
                    'vnf_type': v_type,
                    'cpu_required': request['cpu_origin'][v_idx],
                    'mem_required': request['memory_origin'][v_idx],
                    'checked_nodes_remaining': [path_dcs[i] for i in range(current_dc_idx, len(path_dcs))],
                    'cpu_delta': cpu_delta.tolist(),
                    'mem_delta': mem_delta.tolist()
                }
                logger.debug(f"Req {request.get('id', '?')}: VNF {v_idx} (type {v_type}) deployment failed - resource_shortage")
                return False, existing_hvt, {}, reason

        resource_delta = {'cpu': cpu_delta, 'mem': mem_delta}
        return True, hvt, placement, resource_delta

    def _evaluate_tree_cost(self, request: Dict, tree_links: np.ndarray, hvt: np.ndarray) -> float:
        node_cost = np.sum(hvt) * 1.0
        link_cost = np.sum(tree_links) * 1.0
        # 保持旧权重函数
        return 0.2 * (link_cost / 90.0) + 0.8 * (node_cost / 8.0)

    # ========== 接口兼容 (训练/环境) ==========
    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """Stage 1: Source -> Dest; 返回与旧 env 兼容的 8 个值"""
        state = self._normalize_state(state)
        src = request['source']
        dst = request['dest'][d_idx]
        nodes, dist, links = self._get_path_info(src, dst, k)
        if not nodes:
            return 0.0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dst, 0.0, {}

        score = self._calc_path_eval(nodes, links, state, src, dst)

        temp_state = copy.deepcopy(state)
        feasible, new_hvt, placement, res_info = self._try_deploy_vnf(request, nodes, temp_state,
                                                                     np.zeros((self.node_num, self.type_num)))

        tree_vec = np.zeros(self.link_num)
        if feasible:
            for lid in links:
                if lid - 1 < len(tree_vec):
                    tree_vec[lid - 1] = 1
        cost = self._evaluate_tree_cost(request, tree_vec, new_hvt) if feasible else 0.0

        # 返回：score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement
        # 如果不 feasible，res_info 为失败原因 dict（但这里按历史接口只返回 placement）
        return score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement

    def _calc_atnp(self, current_tree: Dict, conn_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """Stage 2"""
        state = self._normalize_state(state)
        request = state.get('request')
        if request is None:
            return {'feasible': False}, 0.0, (0, 0), 0.0

        best_eval = -1.0
        best_res = None
        best_action = (0, 0)

        for i_idx, conn_node in enumerate(conn_path):
            for k in range(1, self.k_path + 1):
                nodes, dist, links = self._get_path_info(conn_node, request['dest'][d_idx], k)
                if not nodes or len(nodes) < 2:
                    continue
                if set(nodes[1:]) & nodes_on_tree:
                    continue

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
                            if lid - 1 < len(tree_vec):
                                tree_vec[lid - 1] = 1
                        best_res = {
                            'tree': tree_vec, 'hvt': new_hvt, 'new_path_full': nodes,
                            'feasible': True, 'placement': placement, 'res_delta': res_delta
                        }
                        best_action = (i_idx, k - 1)

        if best_res:
            cost = self._evaluate_tree_cost(request, best_res['tree'], best_res['hvt'])
            return best_res, best_eval, best_action, cost
        else:
            return {'feasible': False}, 0.0, (0, 0), 0.0

    # ========== 构造树（含 Look-ahead） ==========
    def _construct_tree(self, request: Dict, network_state: Dict,
                        forced_first_dest_idx: Optional[int] = None) -> Tuple[Optional[Dict], List]:
        """
        返回 (tree_dict or None, traj_or_failed_unadded_list)
        增加：迭代和超时限制，防止单请求无限耗时
        """
        start_time = time.time()
        MAX_ITERATIONS = 200
        MAX_TIME_SECONDS = 20.0

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

        MAX_CANDIDATES = CANDIDATE_SET_SIZE * 3

        iteration_count = 0
        while len(current_tree['added_dest_indices']) < len(dest_indices):
            iteration_count += 1
            # iteration/time guard
            if iteration_count > MAX_ITERATIONS:
                logger.warning(f"Req {request.get('id', '?')}: Max iterations ({MAX_ITERATIONS}) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            if time.time() - start_time > MAX_TIME_SECONDS:
                logger.warning(f"Req {request.get('id', '?')}: Timeout ({MAX_TIME_SECONDS}s) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            unadded = [d for d in dest_indices if d not in current_tree['added_dest_indices']]
            candidates = []

            # A. 生成候选
            if not ordered_paths:
                targets = [forced_first_dest_idx] if forced_first_dest_idx is not None else unadded
                for d_idx in targets:
                    if d_idx not in unadded:
                        continue
                    if len(candidates) >= MAX_CANDIDATES:
                        break
                    for k in range(1, self.k_path + 1):
                        score, nodes, t_vec, h_vec, feas, _, cost, pl = self._calc_eval(request, d_idx, k, current_sim_state)
                        if feas:
                            # 计算 res_delta（_try_deploy_vnf 已返回 detailed info internally if fail）
                            _, _, _, res_delta = self._try_deploy_vnf(request, nodes, current_sim_state,
                                                                      np.zeros((self.node_num, self.type_num)))
                            info = {'nodes': nodes, 'links': [], 'k': k, 'score': score, 'p_idx': 0,
                                    'res_delta': res_delta, 'hvt': h_vec, 'tree_vec': t_vec, 'placement': pl,
                                    'd_idx': d_idx}
                            candidates.append(info)
            else:
                for p_idx, path in enumerate(ordered_paths):
                    for d_idx in unadded:
                        if len(candidates) >= MAX_CANDIDATES:
                            break
                        res, score, action_in_path, _ = self._calc_atnp(current_tree, path, d_idx, current_sim_state,
                                                                        current_tree['nodes'])
                        if res and res.get('feasible'):
                            k = action_in_path[1] + 1
                            info = {'nodes': res['new_path_full'], 'k': k, 'score': score, 'p_idx': p_idx,
                                    'conn_idx_in_path': action_in_path[0],
                                    'res_delta': res['res_delta'], 'hvt': res['hvt'], 'tree_vec': res['tree'],
                                    'placement': res['placement'], 'd_idx': d_idx}
                            candidates.append(info)

            if not candidates:
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            # B. 筛选
            candidates.sort(key=lambda x: x['score'], reverse=True)
            candidate_set = candidates[:CANDIDATE_SET_SIZE]

            # C. Look-ahead 选择
            best_global_otv = float('inf')
            selected_info = None

            # dynamic depth
            current_lookahead_depth = min(LOOKAHEAD_DEPTH, max(0, len(unadded) - 1))

            for info in candidate_set:
                d_idx = info['d_idx']

                temp_tree_sim = copy.deepcopy(current_tree)
                temp_state_sim = copy.deepcopy(current_sim_state)

                # apply candidate simulation using resource_delta if any
                self._apply_path_to_tree(temp_tree_sim, info, request, temp_state_sim,
                                         real_deploy=True, resource_delta=info.get('res_delta'))

                # greedy lookahead expansion (depth-limited)
                remaining_after = [d for d in unadded if d != d_idx]
                subsequent_count = 0
                while subsequent_count < current_lookahead_depth and remaining_after:
                    next_candidates = []
                    # obtain simulated paths
                    current_sim_paths = list(temp_tree_sim['paths_map'].values()) if temp_tree_sim['paths_map'] else [[request['source']]]

                    for next_d_idx in remaining_after:
                        for path in current_sim_paths:
                            res, score, _, _ = self._calc_atnp(temp_tree_sim, path, next_d_idx, temp_state_sim, temp_tree_sim['nodes'])
                            if res and res.get('feasible'):
                                next_candidates.append((score, next_d_idx, res))

                    if not next_candidates:
                        break

                    next_candidates.sort(key=lambda x: x[0], reverse=True)
                    best_score, best_next_d, best_res = next_candidates[0]

                    temp_info_next = {
                        'nodes': best_res['new_path_full'],
                        'hvt': best_res['hvt'],
                        'tree_vec': best_res['tree']
                    }

                    self._apply_path_to_tree(temp_tree_sim, temp_info_next, request, temp_state_sim,
                                             real_deploy=True, resource_delta=best_res.get('res_delta'))
                    remaining_after.remove(best_next_d)
                    subsequent_count += 1

                otv = self._evaluate_tree_cost(request, temp_tree_sim['tree'], temp_tree_sim['hvt'])
                if otv < best_global_otv:
                    best_global_otv = otv
                    selected_info = info

            # D. 执行
            if selected_info:
                d_idx = selected_info['d_idx']
                self._apply_path_to_tree(current_tree, selected_info, request, current_sim_state,
                                         real_deploy=True, resource_delta=selected_info.get('res_delta'))

                current_tree['added_dest_indices'].append(d_idx)
                ordered_paths.append(selected_info['nodes'])

                p_idx = selected_info['p_idx']
                k_idx = selected_info['k'] - 1
                placement = selected_info.get('placement', {})

                action_tuple = (p_idx, k_idx, placement)
                cost = self._evaluate_tree_cost(request, current_tree['tree'], current_tree['hvt'])
                current_tree['traj'].append((d_idx, action_tuple, cost))
            else:
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

        return current_tree, current_tree['traj']

    def _apply_path_to_tree(self, tree_struct, info, request, state, real_deploy=False, resource_delta=None):
        """
        更新树结构和资源
        严格资源检查：断言形状、非负、可行性（提前抛错）
        """
        nodes = info['nodes']

        # 更新链路
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

        tree_struct['nodes'].update(nodes)
        dest_id = nodes[-1]
        tree_struct['paths_map'][dest_id] = nodes

        # 更新 HVT
        if 'hvt' in info:
            tree_struct['hvt'] = np.maximum(tree_struct['hvt'], info['hvt'])

        # 节点资源扣除（严格检查）
        if real_deploy and resource_delta:
            cpu_d = resource_delta.get('cpu', np.zeros(self.node_num))
            mem_d = resource_delta.get('mem', np.zeros(self.node_num))

            # 形状断言
            if cpu_d.shape[0] != self.node_num or mem_d.shape[0] != self.node_num:
                logger.error(f"Resource delta shape mismatch: cpu {cpu_d.shape}, mem {mem_d.shape}, expected {self.node_num}")
                raise AssertionError("Resource delta shape mismatch")

            # 非负性检查
            if not np.all(cpu_d >= 0) or not np.all(mem_d >= 0):
                logger.error("Negative resource delta detected")
                raise AssertionError("Negative resource delta detected")

            # 可行性检查（容忍微小浮点误差）
            cpu_after = state['cpu'] - cpu_d
            mem_after = state['mem'] - mem_d
            if np.any(cpu_after < -1e-8):
                logger.error(f"CPU would become negative: min={np.min(cpu_after):.6f}")
                raise ValueError("Resource deduction would violate CPU constraints")
            if np.any(mem_after < -1e-8):
                logger.error(f"MEM would become negative: min={np.min(mem_after):.6f}")
                raise ValueError("Resource deduction would violate MEM constraints")

            # 安全扣减
            state['cpu'] = np.maximum(cpu_after, 0.0)
            state['mem'] = np.maximum(mem_after, 0.0)

    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> Tuple[Optional[Dict], List]:
        """
        主入口：先做粗略资源预检查，再尝试构造树，如失败触发 recall 流程（可在外层扩展）
        """
        self.metrics['total_requests'] += 1

        network_state = self._normalize_state(network_state)
        network_state['request'] = request

        # 快速全局资源检查（仅作为粗略过滤）
        total_cpu_req = sum(request['cpu_origin'])
        total_mem_req = sum(request['memory_origin'])
        if total_cpu_req > np.sum(network_state['cpu']) or total_mem_req > np.sum(network_state['mem']):
            logger.debug(f"Req {request.get('id','?')}: skipped by global resource check")
            self.metrics['rejected'] += 1
            self.metrics['failure_reasons'].setdefault('global_resource_shortage', 0)
            self.metrics['failure_reasons']['global_resource_shortage'] += 1
            return None, []

        res_tree, res_traj = self._construct_tree(request, network_state)
        if res_tree is not None:
            self.metrics['accepted'] += 1
            return res_tree, res_traj

        # recall attempts: try forcing each failed dest first (simple heuristic)
        failed_unadded = res_traj
        if failed_unadded:
            for fail_idx in failed_unadded:
                recall_tree, recall_traj = self._construct_tree(request, network_state, forced_first_dest_idx=fail_idx)
                if recall_tree is not None:
                    self.metrics['accepted'] += 1
                    return recall_tree, recall_traj

        self.metrics['rejected'] += 1
        self.metrics['failure_reasons'].setdefault('construct_tree_failed', 0)
        self.metrics['failure_reasons']['construct_tree_failed'] += 1
        return None, []

    def _recall_strategy(self, request: Dict, network_state: Dict):
        logger.warning(f"Recall triggered for request {request.get('id','?')}. No feasible path found.")
        return None, []

if __name__ == "__main__":
    pass
