#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce.py - BEST OF BOTH WORLDS
"""
融合版本特性：
1. ✅ Document 4 的 SolverConfig 配置类
2. ✅ Document 4 的 Rollback 机制
3. ✅ Document 4 的增强 Recall 策略
4. ✅ Document 5 的简洁代码结构
5. ✅ Document 4 的完整资源验证（含带宽）
6. ✅ Document 5 的清晰注释风格
"""

from __future__ import annotations
import time
import copy
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

import numpy as np
import scipy.io as sio

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Expert] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ========== 配置类（来自 Document 4）==========
@dataclass
class SolverConfig:
    """集中式配置管理"""
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.4
    candidate_set_size: int = 4
    lookahead_depth: int = 3
    k_path: int = 5
    max_cache_size: int = 5000
    max_iterations: int = 200
    max_time_seconds: float = 20.0
    max_candidates: int = 12
    otv_link_weight: float = 0.2
    otv_node_weight: float = 0.8
    otv_norm_link: float = 90.0
    otv_norm_node: float = 8.0

    def __post_init__(self):
        """参数验证"""
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1 and 0 <= self.gamma <= 1):
            raise ValueError("Alpha, beta, gamma must be between 0 and 1")
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 1e-6:
            logger.warning("Score weights do not sum to 1.0")


def parse_mat_request(req_obj) -> Dict:
    """解析请求（兼容 Python Dict 和 MATLAB 格式）"""
    # 1. 如果已经是 Python 字典 (来自 .pkl)，直接返回
    if isinstance(req_obj, dict):
        return req_obj

    # 2. 否则尝试解析 MATLAB 格式 (保留原有逻辑以防万一)
    try:
        return {
            'id': int(req_obj['id'][0, 0]),
            'source': int(req_obj['source'][0, 0]),
            'dest': [int(d) for d in req_obj['dest'].flatten()],
            'vnf': [int(v) for v in req_obj['vnf'].flatten()],
            'bw_origin': float(req_obj['bw_origin'][0, 0]),
            'cpu_origin': [float(c) for c in req_obj['cpu_origin'].flatten()],
            'memory_origin': [float(m) for m in req_obj['memory_origin'].flatten()],
            'arrival_time': int(req_obj.get('arrival_time', [[0]])[0, 0]),
            'leave_time': int(req_obj.get('leave_time', [[0]])[0, 0]),
        }
    except:
        # 旧版兼容
        return {
            'id': int(req_obj[0][0][0]),
            'source': int(req_obj[0][1][0]),
            'dest': [int(x) for x in req_obj[0][2].flatten()],
            'vnf': [int(x) for x in req_obj[0][3].flatten()],
            'cpu_origin': [float(x) for x in req_obj[0][4].flatten()],
            'memory_origin': [float(x) for x in req_obj[0][5].flatten()],
            'bw_origin': float(req_obj[0][6][0][0])
        }
# def parse_mat_request(req_obj) -> Dict:
#     """解析 MATLAB 请求（兼容两种格式）"""
#     try:
#         return {
#             'id': int(req_obj['id'][0, 0]),
#             'source': int(req_obj['source'][0, 0]),
#             'dest': [int(d) for d in req_obj['dest'].flatten()],
#             'vnf': [int(v) for v in req_obj['vnf'].flatten()],
#             'bw_origin': float(req_obj['bw_origin'][0, 0]),
#             'cpu_origin': [float(c) for c in req_obj['cpu_origin'].flatten()],
#             'memory_origin': [float(m) for m in req_obj['memory_origin'].flatten()],
#             'arrival_time': int(req_obj.get('arrival_time', [[0]])[0, 0]),
#             'leave_time': int(req_obj.get('leave_time', [[0]])[0, 0]),
#         }
#     except:
#         return {
#             'id': int(req_obj[0][0][0]),
#             'source': int(req_obj[0][1][0]),
#             'dest': [int(x) for x in req_obj[0][2].flatten()],
#             'vnf': [int(x) for x in req_obj[0][3].flatten()],
#             'cpu_origin': [float(x) for x in req_obj[0][4].flatten()],
#             'memory_origin': [float(x) for x in req_obj[0][5].flatten()],
#             'bw_origin': float(req_obj[0][6][0][0])
#         }
#

class MSFCE_Solver:
    """MSFC-CE 专家算法求解器（融合增强版）"""

    def __init__(self, path_db_file: Path, topology_matrix: np.ndarray,
                 dc_nodes: List[int], capacities: Dict,
                 config: Optional[SolverConfig] = None):

        self.config = config or SolverConfig()

        # 加载 Path DB
        if not Path(path_db_file).exists():
            raise FileNotFoundError(f"Path DB missing: {path_db_file}")

        try:
            mat = sio.loadmat(path_db_file)
            self.path_db = mat['Paths']
            logger.info(f"Loaded Path DB from {path_db_file}")
        except Exception as e:
            raise RuntimeError(f"Path DB load failed: {e}")

        # 网络拓扑
        self.node_num = int(topology_matrix.shape[0])
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        # VNF 类型和 DC 节点
        self.type_num = 8
        self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        # 资源容量
        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        # K 条路径
        self.k_path = int(self.config.k_path)
        self.k_path_count = self.k_path  # 兼容旧代码

        # LRU 缓存
        self._path_eval_cache = OrderedDict()
        self.MAX_CACHE_SIZE = int(self.config.max_cache_size)

        # 性能指标
        self.metrics = {
            'total_requests': 0,
            'accepted': 0,
            'rejected': 0,
            'failure_reasons': {},
            'cache_hits': 0,
            'cache_misses': 0,
            'processing_times': [],
            'errors': 0,
        }

    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        """构建链路映射"""
        link_map = {}
        lid = 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

    def _normalize_state(self, state: Dict) -> Dict:
        """标准化状态字典"""
        normalized = {}
        normalized['bw'] = state.get('bw', state.get('bandwidth',
                                                     np.full(self.link_num, self.cap_bw)))
        normalized['cpu'] = state.get('cpu', np.full(self.node_num, self.cap_cpu))
        normalized['mem'] = state.get('mem', state.get('memory',
                                                       np.full(self.node_num, self.cap_mem)))
        normalized['hvt'] = state.get('hvt', state.get('hvt_all',
                                                       np.zeros((self.node_num, self.type_num))))
        normalized['bw_ref_count'] = state.get('bw_ref_count',
                                               np.zeros(self.link_num))
        if 'request' in state:
            normalized['request'] = state['request']
        return normalized

    # ========== 核心方法：路径查询（修复自环）==========
    def _get_path_info(self, src: int, dst: int, k: int):
        """
        获取路径信息（1-based 索引）
        ✅ 修复：处理 src==dst 自环情况
        """
        if self.path_db is None:
            return [], 0, []

        # ✅ 自环处理
        if src == dst:
            return [src], 0, []

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
        except (IndexError, KeyError) as e:
            logger.warning(f"Path lookup failed {src}->{dst} k={k}: {e}")
            return [], 0, []

    def _get_max_hops(self, src: int, dst: int) -> int:
        """获取最大跳数"""
        try:
            cell = self.path_db[src - 1, dst - 1]
            return int(cell['pathsdistance'][self.k_path - 1][0])
        except:
            return 10

    # ========== 路径评分（带 LRU 缓存）==========
    def _calc_path_eval(self, nodes: List[int], links: List[int],
                        state: Dict, src_node: int, dst_node: int) -> float:
        """
        计算路径评分（静态部分缓存 + 动态资源）
        ✅ Document 4 的缓存策略
        """
        if not nodes:
            return 0.0

        cache_key = (src_node, dst_node, tuple(nodes))

        # 缓存查询（LRU）
        if cache_key in self._path_eval_cache:
            term1, term2 = self._path_eval_cache.pop(cache_key)
            self._path_eval_cache[cache_key] = (term1, term2)
            self.metrics['cache_hits'] += 1
        else:
            # 静态部分计算
            max_hops = self._get_max_hops(src_node, dst_node)
            current_hops = len(nodes) - 1
            term1 = 1.0 - (current_hops / max(1, max_hops))

            dc_count = sum(1 for n in nodes if n in self.DC)
            term2 = dc_count / max(1, self.dc_num)

            # 存入缓存
            self._path_eval_cache[cache_key] = (term1, term2)
            self.metrics['cache_misses'] += 1

            # LRU 淘汰
            if len(self._path_eval_cache) > self.MAX_CACHE_SIZE:
                self._path_eval_cache.popitem(last=False)

        # 动态部分（实时资源）
        sr_val = 0.0
        for n in nodes:
            if n in self.DC:
                idx = n - 1
                if idx < len(state['cpu']):
                    sr_val += (state['cpu'][idx] + state['mem'][idx]) / \
                              (self.cap_cpu + self.cap_mem)
        for lid in links:
            idx = lid - 1
            if idx < len(state['bw']):
                sr_val += state['bw'][idx] / self.cap_bw

        norm_factor = max(1, len(nodes) + len(links))
        term3 = sr_val / norm_factor

        return float(self.config.alpha * term1 +
                     self.config.beta * term2 +
                     self.config.gamma * term3)

    # ========== VNF 部署（详细失败原因）==========
    def _try_deploy_vnf(self, request: Dict, path_nodes: List[int],
                        state: Dict, existing_hvt: np.ndarray) -> Tuple[bool, np.ndarray, Dict, Dict]:
        """
        尝试部署 VNF
        ✅ Document 5 的详细失败原因记录
        Returns: (feasible, hvt, placement, resource_delta_or_reason)
        """
        req_vnfs = request.get('vnf', [])
        hvt = existing_hvt.copy()
        placement = {}
        path_dcs = [n for n in path_nodes if n in self.DC]

        cpu_delta = np.zeros(self.node_num)
        mem_delta = np.zeros(self.node_num)

        # ✅ 详细失败原因：DC 不足
        if len(path_dcs) < len(req_vnfs):
            reason = {
                'type': 'not_enough_dcs',
                'required': len(req_vnfs),
                'available': len(path_dcs),
                'path': path_nodes
            }
            logger.debug(f"Req {request.get('id', '?')}: {reason['type']}")
            return False, existing_hvt, {}, reason

        current_dc_idx = 0
        for v_idx, v_type in enumerate(req_vnfs):
            deployed = False

            # 1. 尝试复用
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

            # ✅ 详细失败原因：资源不足
            if not deployed:
                reason = {
                    'type': 'resource_shortage',
                    'vnf_idx': v_idx,
                    'vnf_type': v_type,
                    'cpu_required': request['cpu_origin'][v_idx],
                    'mem_required': request['memory_origin'][v_idx],
                    'checked_nodes': [path_dcs[i] for i in range(current_dc_idx, len(path_dcs))]
                }
                logger.debug(f"Req {request.get('id', '?')}: VNF {v_idx} resource_shortage")
                return False, existing_hvt, {}, reason

        resource_delta = {'cpu': cpu_delta, 'mem': mem_delta}
        return True, hvt, placement, resource_delta

    def _evaluate_otv(self, request: Dict, tree_links: np.ndarray, hvt: np.ndarray) -> float:
        """计算 OTV 成本"""
        node_cost = np.sum(hvt)
        link_cost = np.sum(tree_links)
        return self.config.otv_link_weight * (link_cost / self.config.otv_norm_link) + \
            self.config.otv_node_weight * (node_cost / self.config.otv_norm_node)

    # ========== 资源验证（含带宽检查）==========
    def _validate_resource_deduction(self, state: Dict, resource_delta: Dict,
                                     request: Dict, links_used: Optional[List[int]] = None) -> bool:
        """
        ✅ Document 4 的完整资源验证（含带宽）
        """
        cpu_d = resource_delta.get('cpu', np.zeros(self.node_num))
        mem_d = resource_delta.get('mem', np.zeros(self.node_num))

        # 形状验证
        if cpu_d.shape != (self.node_num,) or mem_d.shape != (self.node_num,):
            logger.error(f"Shape mismatch: cpu{cpu_d.shape}, mem{mem_d.shape}")
            return False

        # 非负性验证
        if np.any(cpu_d < -1e-10) or np.any(mem_d < -1e-10):
            logger.error("Negative delta detected")
            return False

        # 容量验证
        if np.any(state['cpu'] - cpu_d < -1e-8):
            logger.error(f"CPU violation at nodes: {np.where(state['cpu'] - cpu_d < 0)[0]}")
            return False
        if np.any(state['mem'] - mem_d < -1e-8):
            logger.error(f"MEM violation at nodes: {np.where(state['mem'] - mem_d < 0)[0]}")
            return False

        # ✅ 带宽验证
        if links_used:
            bw_required = request.get('bw_origin', 0.0)
            for lid in links_used:
                idx = lid - 1
                if idx < len(state['bw']):
                    if state['bw'][idx] < bw_required - 1e-8:
                        logger.error(f"BW insufficient on link {lid}: {state['bw'][idx]:.4f} < {bw_required:.4f}")
                        return False

        return True

    # ========== 应用路径到树（带回滚）==========
    def _apply_path_to_tree(self, tree_struct, info, request, state,
                            real_deploy=False, resource_delta=None):
        """
        ✅ Document 5 的严格断言 + Document 4 的带宽检查
        """
        nodes = info['nodes']
        links_used = []

        # 更新链路
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            if (u, v) in self.link_map:
                lid = self.link_map[(u, v)]
                idx = lid - 1
                links_used.append(lid)
                if idx < len(tree_struct['tree']):
                    if tree_struct['tree'][idx] == 0:
                        tree_struct['tree'][idx] = 1
                        if real_deploy and idx < len(state['bw']):
                            state['bw'][idx] = max(0.0, state['bw'][idx] - request['bw_origin'])

        tree_struct['nodes'].update(nodes)
        tree_struct['paths_map'][nodes[-1]] = nodes

        if 'hvt' in info:
            tree_struct['hvt'] = np.maximum(tree_struct['hvt'], info['hvt'])

        # 资源扣减（验证 + 应用）
        if real_deploy and resource_delta:
            ok = self._validate_resource_deduction(state, resource_delta, request, links_used)
            if not ok:
                raise ValueError("Resource deduction validation failed")

            cpu_d = resource_delta['cpu']
            mem_d = resource_delta['mem']
            state['cpu'] = np.maximum(state['cpu'] - cpu_d, 0.0)
            state['mem'] = np.maximum(state['mem'] - mem_d, 0.0)

    def _apply_path_to_tree_with_rollback(self, tree_struct, info, request, state,
                                          real_deploy=False, resource_delta=None) -> bool:
        """
        ✅ Document 4 的 Rollback 机制
        """
        original_state = {
            'cpu': state['cpu'].copy(),
            'mem': state['mem'].copy(),
            'bw': state['bw'].copy()
        }
        original_tree = {
            'tree': tree_struct['tree'].copy(),
            'hvt': tree_struct['hvt'].copy(),
            'paths_map': copy.deepcopy(tree_struct['paths_map']),
            'nodes': set(tree_struct['nodes'])
        }

        try:
            self._apply_path_to_tree(tree_struct, info, request, state,
                                     real_deploy, resource_delta)
            return True
        except Exception as e:
            # 回滚
            state.update(original_state)
            tree_struct['tree'] = original_tree['tree']
            tree_struct['hvt'] = original_tree['hvt']
            tree_struct['paths_map'] = original_tree['paths_map']
            tree_struct['nodes'] = original_tree['nodes']
            logger.error(f"Rollback: {e}")
            return False

    # ========== 接口方法（兼容性）==========
    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        """返回 8 个值（兼容旧接口）"""
        state = self._normalize_state(state)
        src = request['source']
        dst = request['dest'][d_idx]
        nodes, dist, links = self._get_path_info(src, dst, k)

        if not nodes:
            return 0.0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dst, 0.0, {}

        score = self._calc_path_eval(nodes, links, state, src, dst)
        temp_state = copy.deepcopy(state)
        feasible, new_hvt, placement, _ = self._try_deploy_vnf(
            request, nodes, temp_state, np.zeros((self.node_num, self.type_num)))

        tree_vec = np.zeros(self.link_num)
        if feasible:
            for lid in links:
                if lid - 1 < len(tree_vec):
                    tree_vec[lid - 1] = 1

        cost = self._evaluate_otv(request, tree_vec, new_hvt) if feasible else 0.0
        return score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement

    def _calc_atnp(self, current_tree: Dict, conn_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
        """Stage 2: 连接新目标到树"""
        state = self._normalize_state(state)
        request = state.get('request')
        if request is None:
            return {'feasible': False}, 0.0, (0, 0), 0.0

        best_eval = -1.0
        best_res = None
        best_action = (0, 0)

        for i_idx, conn_node in enumerate(conn_path):
            for k in range(1, self.k_path + 1):
                # ✅ 修复：删除重复的这一行
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
                    existing_hvt = current_tree.get('hvt', np.zeros((self.node_num, self.type_num)))

                    feasible, new_hvt, placement, res_delta = self._try_deploy_vnf(
                        request, full_nodes, temp_state, existing_hvt)

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
            cost = self._evaluate_otv(request, best_res['tree'], best_res['hvt'])
            return best_res, best_eval, best_action, cost
        else:
            return {'feasible': False}, 0.0, (0, 0), 0.0
    # ========== 资源预检查 ==========
    def _check_resource_feasibility(self, request: Dict, state: Dict) -> bool:
        """快速全局资源检查"""
        total_cpu_req = sum(request.get('cpu_origin', []))
        total_mem_req = sum(request.get('memory_origin', []))
        total_bw_req = request.get('bw_origin', 0.0) * len(request.get('dest', []))

        available_cpu = np.sum(state['cpu'])
        available_mem = np.sum(state['mem'])
        available_bw = np.sum(state['bw'])

        if total_cpu_req > available_cpu or total_mem_req > available_mem:
            return False
        if total_bw_req > available_bw:
            logger.debug("Total BW requirement high (may still succeed due to sharing)")
        return True

    def _get_adaptive_lookahead_depth(self, num_remaining: int) -> int:
        """动态调整 lookahead 深度"""
        if num_remaining <= 2:
            return min(num_remaining, self.config.lookahead_depth)
        elif num_remaining <= 5:
            return min(2, self.config.lookahead_depth)
        else:
            return 1

    # ========== 树构建主流程 ==========
    def _construct_tree(self, request: Dict, network_state: Dict,
                        forced_first_dest_idx: Optional[int] = None) -> Tuple[Optional[Dict], List]:
        """
        核心树构建逻辑（含候选集 + Lookahead）
        ✅ Document 5 的清晰结构 + Document 4 的 Rollback
        """
        start_time = time.time()
        iteration_count = 0

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

        MAX_CANDIDATES = int(self.config.max_candidates)
        MAX_ITER = int(self.config.max_iterations)
        MAX_TIME = float(self.config.max_time_seconds)

        while len(current_tree['added_dest_indices']) < len(dest_indices):
            iteration_count += 1

            # ✅ 迭代/超时限制
            if iteration_count > MAX_ITER:
                logger.warning(f"Req {request['id']}: Max iterations ({MAX_ITER}) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            if time.time() - start_time > MAX_TIME:
                logger.warning(f"Req {request['id']}: Timeout ({MAX_TIME}s) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            unadded = [d for d in dest_indices if d not in current_tree['added_dest_indices']]
            candidates = []

            # A. 候选集生成
            if not ordered_paths:
                # Stage 1: Source -> Dest
                targets = [forced_first_dest_idx] if forced_first_dest_idx is not None else unadded
                for d_idx in targets:
                    if d_idx not in unadded:
                        continue
                    if len(candidates) >= MAX_CANDIDATES:
                        break

                    for k in range(1, self.k_path + 1):
                        score, nodes, t_vec, h_vec, feas, _, cost, pl = self._calc_eval(
                            request, d_idx, k, current_sim_state)

                        if feas:
                            # 获取 res_delta
                            _, _, _, res_delta = self._try_deploy_vnf(
                                request, nodes, current_sim_state,
                                np.zeros((self.node_num, self.type_num)))

                            info = {
                                'nodes': nodes, 'k': k, 'score': score, 'p_idx': 0,
                                'res_delta': res_delta, 'hvt': h_vec, 'tree_vec': t_vec,
                                'placement': pl, 'd_idx': d_idx
                            }
                            candidates.append(info)
            else:
                # Stage 2: Tree -> New Dest
                for p_idx, path in enumerate(ordered_paths):
                    for d_idx in unadded:
                        if len(candidates) >= MAX_CANDIDATES:
                            break

                        res, score, action_in_path, _ = self._calc_atnp(
                            current_tree, path, d_idx, current_sim_state, current_tree['nodes'])

                        if res and res.get('feasible'):
                            k = action_in_path[1] + 1
                            info = {
                                'nodes': res['new_path_full'], 'k': k, 'score': score,
                                'p_idx': p_idx, 'conn_idx_in_path': action_in_path[0],
                                'res_delta': res['res_delta'], 'hvt': res['hvt'],
                                'tree_vec': res['tree'], 'placement': res['placement'],
                                'd_idx': d_idx
                            }
                            candidates.append(info)

            if not candidates:
                return None, unadded

            # B. 候选集排序
            candidates.sort(key=lambda x: x['score'], reverse=True)
            candidate_set = candidates[:int(self.config.candidate_set_size)]

            # C. Lookahead 选择
            best_global_otv = float('inf')
            selected_info = None
            current_lookahead_depth = self._get_adaptive_lookahead_depth(len(unadded) - 1)

            for info in candidate_set:
                d_idx = info['d_idx']

                # 模拟树副本
                temp_tree_sim = copy.deepcopy(current_tree)
                temp_state_sim = copy.deepcopy(current_sim_state)

                # ✅ 使用 Rollback 机制
                applied = self._apply_path_to_tree_with_rollback(
                    temp_tree_sim, info, request, temp_state_sim,
                    real_deploy=True, resource_delta=info.get('res_delta'))

                if not applied:
                    continue

                # 贪婪 Lookahead 扩展
                remaining_after = [d for d in unadded if d != d_idx]
                subsequent_count = 0

                while subsequent_count < current_lookahead_depth and remaining_after:
                    next_candidates = []
                    current_sim_paths = list(temp_tree_sim['paths_map'].values()) \
                                       if temp_tree_sim['paths_map'] else [[request['source']]]

                    for next_d_idx in remaining_after:
                        for path in current_sim_paths:
                            res, score, _, _ = self._calc_atnp(
                                temp_tree_sim, path, next_d_idx,
                                temp_state_sim, temp_tree_sim['nodes'])

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

                    applied2 = self._apply_path_to_tree_with_rollback(
                        temp_tree_sim, temp_info_next, request, temp_state_sim,
                        real_deploy=True, resource_delta=best_res['res_delta'])

                    if not applied2:
                        break

                    remaining_after.remove(best_next_d)
                    subsequent_count += 1

                # 计算 OTV
                otv = self._evaluate_otv(request, temp_tree_sim['tree'], temp_tree_sim['hvt'])
                if otv < best_global_otv:
                    best_global_otv = otv
                    selected_info = info

            # D. 执行最优选择
            if selected_info:
                d_idx = selected_info['d_idx']

                applied = self._apply_path_to_tree_with_rollback(
                    current_tree, selected_info, request, current_sim_state,
                    real_deploy=True, resource_delta=selected_info.get('res_delta'))

                if not applied:
                    self._record_failure(request.get('id', '?'),
                                       {'type': 'apply_failed', 'info': 'final apply failed'})
                    return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

                current_tree['added_dest_indices'].append(d_idx)
                ordered_paths.append(selected_info['nodes'])

                # 记录轨迹
                p_idx = selected_info['p_idx']
                k_idx = selected_info['k'] - 1
                placement = selected_info.get('placement', {})
                action_tuple = (p_idx, k_idx, placement)
                cost = self._evaluate_otv(request, current_tree['tree'], current_tree['hvt'])
                current_tree['traj'].append((d_idx, action_tuple, cost))
            else:
                return None, unadded

        return current_tree, current_tree['traj']

    # ========== 增强 Recall 策略 ==========
    def _estimate_destination_resource(self, request: Dict, d_idx: int,
                                       network_state: Dict) -> float:
        """估算目标节点的资源需求（启发式）"""
        cpu = sum(request.get('cpu_origin', []))
        mem = sum(request.get('memory_origin', []))
        bw = request.get('bw_origin', 0.0)
        return float(cpu + mem + bw * 10.0)

    def _enhanced_recall_strategy(self, request: Dict, network_state: Dict,
                                  failed_unadded: List[int]) -> Tuple[Optional[Dict], List]:
        """
        ✅ Document 4 的增强 Recall 策略（按资源需求排序）
        """
        if not failed_unadded:
            return None, []

        logger.info(f"Recall for req {request.get('id','?')} with {len(failed_unadded)} failed dests")

        # 按资源需求排序（从小到大）
        dest_resources = [(d_idx, self._estimate_destination_resource(request, d_idx, network_state))
                         for d_idx in failed_unadded]
        dest_resources.sort(key=lambda x: x[1])

        # 优先尝试资源需求小的
        for d_idx, _ in dest_resources:
            recall_tree, recall_traj = self._construct_tree(
                request, network_state, forced_first_dest_idx=d_idx)

            if recall_tree is not None:
                logger.info(f"Recall successful using dest {d_idx} first")
                return recall_tree, recall_traj

        return None, []

    # ========== 性能监控 ==========
    def _record_failure(self, request_id, reason_dict):
        """记录失败原因"""
        reason_type = reason_dict.get('type', 'unknown')
        self.metrics['failure_reasons'][reason_type] = \
            self.metrics['failure_reasons'].get(reason_type, 0) + 1

    def clear_cache(self):
        """清空路径评分缓存"""
        self._path_eval_cache.clear()
        logger.info("Path evaluation cache cleared")

    def export_metrics(self, path: Optional[Path] = None):
        """导出性能指标到 CSV"""
        import csv
        if path is None:
            path = Path('expert_metrics.csv')

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Requests', self.metrics['total_requests']])
            writer.writerow(['Accepted', self.metrics['accepted']])
            writer.writerow(['Rejected', self.metrics['rejected']])

            accept_rate = self.metrics['accepted'] / max(1, self.metrics['total_requests'])
            writer.writerow(['Accept Rate', f"{accept_rate:.2%}"])

            writer.writerow([])
            writer.writerow(['Failure Reason', 'Count'])
            for reason, count in self.metrics.get('failure_reasons', {}).items():
                writer.writerow([reason, count])

            if self.metrics.get('processing_times'):
                writer.writerow([])
                writer.writerow(['Avg Processing Time (s)',
                               np.mean(self.metrics['processing_times'])])

        logger.info(f"Metrics exported to {path}")

    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'total_requests': self.metrics['total_requests'],
            'acceptance_rate': self.metrics['accepted'] / max(1, self.metrics['total_requests']),
            'cache_hit_rate': self.metrics['cache_hits'] /
                             max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'failure_reasons': self.metrics.get('failure_reasons', {}),
        }

        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report.update({
                'avg_processing_time': float(np.mean(times)),
                'max_processing_time': float(max(times)),
                'min_processing_time': float(min(times)),
            })

        return report

    def get_detailed_performance_report(self) -> Dict:
        """
        ✅ Document 4 的详细性能报告
        """
        report = self.get_performance_report()

        # 缓存效率
        cache_eff = {
            'cache_size': len(self._path_eval_cache),
            'cache_max_size': self.MAX_CACHE_SIZE,
            'cache_utilization': len(self._path_eval_cache) / max(1, self.MAX_CACHE_SIZE)
        }
        report['cache_efficiency'] = cache_eff

        # 最近性能趋势
        if self.metrics.get('processing_times'):
            times = self.metrics['processing_times']
            report['recent_performance'] = {
                'last_10_avg': float(np.mean(times[-10:])) if len(times) >= 10 else 0.0,
                'trend': 'improving' if len(times) > 1 and times[-1] < times[0] else 'stable'
            }

        return report

    # ========== 主入口 ==========
    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> Tuple[Optional[Dict], List]:
        """
        专家算法主入口

        Args:
            request: 请求字典（包含 id, source, dest, vnf, cpu_origin, memory_origin, bw_origin）
            network_state: 网络状态（包含 cpu, mem, bw, hvt 等）

        Returns:
            (tree_struct, traj) 成功时返回树结构和轨迹
            (None, []) 失败时返回 None
        """
        start_time = time.time()
        self.metrics['total_requests'] += 1

        try:
            network_state = self._normalize_state(network_state)
            network_state['request'] = request

            # 1. 快速资源预检查
            if not self._check_resource_feasibility(request, network_state):
                logger.warning(f"Req {request.get('id','?')} skipped: insufficient resources")
                self.metrics['rejected'] += 1
                self._record_failure(request.get('id','?'), {'type': 'global_resource_shortage'})
                return None, []

            # 2. 正常树构建
            res_tree, res_traj = self._construct_tree(request, network_state)

            proc_time = time.time() - start_time
            self.metrics['processing_times'].append(proc_time)

            if res_tree is not None:
                self.metrics['accepted'] += 1
                return res_tree, res_traj

            # 3. 触发增强 Recall
            failed_dests = res_traj
            if failed_dests:
                recall_tree, recall_traj = self._enhanced_recall_strategy(
                    request, network_state, failed_dests)

                if recall_tree is not None:
                    self.metrics['accepted'] += 1
                    return recall_tree, recall_traj

            # 4. 最终失败
            self.metrics['rejected'] += 1
            self._record_failure(request.get('id','?'), {'type': 'construct_tree_failed'})
            return None, []

        except Exception as e:
            logger.exception(f"Unexpected error in req {request.get('id','?')}: {e}")
            self.metrics['errors'] += 1
            self.metrics['rejected'] += 1
            return None, []


if __name__ == "__main__":
    logger.info("Expert MSFCE module loaded (Fusion Version)")
    logger.info("Features: SolverConfig + Rollback + Enhanced Recall + Full Validation")