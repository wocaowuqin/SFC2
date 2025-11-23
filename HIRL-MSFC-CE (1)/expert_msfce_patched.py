#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expert_msfce.py (patched)
# Consolidated patch implementing suggested fixes:
# - _get_path_info handles src==dst and better exception handling
# - _try_deploy_vnf returns detailed failure reasons
# - _apply_path_to_tree validates resource deduction before applying
# - LRU cache for path evaluation
# - iteration / timeout limits for lookahead
# - metrics collection and export
# - SolverConfig dataclass for centralized configuration
# - enhanced recall strategy (heuristic)
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

# --- logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Expert] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------- Configuration --------------------
@dataclass
class SolverConfig:
    alpha: float = 0.3
    beta: float = 0.3
    gamma: float = 0.4
    candidate_set_size: int = 4
    lookahead_depth: int = 3
    k_path: int = 5
    max_cache_size: int = 5000
    max_iterations: int = 200
    max_time_seconds: float = 20.0
    max_candidates: int = 12  # candidate_set_size * 3
    otv_link_weight: float = 0.2
    otv_node_weight: float = 0.8
    otv_norm_link: float = 90.0
    otv_norm_node: float = 8.0

    def __post_init__(self):
        if not (0 <= self.alpha <= 1 and 0 <= self.beta <= 1 and 0 <= self.gamma <= 1):
            raise ValueError("Alpha, beta, gamma must be between 0 and 1")
        if abs(self.alpha + self.beta + self.gamma - 1.0) > 1e-6:
            logger.warning("Score weights (alpha+beta+gamma) do not sum to 1.0")


# -------------------- Solver --------------------
class MSFCE_Solver:
    def __init__(self, path_db_file: Path, topology_matrix: np.ndarray,
                 dc_nodes: List[int], capacities: Dict, config: Optional[SolverConfig] = None):
        self.config = config or SolverConfig()

        # load path DB
        if not Path(path_db_file).exists():
            logger.error(f"Path DB file not found: {path_db_file}")
            raise FileNotFoundError(f"Path DB file missing: {path_db_file}")

        try:
            mat = sio.loadmat(path_db_file)
            if 'Paths' not in mat:
                raise KeyError("Key 'Paths' not in .mat file")
            self.path_db = mat['Paths']
            logger.info(f"Successfully loaded Path DB from {path_db_file}")
        except Exception as e:
            logger.critical(f"Failed to load Path DB structure: {e}")
            raise RuntimeError("Path DB load failed") from e

        self.node_num = int(topology_matrix.shape[0])
        self.link_num, self.link_map = self._create_link_map(topology_matrix)

        self.type_num = 8
        self.DC = set(dc_nodes)
        self.dc_num = len(dc_nodes)

        self.cap_cpu = float(capacities['cpu'])
        self.cap_mem = float(capacities['memory'])
        self.cap_bw = float(capacities['bandwidth'])

        self.k_path = int(self.config.k_path)

        # LRU cache for path eval (store static terms only)
        self._path_eval_cache = OrderedDict()
        self.MAX_CACHE_SIZE = int(self.config.max_cache_size)

        # metrics
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

    # -------------------- utilities --------------------
    def _create_link_map(self, topo: np.ndarray) -> Tuple[int, Dict]:
        link_map = {}
        lid = 1
        for i in range(topo.shape[0]):
            for j in range(i + 1, topo.shape[0]):
                if not np.isinf(topo[i, j]) and topo[i, j] > 0:
                    # store 1-based node ids as keys
                    link_map[(i + 1, j + 1)] = lid
                    link_map[(j + 1, i + 1)] = lid
                    lid += 1
        return lid - 1, link_map

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

    # -------------------- path lookup --------------------
    def _get_path_info(self, src: int, dst: int, k: int):
        """
        Get path info (1-based node IDs). Handle src==dst and missing indices gracefully.
        Returns (nodes, dist, links)
        """
        if self.path_db is None:
            return [], 0, []

        # treat trivial self-path
        if src == dst:
            return [src], 0, []

        try:
            cell = self.path_db[src-1, dst-1]
            dist = int(cell['pathsdistance'][k-1][0])
            nodes = cell['paths'][k-1, :dist+1].astype(int).tolist()
            links = []
            for i in range(len(nodes)-1):
                u, v = nodes[i], nodes[i+1]
                if (u, v) in self.link_map:
                    links.append(self.link_map[(u, v)])
            return nodes, dist, links
        except (IndexError, KeyError) as e:
            logger.warning(f"Path lookup failed for {src}->{dst} k={k}: {e}")
            return [], 0, []
        except Exception as e:
            # unexpected errors: log and return empty
            logger.debug(f"Path lookup unexpected error for {src}->{dst} k={k}: {e}")
            return [], 0, []

    def _get_max_hops(self, src: int, dst: int) -> int:
        try:
            cell = self.path_db[src-1, dst-1]
            return int(cell['pathsdistance'][self.k_path-1][0])
        except Exception:
            return 10

    # -------------------- path evaluation (with caching) --------------------
    def _calc_path_eval(self, nodes: List[int], links: List[int],
                        state: Dict, src_node: int, dst_node: int) -> float:
        if not nodes:
            return 0.0

        cache_key = (src_node, dst_node, len(nodes))
        if cache_key in self._path_eval_cache:
            # LRU update
            term1, term2 = self._path_eval_cache.pop(cache_key)
            self._path_eval_cache[cache_key] = (term1, term2)
            self.metrics['cache_hits'] = self.metrics.get('cache_hits', 0) + 1
        else:
            self.metrics['cache_misses'] = self.metrics.get('cache_misses', 0) + 1
            # compute static part: hops and DC count
            max_hops = self._get_max_hops(src_node, dst_node)
            current_hops = len(nodes) - 1
            term1 = 1.0 - (current_hops / max(1, max_hops))
            dc_count = sum(1 for n in nodes if n in self.DC)
            term2 = dc_count / max(1, self.dc_num)
            # store in LRU cache
            self._path_eval_cache[cache_key] = (term1, term2)
            if len(self._path_eval_cache) > self.MAX_CACHE_SIZE:
                self._path_eval_cache.popitem(last=False)

        # dynamic part: remaining resources
        term1, term2 = self._path_eval_cache[cache_key]
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

        return float(self.config.alpha * term1 + self.config.beta * term2 + self.config.gamma * term3)

    # -------------------- try deploy VNF (with detailed reasons) --------------------
    def _try_deploy_vnf(self, request: Dict, path_nodes: List[int],
                        state: Dict, existing_hvt: np.ndarray) -> Tuple[bool, np.ndarray, Dict, Dict]:
        """
        Attempt to deploy VNFs along path_nodes. Returns (feasible, new_hvt, placement, resource_delta_or_reason)
        resource_delta_or_reason is resource_delta dict on success, or a reason dict on failure.
        """
        req_vnfs = request.get('vnf', [])
        hvt = existing_hvt.copy()
        placement = {}
        path_dcs = [n for n in path_nodes if n in self.DC]

        cpu_delta = np.zeros(self.node_num)
        mem_delta = np.zeros(self.node_num)

        # Not enough DCs on path
        if len(path_dcs) < len(req_vnfs):
            reason = {
                'type': 'not_enough_dcs',
                'required': len(req_vnfs),
                'available': len(path_dcs),
                'path': path_nodes
            }
            logger.debug(f"Req {request.get('id', '?')}: {reason['type']} - need {reason['required']}, have {reason['available']}")
            return False, existing_hvt, {}, reason

        current_dc_idx = 0
        for v_idx, v_type in enumerate(req_vnfs):
            deployed = False
            # 1) try reuse existing VNF instance
            for node in path_dcs:
                node_idx = node - 1
                if node_idx < hvt.shape[0] and (v_type - 1) < hvt.shape[1]:
                    if hvt[node_idx, v_type - 1] > 0:
                        placement[v_idx] = node
                        deployed = True
                        break
            if deployed:
                continue

            # 2) deploy new instance
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

            if not deployed:
                reason = {
                    'type': 'resource_shortage',
                    'vnf_idx': v_idx,
                    'vnf_type': v_type,
                    'cpu_required': request['cpu_origin'][v_idx],
                    'mem_required': request['memory_origin'][v_idx],
                    'checked_nodes': [path_dcs[i] for i in range(current_dc_idx, len(path_dcs))],
                    'cpu_delta': cpu_delta.tolist(),
                    'mem_delta': mem_delta.tolist()
                }
                logger.debug(f"Req {request.get('id','?')}: VNF {v_idx} (type {v_type}) deployment failed - {reason['type']}")
                return False, existing_hvt, {}, reason

        resource_delta = {'cpu': cpu_delta, 'mem': mem_delta}
        return True, hvt, placement, resource_delta

    # -------------------- OTV --------------------
    def _evaluate_otv(self, request: Dict, tree_links: np.ndarray, hvt: np.ndarray) -> float:
        node_cost = np.sum(hvt)
        link_cost = np.sum(tree_links)
        return self.config.otv_link_weight * (link_cost / self.config.otv_norm_link) + \
               self.config.otv_node_weight * (node_cost / self.config.otv_norm_node)

    # -------------------- resource validation --------------------
    def _validate_resource_deduction(self, state: Dict, resource_delta: Dict, request: Dict) -> bool:
        cpu_d = resource_delta.get('cpu', np.zeros(self.node_num))
        mem_d = resource_delta.get('mem', np.zeros(self.node_num))

        # shape checks
        if cpu_d.shape != (self.node_num,) or mem_d.shape != (self.node_num,):
            logger.error(f"Resource delta shape invalid: cpu{cpu_d.shape}, mem{mem_d.shape}")
            return False

        # non-negativity
        if np.any(cpu_d < -1e-10) or np.any(mem_d < -1e-10):
            logger.error("Negative resource delta detected")
            return False

        # capacity checks (allow tiny epsilon)
        cpu_violation = state['cpu'] - cpu_d < -1e-8
        mem_violation = state['mem'] - mem_d < -1e-8

        if np.any(cpu_violation):
            violating_nodes = np.where(cpu_violation)[0]
            logger.error(f"CPU violation at nodes: {violating_nodes}")
            return False

        if np.any(mem_violation):
            violating_nodes = np.where(mem_violation)[0]
            logger.error(f"Memory violation at nodes: {violating_nodes}")
            return False

        # bandwidth verification placeholder (if needed)
        return True

    # -------------------- apply path to tree with safety --------------------
    def _apply_path_to_tree(self, tree_struct, info, request, state, real_deploy=False, resource_delta=None):
        nodes = info['nodes']

        # update links
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

        if 'hvt' in info:
            tree_struct['hvt'] = np.maximum(tree_struct['hvt'], info['hvt'])

        if real_deploy and resource_delta:
            cpu_d = resource_delta.get('cpu', np.zeros(self.node_num))
            mem_d = resource_delta.get('mem', np.zeros(self.node_num))

            # validate before applying
            ok = self._validate_resource_deduction(state, resource_delta, request)
            if not ok:
                raise ValueError("Resource deduction validation failed; aborting apply_path_to_tree")

            state['cpu'] = np.maximum(state['cpu'] - cpu_d, 0.0)
            state['mem'] = np.maximum(state['mem'] - mem_d, 0.0)

    # -------------------- candidate / lookahead / construct tree --------------------
    def _check_resource_feasibility(self, request: Dict, state: Dict) -> bool:
        total_cpu_req = sum(request.get('cpu_origin', []))
        total_mem_req = sum(request.get('memory_origin', []))
        total_bw_req = request.get('bw_origin', 0.0) * len(request.get('dest', []))

        available_cpu = np.sum(state['cpu'])
        available_mem = np.sum(state['mem'])
        available_bw = np.sum(state['bw'])

        if total_cpu_req > available_cpu or total_mem_req > available_mem:
            return False
        if total_bw_req > available_bw:
            # allow, but warn
            logger.debug("Total bandwidth required > available_bw; may still be feasible due to sharing")
        return True

    def _get_adaptive_lookahead_depth(self, num_remaining: int) -> int:
        if num_remaining <= 2:
            return min(num_remaining, self.config.lookahead_depth)
        elif num_remaining <= 5:
            return min(2, self.config.lookahead_depth)
        else:
            return 1

    def _calc_eval(self, request: Dict, d_idx: int, k: int, state: Dict):
        state = self._normalize_state(state)
        src = request['source']
        dst = request['dest'][d_idx]
        nodes, dist, links = self._get_path_info(src, dst, k)
        if not nodes:
            return 0.0, [], np.zeros(self.link_num), np.zeros((self.node_num, self.type_num)), False, dst, 0.0, {}, {}

        score = self._calc_path_eval(nodes, links, state, src, dst)
        temp_state = copy.deepcopy(state)
        feasible, new_hvt, placement, res_delta = self._try_deploy_vnf(request, nodes, temp_state, np.zeros((self.node_num, self.type_num)))

        tree_vec = np.zeros(self.link_num)
        if feasible:
            for lid in links:
                if lid - 1 < len(tree_vec):
                    tree_vec[lid - 1] = 1
        cost = self._evaluate_otv(request, tree_vec, new_hvt) if feasible else 0.0
        return score, nodes, tree_vec, new_hvt, feasible, dst, cost, placement, res_delta

    def _calc_atnp(self, current_tree: Dict, conn_path: List[int], d_idx: int,
                   state: Dict, nodes_on_tree: Set[int]):
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
                    existing_hvt = current_tree.get('hvt', current_tree.get('hvt_vec', np.zeros((self.node_num, self.type_num))))

                    feasible, new_hvt, placement, res_delta = self._try_deploy_vnf(request, full_nodes, temp_state, existing_hvt)
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

    def _construct_tree(self, request: Dict, network_state: Dict,
                        forced_first_dest_idx: Optional[int] = None) -> Tuple[Optional[Dict], List]:
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
            # safety guards
            if iteration_count > MAX_ITER:
                logger.warning(f"Req {request['id']}: Max iterations ({MAX_ITER}) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]
            if time.time() - start_time > MAX_TIME:
                logger.warning(f"Req {request['id']}: Timeout ({MAX_TIME}s) reached")
                return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

            unadded = [d for d in dest_indices if d not in current_tree['added_dest_indices']]
            candidates = []

            # A. generate candidates
            if not ordered_paths:
                targets = [forced_first_dest_idx] if forced_first_dest_idx is not None else unadded
                for d_idx in targets:
                    if d_idx not in unadded:
                        continue
                    if len(candidates) >= MAX_CANDIDATES:
                        break
                    for k in range(1, self.k_path + 1):
                        score, nodes, t_vec, h_vec, feas, _, cost, pl, res_delta = self._calc_eval(request, d_idx, k, current_sim_state)
                        if feas:
                            info = {'nodes': nodes, 'links': [], 'k': k, 'score': score, 'p_idx': 0,
                                    'res_delta': res_delta, 'hvt': h_vec, 'tree_vec': t_vec, 'placement': pl, 'd_idx': d_idx}
                            candidates.append(info)
            else:
                for p_idx, path in enumerate(ordered_paths):
                    for d_idx in unadded:
                        if len(candidates) >= MAX_CANDIDATES:
                            break
                        res, score, action_in_path, _ = self._calc_atnp(current_tree, path, d_idx, current_sim_state, current_tree['nodes'])
                        if res and res.get('feasible'):
                            k = action_in_path[1] + 1
                            info = {'nodes': res['new_path_full'], 'k': k, 'score': score, 'p_idx': p_idx,
                                    'conn_idx_in_path': action_in_path[0],
                                    'res_delta': res['res_delta'], 'hvt': res['hvt'], 'tree_vec': res['tree'],
                                    'placement': res['placement'], 'd_idx': d_idx}
                            candidates.append(info)

            if not candidates:
                return None, unadded

            # B. filter
            candidates.sort(key=lambda x: x['score'], reverse=True)
            candidate_set = candidates[:int(self.config.candidate_set_size)]

            # C. lookahead
            best_global_otv = float('inf')
            selected_info = None
            current_lookahead_depth = self._get_adaptive_lookahead_depth(len(unadded) - 1)

            for info in candidate_set:
                d_idx = info['d_idx']

                temp_tree_sim = copy.deepcopy(current_tree)
                temp_state_sim = copy.deepcopy(current_sim_state)

                # if res_delta is a failure reason dict, skip
                res_delta = info.get('res_delta')
                if isinstance(res_delta, dict) and res_delta.get('type') is not None and not res_delta.get('type') == 'ok':
                    # this candidate shows failure on try_deploy; skip for lookahead
                    continue

                try:
                    self._apply_path_to_tree(temp_tree_sim, info, request, temp_state_sim,
                                            real_deploy=True, resource_delta=res_delta)
                except Exception as e:
                    logger.debug(f"Sim apply failed during lookahead: {e}")
                    continue

                remaining_after = [d for d in unadded if d != d_idx]
                subsequent_count = 0

                while subsequent_count < current_lookahead_depth and remaining_after:
                    next_candidates = []
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
                        'tree_vec': best_res['tree'],
                    }
                    try:
                        self._apply_path_to_tree(temp_tree_sim, temp_info_next, request, temp_state_sim,
                                                real_deploy=True, resource_delta=best_res['res_delta'])
                    except Exception as e:
                        logger.debug(f"Sim apply failed when expanding lookahead: {e}")
                        break

                    remaining_after.remove(best_next_d)
                    subsequent_count += 1

                otv = self._evaluate_otv(request, temp_tree_sim['tree'], temp_tree_sim['hvt'])
                if otv < best_global_otv:
                    best_global_otv = otv
                    selected_info = info

            # D. apply selected_info
            if selected_info:
                d_idx = selected_info['d_idx']
                try:
                    self._apply_path_to_tree(current_tree, selected_info, request, current_sim_state,
                                            real_deploy=True, resource_delta=selected_info['res_delta'])
                except Exception as e:
                    # record failure
                    reason = {'type': 'apply_failed', 'error': str(e)}
                    self._record_failure(request.get('id', '?'), reason)
                    return None, [d for d in dest_indices if d not in current_tree['added_dest_indices']]

                current_tree['added_dest_indices'].append(d_idx)
                ordered_paths.append(selected_info['nodes'])

                p_idx = selected_info['p_idx']
                k_idx = selected_info['k'] - 1
                placement = selected_info.get('placement', {})
                action_tuple = (p_idx, k_idx, placement)
                cost = self._evaluate_otv(request, current_tree['tree'], current_tree['hvt'])
                current_tree['traj'].append((d_idx, action_tuple, cost))
            else:
                return None, unadded

        return current_tree, current_tree['traj']

    # -------------------- recall strategy --------------------
    def _estimate_destination_resource(self, request: Dict, d_idx: int, network_state: Dict) -> float:
        # crude estimator: sum of vnf cpu+mem plus bw factor; lower is better
        cpu = sum(request.get('cpu_origin', []))
        mem = sum(request.get('memory_origin', []))
        bw = request.get('bw_origin', 0.0)
        return float(cpu + mem + bw * 10.0)

    def _enhanced_recall_strategy(self, request: Dict, network_state: Dict, failed_unadded: List[int]) -> Tuple[Optional[Dict], List]:
        if not failed_unadded:
            return None, []

        logger.info(f"Starting enhanced recall for request {request.get('id', '?')} with {len(failed_unadded)} failed destinations")

        dest_resources = []
        for d_idx in failed_unadded:
            resource_score = self._estimate_destination_resource(request, d_idx, network_state)
            dest_resources.append((d_idx, resource_score))

        dest_resources.sort(key=lambda x: x[1])  # ascending: try cheapest first

        for d_idx, _ in dest_resources:
            recall_tree, recall_traj = self._construct_tree(request, network_state, forced_first_dest_idx=d_idx)
            if recall_tree is not None:
                logger.info(f"Recall successful with destination {d_idx} as first")
                return recall_tree, recall_traj

        # optional: relaxed constraints could be tried here
        return None, []

    # -------------------- failure recording / metrics --------------------
    def _record_failure(self, request_id, reason_dict):
        reason_type = reason_dict.get('type', 'unknown')
        self.metrics['failure_reasons'][reason_type] = self.metrics['failure_reasons'].get(reason_type, 0) + 1
        self.metrics['rejected'] += 1

    def export_metrics(self, path: Optional[Path] = None):
        import csv
        if path is None:
            path = Path('expert_metrics.csv')
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Requests', self.metrics.get('total_requests', 0)])
            writer.writerow(['Accepted', self.metrics.get('accepted', 0)])
            writer.writerow(['Rejected', self.metrics.get('rejected', 0)])
            writer.writerow(['Accept Rate', f"{self.metrics.get('accepted', 0)/max(1, self.metrics.get('total_requests', 1)):.2%}"])
            writer.writerow([])
            writer.writerow(['Failure Reason', 'Count'])
            for reason, count in self.metrics.get('failure_reasons', {}).items():
                writer.writerow([reason, count])
        logger.info(f"Metrics exported to {path}")

    def get_performance_report(self) -> Dict:
        report = {
            'total_requests': self.metrics.get('total_requests', 0),
            'acceptance_rate': self.metrics.get('accepted', 0) / max(1, self.metrics.get('total_requests', 1)),
            'cache_hit_rate': self.metrics.get('cache_hits', 0) / max(1, self.metrics.get('cache_hits', 0) + self.metrics.get('cache_misses', 0)),
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

    # -------------------- top-level API --------------------
    def solve_request_for_expert(self, request: Dict, network_state: Dict) -> Tuple[Optional[Dict], List]:
        start_time = time.time()
        self.metrics['total_requests'] += 1
        try:
            network_state = self._normalize_state(network_state)
            network_state['request'] = request

            if not self._check_resource_feasibility(request, network_state):
                logger.warning(f"Request {request.get('id', '?')} skipped: Insufficient total resources.")
                self.metrics['rejected'] += 1
                return None, []

            res_tree, res_traj = self._construct_tree(request, network_state)
            processing_time = time.time() - start_time
            self.metrics['processing_times'].append(processing_time)

            if res_tree is not None:
                self.metrics['accepted'] += 1
                return res_tree, res_traj

            # attempt enhanced recall based on failed destinations returned in res_traj
            failed_dests = res_traj
            if failed_dests:
                recall_tree, recall_traj = self._enhanced_recall_strategy(request, network_state, failed_dests)
                if recall_tree is not None:
                    self.metrics['accepted'] += 1
                    return recall_tree, recall_traj

            self.metrics['rejected'] += 1
            return None, []
        except Exception as e:
            logger.exception(f"Unexpected error processing request {request.get('id', '?')}: {e}")
            self.metrics['errors'] += 1
            return None, []


if __name__ == "__main__":
    # simple smoke test creation (requires a real mat file and topology)
    p = Path('/mnt/data/expert_msfce_patched.py')
    logger.info(f"Module created at {p}")
