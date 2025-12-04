# sfc_backup_system/backup_policy.py

import logging
import numpy as np  # 修复 1: 补充导入 numpy
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

from .utils import ensure_list, build_tree_vec, build_hvt_from_placement
from .tree_cache import TreeCache
from .path_finder import PathFinder
from .path_eval import evaluate_path_score
from .vnf_placement import VNFPlacement

logger = logging.getLogger(__name__)


class BackupPolicy:
    """
    完整增强版 BackupPolicy (已修复 Bug)
    - 动态策略选择
    - 资源感知
    - TreeCache 深度整合
    - 多层 Never-Fail 降级
    - 性能统计
    """

    def __init__(self, expert, n: int, L: int, K_vnf: int,
                 dc_nodes: Optional[List[int]] = None):

        self.expert = expert
        self.n = n
        self.L = L
        self.K_vnf = K_vnf

        self.dc_nodes = set(dc_nodes or [])
        self.current_tree: Dict = {"nodes": [], "adjacency": {}}
        self.current_request: Dict = {}

        self.path_finder = PathFinder(expert, n)
        self.tree_cache = TreeCache()

        # 性能统计
        self.stats = {
            "total_calls": 0,
            "strategy_success": {
                "resource_aware": 0,
                "smart_greedy": 0,
                "minimal": 0,
                "never_fail": 0
            },
            "total_failures": 0,
            "avg_score": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }

    # --------------------------
    # Tree / Request 更新方法
    # --------------------------
    def set_current_tree(self, tree: Dict):
        """
        设置当前树结构（由 Expert 生成）
        必须包含: nodes (List/Set), adjacency (Dict)
        """
        if tree is None:
            self.current_tree = {"nodes": [], "adjacency": {}}
            logger.warning("BackupPolicy: set_current_tree received None, using empty tree")
            return

        # ✅ 标准化树结构
        if isinstance(tree, list):  # 兼容只传节点列表的情况
            self.current_tree = {
                "nodes": list(tree),
                "adjacency": {}
            }
        else:
            self.current_tree = {
                "nodes": list(tree.get("nodes", [])),
                "adjacency": dict(tree.get("adjacency", {})),
                "root": tree.get("root")
            }

        # ✅ 缓存失效
        self.tree_cache.invalidate()

        logger.debug(
            f"[BackupPolicy] Set new tree: size={len(self.current_tree['nodes'])}"
        )

    def set_current_request(self, request: Dict):
        """保存当前业务请求"""
        self.current_request = request or {}

    # 别名方法
    def update_tree(self, tree: Dict):
        self.set_current_tree(tree)

    def update_request(self, request: Dict):
        self.set_current_request(request)

    # --------------------------
    # 工具函数
    # --------------------------
    def _is_dc_node(self, nd: int) -> bool:
        """检查节点是否为数据中心"""
        if self.dc_nodes:
            return int(nd) in self.dc_nodes
        return int(nd) % 3 == 0

    # --------------------------
    # 改进 1：增强版 top-k 选择 + TreeCache
    # --------------------------
    def _select_top_k(self, node_list: List[int], dst: int, k: int = 5) -> List[int]:
        """使用 TreeCache 缓存路径代价进行智能选择"""
        if not node_list:
            return []
        if len(node_list) <= k:
            return list(node_list)

        scored = []
        adjacency = self.current_tree.get("adjacency", {})

        for nd in node_list:
            cost = self.tree_cache.get(nd, dst)  # tree_cache.py 中方法名为 get

            if cost is None:
                self.stats["cache_misses"] += 1
                cost = self._bfs_tree_distance(nd, dst, adjacency)
                self.tree_cache.set(nd, dst, cost)  # tree_cache.py 中方法名为 set
            else:
                self.stats["cache_hits"] += 1

            scored.append((cost, nd))

        scored.sort(key=lambda x: x[0])
        selected = [nd for _, nd in scored[:k]]
        return selected

    def _bfs_tree_distance(self, src: int, dst: int, adjacency: Dict) -> int:
        """BFS 计算树内两点距离"""
        if src == dst: return 0
        if not adjacency or src not in adjacency: return 999

        q = deque([(src, 0)])
        visited = {src}
        while q:
            u, dist = q.popleft()
            if u == dst: return dist
            for v in adjacency.get(u, []):
                if v not in visited:
                    visited.add(v)
                    q.append((v, dist + 1))
        return 999

    # --------------------------
    # 改进 2：网络状态标准化
    # --------------------------
    def _prepare_network_state(self, network_state: Dict) -> Dict:
        """统一整理 network_state 输入格式"""
        state = dict(network_state or {})
        state.setdefault("cpu", {})
        state.setdefault("mem", {})
        state.setdefault("bw", {})
        state.setdefault("hvt", {})

        req = self.current_request
        state.setdefault("cpu_demand", ensure_list(req.get("cpu_origin", [])))
        state.setdefault("mem_demand", ensure_list(req.get("memory_origin", [])))
        state.setdefault("bw_demand", float(req.get("bw_origin", 0)))
        state.setdefault("vnf_seq", ensure_list(req.get("vnf") or req.get("vnf_list") or []))
        state.setdefault("tree_nodes", list(self.current_tree.get("nodes", [])))
        state.setdefault("source", int(req.get("source", 1)))
        return state

    # --------------------------
    # 资源检查工具
    # --------------------------
    def _check_bandwidth_feasible(self, links, state):
        """
        检查链路带宽是否满足需求
        """
        bw_resources = state.get('bw')  # numpy array or dict

        # 1. 优先从已经 prepare 过的 state 中获取标准化需求
        bw_demand = state.get("bw_demand")

        # 2. 如果 state 中意外没有，尝试从 request 中获取 (兼容 bw_origin)
        if bw_demand is None:
            req = self.current_request
            # 您的数据主要使用 bw_origin
            bw_demand = req.get("bw_origin") or req.get("bw_demand") or req.get("bw")

            # 处理嵌套的 demand 字典
            if bw_demand is None and isinstance(req.get("demand"), dict):
                bw_demand = req["demand"].get("bw")

        # 3. 如果仍然找不到（或者是 0），则默认不需要带宽检查
        if bw_demand is None:
            # 记录警告但允许通过，或者抛出更明确的错误
            # 这里我们假设如果没有定义带宽，则默认为 0
            bw_demand = 0.0

        # 如果需求为 0，直接通过
        if bw_demand <= 1e-6:
            return True

        # 4. 执行检查
        for lid in links:
            idx = int(lid) - 1

            # 获取当前链路可用带宽
            if isinstance(bw_resources, (np.ndarray, list)):
                if idx < 0 or idx >= len(bw_resources):
                    logger.warning(f"[BW Check] Link ID {lid} out of bounds")
                    return False
                avail_bw = float(bw_resources[idx])
            elif isinstance(bw_resources, dict):
                avail_bw = float(bw_resources.get(idx, 0))
            else:
                # 如果没有资源信息，保守起见认为不足
                return False

            if avail_bw < bw_demand:
                return False

        return True

    def _check_node_resources_feasible(self, nodes: List[int], placement: Dict[int, int], state: Dict) -> bool:
        """检查节点资源"""
        cpu_demand = state.get("cpu_demand", [])
        mem_demand = state.get("mem_demand", [])
        cpu_res = state.get("cpu", {})
        mem_res = state.get("mem", {})

        for vnf_idx, node in placement.items():
            node_idx = int(node) - 1
            if vnf_idx >= len(cpu_demand) or vnf_idx >= len(mem_demand): continue

            cpu_need = cpu_demand[vnf_idx]
            mem_need = mem_demand[vnf_idx]

            # 使用 loose check (>= 0.1) 或者 strict check
            if cpu_res.get(node_idx, 0) < cpu_need or mem_res.get(node_idx, 0) < mem_need:
                return False
        return True

    # --------------------------
    # 策略 1: 资源感知
    # --------------------------
    def _resource_aware_strategy(self, dst: int, state: Dict) -> Dict:
        tree_nodes = state.get("tree_nodes", [])
        if not tree_nodes: return {"feasible": False}

        candidates = self._select_top_k(tree_nodes, dst, k=8)
        for cand in candidates:
            nodes, links = self.path_finder.find_any_path(cand, dst)
            if not nodes or not links: continue

            if not self._check_bandwidth_feasible(links, state): continue

            placement = VNFPlacement.resource_aware(state["vnf_seq"], nodes, state, self._is_dc_node)
            if placement is None: continue

            if not self._check_node_resources_feasible(nodes, placement, state): continue

            score = evaluate_path_score(nodes, links, state, self._is_dc_node)
            return {"feasible": True, "nodes": nodes, "links": links, "placement": placement, "score": score}

        return {"feasible": False}

    # --------------------------
    # 策略 2: Smart Greedy
    # --------------------------
    def _smart_greedy_strategy(self, dst: int, state: Dict) -> Dict:
        tree_nodes = state.get("tree_nodes", [])
        if not tree_nodes: return {"feasible": False}

        candidates = self._select_top_k(tree_nodes, dst, k=6)
        best_plan = None
        best_score = -1e9

        for cand in candidates:
            nodes, links = self.path_finder.find_any_path(cand, dst)
            if not nodes or not links: continue

            score = evaluate_path_score(nodes, links, state, self._is_dc_node)
            if score > best_score:
                best_score = score
                vnf_seq = state["vnf_seq"]
                placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}
                best_plan = {"feasible": True, "nodes": nodes, "links": links, "placement": placement, "score": score}

        return best_plan or {"feasible": False}

    # --------------------------
    # 策略 3: Minimal
    # --------------------------
    def _minimal_strategy(self, dst: int, state: Dict) -> Dict:
        tree_nodes = state.get("tree_nodes", [])
        if not tree_nodes: return {"feasible": False}

        candidates = self._select_top_k(tree_nodes, dst, k=6)
        best_plan = None
        best_hops = float('inf')

        for cand in candidates:
            nodes, links = self.path_finder.find_any_path(cand, dst)
            if not nodes or not links: continue

            hops = len(nodes) - 1
            if hops < best_hops:
                best_hops = hops
                vnf_seq = state["vnf_seq"]
                placement = VNFPlacement.simple_round_robin(vnf_seq, nodes, self._is_dc_node) if vnf_seq else {}
                best_plan = {"feasible": True, "nodes": nodes, "links": links, "placement": placement, "score": -hops}

        return best_plan or {"feasible": False}

    # --------------------------
    # 改进 3：增强 Never-Fail 策略
    # --------------------------
    def _never_fail_strategy(self, dst: int, state: Dict) -> Dict:
        """多层降级策略"""
        src = state.get("source", 1)
        node_num = getattr(self.expert, "node_num", self.n)
        tree_nodes = state.get("tree_nodes", [])

        # Layer 1: k-paths from tree nodes
        for nd in tree_nodes:
            for k in range(1, self.path_finder.max_k + 1):
                try:
                    # 修复 2: 这里的解包需要处理 3 个返回值 (nodes, dist, links)
                    # 之前写成了 nodes, links = result[:2] 是错的，因为 result[1] 是 distance
                    nodes, _, links = self.path_finder.get_k_path(nd, dst, k)

                    if nodes and links:
                        return {"feasible": True, "nodes": nodes, "links": links, "placement": {}, "score": -len(nodes)}
                except Exception:
                    pass

        # Layer 2: DC relay
        dc_nodes = [n for n in range(1, node_num + 1) if self._is_dc_node(n) and n not in {src, dst}]
        for relay in dc_nodes:
            try:
                nodes, links = self.path_finder.compose_via_relay(src, relay, dst)
                if nodes and links:
                    return {"feasible": True, "nodes": nodes, "links": links, "placement": {},
                            "score": -1.2 * len(nodes)}
            except Exception:
                pass

        # Layer 3: All relay
        for relay in range(1, node_num + 1):
            if relay in {src, dst}: continue
            try:
                nodes, links = self.path_finder.compose_via_relay(src, relay, dst)
                if nodes and links:
                    return {"feasible": True, "nodes": nodes, "links": links, "placement": {},
                            "score": -1.5 * len(nodes)}
            except Exception:
                pass

        return {"feasible": False, "nodes": [], "links": [], "placement": {}, "score": -1e9}

    # --------------------------
    # 改进 4：动态策略选择
    # --------------------------
    def _select_strategies(self, state: Dict) -> List[Tuple]:
        vnf_count = len(state.get("vnf_seq", []))
        cpu_res = state.get("cpu", {})
        avg_cpu_util = 1.0 - (sum(cpu_res.values()) / max(1, len(cpu_res) * 2000)) if cpu_res else 0.5

        if vnf_count >= 3 and avg_cpu_util > 0.7:
            return [(self._resource_aware_strategy, "resource_aware", 1.0),
                    (self._smart_greedy_strategy, "smart_greedy", 0.8),
                    (self._minimal_strategy, "minimal", 0.6)]

        if vnf_count == 0 or avg_cpu_util < 0.3:
            return [(self._minimal_strategy, "minimal", 1.0),
                    (self._smart_greedy_strategy, "smart_greedy", 0.9),
                    (self._resource_aware_strategy, "resource_aware", 0.7)]

        return [(self._smart_greedy_strategy, "smart_greedy", 1.0),
                (self._resource_aware_strategy, "resource_aware", 0.9),
                (self._minimal_strategy, "minimal", 0.8)]

    # --------------------------
    # 结果标准化
    # --------------------------
    def _normalize_plan(self, plan: Dict) -> Dict:
        if not plan.get("feasible", False):
            return {
                "feasible": False,
                "nodes": [],
                "links": [],
                "placement": {},
                "tree": np.zeros(self.L),  # 需要 numpy
                "hvt": np.zeros((self.n, self.K_vnf)),  # 需要 numpy
                "score": -1e9,
                "backup_type": plan.get("backup_type", "none"),
                "error": plan.get("error", "unknown")
            }

        nodes = plan.get("nodes", [])
        links = plan.get("links", [])
        placement = plan.get("placement", {})

        return {
            "feasible": True,
            "nodes": nodes,
            "links": links,
            "placement": placement,
            "tree": build_tree_vec(links, self.L),
            "hvt": build_hvt_from_placement(placement, self.n, self.K_vnf),
            "new_path_full": nodes,
            "backup_type": plan.get("backup_type", "hybrid"),
            "score": float(plan.get("score", 0.0))
        }

    # --------------------------
    # 主接口：get_backup_plan
    # --------------------------
    def get_backup_plan(self, goal_dest_idx: int, network_state: Dict[str, Any]) -> Dict[str, Any]:
        self.stats["total_calls"] += 1
        req = self.current_request
        if not req: return {"feasible": False, "error": "no_request"}

        try:
            dest_list = req.get("dest", [])
            if goal_dest_idx < 0 or goal_dest_idx >= len(dest_list): raise IndexError
            dst = int(dest_list[goal_dest_idx])
        except Exception:
            return {"feasible": False, "error": "invalid_dest"}

        try:
            state = self._prepare_network_state(network_state)
        except Exception as e:
            return {"feasible": False, "error": str(e)}

        strategies = self._select_strategies(state)

        for func, name, priority in strategies:
            try:
                plan = func(dst, state)
                if plan.get("feasible", False):
                    plan["score"] = plan.get("score", 0) * priority
                    plan["backup_type"] = name
                    self.stats["strategy_success"][name] += 1

                    tc = self.stats["total_calls"]
                    prev_avg = self.stats["avg_score"]
                    self.stats["avg_score"] = (prev_avg * (tc - 1) + plan.get("score", 0)) / tc

                    logger.debug(f"[BackupPolicy] Strategy {name} succeeded")
                    return self._normalize_plan(plan)
            except Exception as e:
                logger.warning(f"Strategy {name} failed: {e}", exc_info=True)

        # Never Fail
        try:
            plan = self._never_fail_strategy(dst, state)
            plan["backup_type"] = "never_fail"
            if plan.get("feasible"):
                self.stats["strategy_success"]["never_fail"] += 1
            else:
                self.stats["total_failures"] += 1
            return self._normalize_plan(plan)
        except Exception as e:
            self.stats["total_failures"] += 1
            return {"feasible": False, "error": str(e)}

    # --------------------------
    # 性能统计接口
    # --------------------------
    def get_statistics(self) -> Dict:
        stats = dict(self.stats)
        total = stats["total_calls"]
        if total > 0:
            stats["success_rate"] = 1.0 - (stats["total_failures"] / total)
            stats["cache_hit_rate"] = stats["cache_hits"] / max(1, stats["cache_hits"] + stats["cache_misses"])
        else:
            stats["success_rate"] = 0.0
            stats["cache_hit_rate"] = 0.0
        return stats

    def reset_statistics(self):
        self.stats = {
            "total_calls": 0,
            "strategy_success": {"resource_aware": 0, "smart_greedy": 0, "minimal": 0, "never_fail": 0},
            "total_failures": 0, "avg_score": 0.0, "cache_hits": 0, "cache_misses": 0
        }