#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hirl_sfc_env_gnn.py - Multicast-Aware Version
多播感知图神经网络环境
"""
import logging
import numpy as np
import torch
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any, Set

from hirl_sfc_env import SFC_HIRL_Env

logger = logging.getLogger(__name__)


class SFC_HIRL_Env_GNN(SFC_HIRL_Env):
    """
    多播感知GNN环境

    创新点:
    1. 多目标集合编码 (Set Transformer输入)
    2. VNF共享状态特征 (共享潜力预测)
    3. 请求向量增强 (多目标信息)
    4. 动态图更新 (树结构演化)
    """

    def __init__(self, input_dir, topo, dc_nodes, capacities, use_gnn=True):
        # 1. 调用父类初始化
        super().__init__(input_dir, topo, dc_nodes, capacities)

        self.use_gnn = use_gnn
        self.topo = topo

        if self.use_gnn:
            self._build_edge_index()

            # ===== 特征维度定义 =====
            # 🔥 增强节点特征
            # 原有: [CPU, Mem, DC, Src, Dst, InTree] + VNF_OneHot
            # 新增: [NumDestNearby, AvgDistToDests, VNFSharingPotential]
            self.node_feat_dim = 6 + self.K_vnf + 3  # 9 + K_vnf

            # 🔥 增强边特征
            # 原有: [BW, InTree, Hop]
            # 新增: [SharedByDests, AvgTraffic]
            self.edge_feat_dim = 3 + 2  # 5

            # 🔥 增强请求向量
            # 原有: 基础请求特征
            # 新增: [NumDests, AvgBW, SharingStrategy, DestSetEncoding...]
            self.request_dim = 10 + 4 + self.NB_HIGH_LEVEL_GOALS  # 动态长度

            # 覆盖 Observation Space
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )

            # ===== 多播专用缓存 =====
            self._dest_dist_cache: Dict[int, np.ndarray] = {}

        logger.info(f"SFC_HIRL_Env_GNN initialized (Multicast Mode). "
                    f"Node feat: {self.node_feat_dim}, Edge feat: {self.edge_feat_dim}, "
                    f"Request dim: {self.request_dim}")

    def _build_edge_index(self):
        """
        构建edge_index (完全保留原逻辑)
        """
        rows, cols = np.where(self.topo > 0)
        self.link_id_to_edge_idx = {}

        use_expert_map = hasattr(self.expert, 'link_map')
        edge_list = []

        for idx, (u, v) in enumerate(zip(rows, cols)):
            edge_list.append([u, v])

            phys_id = -1
            if use_expert_map:
                phys_id = self.expert.link_map.get((u + 1, v + 1))
                if phys_id is None:
                    phys_id = self.expert.link_map.get((v + 1, u + 1))

            if phys_id is not None and phys_id > 0:
                self.link_id_to_edge_idx[phys_id - 1] = idx
            elif not use_expert_map:
                if idx < self.L:
                    self.link_id_to_edge_idx[idx] = idx

        self.edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        self.edge_hops = torch.tensor(
            [float(self.topo[u, v]) for u, v in zip(rows, cols)],
            dtype=torch.float32
        )

        logger.info(f"GNN Graph built: {self.edge_index.shape[1]} edges.")

    def _compute_dest_distances(self, dest_set: Set[int]) -> np.ndarray:
        """
        计算每个节点到目标集合的平均距离

        Args:
            dest_set: 目标节点集合 (1-based)

        Returns:
            avg_dist: [n] 每个节点到目标集合的平均距离
        """
        cache_key = frozenset(dest_set)
        if cache_key in self._dest_dist_cache:
            return self._dest_dist_cache[cache_key]

        avg_dist = np.zeros(self.n)

        if not dest_set:
            return avg_dist

        for i in range(self.n):
            dists = []
            for dest in dest_set:
                dest_idx = dest - 1
                if 0 <= dest_idx < self.n:
                    dists.append(self.shortest_dist[i, dest_idx])

            avg_dist[i] = np.mean(dists) if dists else 999.0

        self._dest_dist_cache[cache_key] = avg_dist
        return avg_dist

    def _compute_vnf_sharing_potential(self, dest_set: Set[int]) -> np.ndarray:
        """
        计算每个节点的VNF共享潜力

        潜力公式:
            potential[i] = α * (1 - avg_dist[i] / max_dist)
                         + β * (avail_cpu[i] / max_cpu)
                         + γ * num_nearby_dests[i]

        Args:
            dest_set: 目标节点集合 (1-based)

        Returns:
            potential: [n] 每个节点的共享潜力分数 ∈ [0, 1]
        """
        # 1. 距离因子
        avg_dist = self._compute_dest_distances(dest_set)
        max_dist = np.max(avg_dist) if np.max(avg_dist) > 0 else 1.0
        dist_factor = 1.0 - avg_dist / max_dist

        # 2. 资源因子
        cpu_util = self.C / max(1, self.C_cap)
        resource_factor = 1.0 - cpu_util

        # 3. 邻近目标数量
        nearby_count = np.zeros(self.n)
        for i in range(self.n):
            for dest in dest_set:
                dest_idx = dest - 1
                if 0 <= dest_idx < self.n:
                    if self.shortest_dist[i, dest_idx] <= 3:  # 3跳内
                        nearby_count[i] += 1

        max_count = np.max(nearby_count) if np.max(nearby_count) > 0 else 1.0
        neighbor_factor = nearby_count / max_count

        # 4. 加权组合
        alpha, beta, gamma = 0.4, 0.3, 0.3
        potential = alpha * dist_factor + beta * resource_factor + gamma * neighbor_factor

        # 归一化到 [0, 1]
        potential = np.clip(potential, 0, 1)

        return potential

    def _get_graph_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成多播感知的图状态 (完整修复版)

        修复:
        - Line 251-252: paths_map 值类型检查
        - 添加详细日志用于调试

        Returns:
            x: [n, node_feat_dim] 节点特征
            edge_index: [2, E] 边索引
            edge_attr: [E, edge_feat_dim] 边特征
            request_vec: [request_dim] 请求向量
        """
        if not self.current_request:
            # 空请求: 返回零张量
            x = torch.zeros((self.n, self.node_feat_dim))
            edge_attr = torch.zeros((self.edge_index.shape[1], self.edge_feat_dim))
            request_vec = torch.zeros(self.request_dim)
            return x, self.edge_index, edge_attr, request_vec

        # ===== 1. 节点特征 [n, 9 + K_vnf] =====
        src = self.current_request['source']
        dest_set = set(self.current_request.get('dest', []))
        tree_set = self.nodes_on_tree

        # 计算辅助特征
        avg_dist = self._compute_dest_distances(dest_set)
        sharing_potential = self._compute_vnf_sharing_potential(dest_set)

        node_feats = []
        for i in range(self.n):
            nid = i + 1

            # 基础特征 (6维)
            feat = [
                1.0 - self.C[i] / max(1, self.C_cap),  # CPU利用率
                1.0 - self.M[i] / max(1, self.M_cap),  # Mem利用率
                1.0 if nid in self.expert.DC else 0.0,  # 是否DC
                1.0 if nid == src else 0.0,  # 是否源节点
                1.0 if nid in dest_set else 0.0,  # 是否目标节点
                1.0 if nid in tree_set else 0.0  # 是否在树上
            ]

            # 🔥 多播增强特征 (3维)
            num_nearby = sum(1 for d in dest_set
                             if 0 <= d - 1 < self.n and self.shortest_dist[i, d - 1] <= 3)
            feat.extend([
                num_nearby / max(1, len(dest_set)),  # 归一化邻近目标数
                1.0 - avg_dist[i] / max(1, np.max(avg_dist)),  # 归一化平均距离
                sharing_potential[i]  # VNF共享潜力
            ])

            # VNF状态 (K_vnf维)
            feat.extend((self.hvt_all[i] / 10.0).tolist())

            node_feats.append(feat)

        x = torch.tensor(node_feats, dtype=torch.float32)

        # ===== 2. 边特征 [E, 5] - 🔧 修复版 =====
        num_edges = self.edge_index.shape[1]
        edge_attrs = torch.zeros((num_edges, self.edge_feat_dim), dtype=torch.float32)

        tree_links = self.current_tree.get('tree', np.zeros(self.L)) if self.current_tree else np.zeros(self.L)

        for phys_idx in range(self.L):
            if phys_idx not in self.link_id_to_edge_idx:
                continue

            edge_idx = self.link_id_to_edge_idx[phys_idx]

            # 基础特征 (3维)
            bw_util = 1.0 - self.B[phys_idx] / max(1, self.B_cap)
            in_tree = 1.0 if tree_links[phys_idx] > 0 else 0.0
            hop = self.edge_hops[edge_idx].item()

            # ===== 🔥 修复: 计算共享率 =====
            shared_by = 0

            try:
                if self.current_tree and 'paths_map' in self.current_tree:
                    paths_map = self.current_tree['paths_map']

                    if not isinstance(paths_map, dict):
                        logger.warning(f"paths_map is not dict: {type(paths_map)}")
                    else:
                        for dest_key, path_info in paths_map.items():
                            links_list = None

                            # 🔧 关键修复: 处理不同的数据结构
                            if isinstance(path_info, dict):
                                # 情况1: {'path': [...], 'links': [...], ...}
                                links_list = path_info.get('links', [])

                            elif isinstance(path_info, list):
                                # 情况2: 直接是链路列表 [link1, link2, ...]
                                links_list = path_info

                            elif isinstance(path_info, (tuple, set)):
                                # 情况3: tuple/set 形式
                                links_list = list(path_info)

                            else:
                                # 情况4: 未知类型,记录警告
                                logger.debug(f"Unknown path_info type: {type(path_info)} for dest {dest_key}")
                                continue

                            # 检查当前链路是否在这条路径中
                            if links_list is not None and phys_idx in links_list:
                                shared_by += 1

            except Exception as e:
                # 防御性: 即使出错也不影响其他特征
                logger.debug(f"Error computing shared_by for link {phys_idx}: {e}")
                shared_by = 0

            # 归一化共享率
            shared_rate = shared_by / max(1, len(dest_set)) if dest_set else 0.0

            # 平均流量 (简化版: 使用带宽利用率)
            avg_traffic = bw_util

            edge_attrs[edge_idx] = torch.tensor([
                bw_util, in_tree, hop, shared_rate, avg_traffic
            ])

        # ===== 3. 请求向量 [request_dim] =====
        # 复用父类的扁平状态作为基础
        flat_state = super(type(self), self)._get_flat_state()
        base_req_vec = flat_state[self.dim_network:]  # 基础请求特征

        # 🔥 多播增强特征
        num_dests = len(dest_set)
        num_served = len(self.served_destinations) if hasattr(self, 'served_destinations') else 0
        vnf_chain = self.current_request.get('vnf', [])
        avg_bw = self.current_request.get('bandwidth', 0) / max(1, num_dests)

        multicast_features = [
            num_dests / max(1, self.NB_HIGH_LEVEL_GOALS),  # 归一化目标数
            num_served / max(1, num_dests),  # 服务进度
            avg_bw / max(1, self.B_cap),  # 归一化平均带宽
            self.sharing_strategy / 3.0 if hasattr(self, 'sharing_strategy') else 0.0  # 共享策略
        ]

        # 目标集合one-hot编码
        dest_encoding = np.zeros(self.NB_HIGH_LEVEL_GOALS)
        for d_idx, dest in enumerate(self.current_request.get('dest', [])):
            if 0 <= d_idx < self.NB_HIGH_LEVEL_GOALS:
                dest_encoding[d_idx] = 1.0

        # 拼接所有特征
        request_vec = np.concatenate([
            base_req_vec[:10],  # 基础请求特征 (前10维)
            multicast_features,  # 多播特征 (4维)
            dest_encoding  # 目标集合编码 (NB_HIGH_LEVEL_GOALS维)
        ])

        request_vec = torch.tensor(request_vec, dtype=torch.float32)

        return x, self.edge_index, edge_attrs, request_vec

    # ===== 接口重写 =====
    def get_state(self):
        """统一状态获取接口"""
        if self.use_gnn:
            return self._get_graph_state()
        return super()._get_flat_state()

    def reset_request(self):
        """重置请求 (调用父类后返回GNN状态)"""
        req, _ = super().reset_request()
        return req, self.get_state()

    def step_low_level(self, goal, action):
        """
        执行低层动作 (调用父类后返回GNN状态)
        """
        _, reward, sub_done, req_done = super().step_low_level(goal, action)
        return self.get_state(), reward, sub_done, req_done

    def clear_cache(self):
        """清空缓存"""
        super()._clear_cache()
        if hasattr(self, '_dest_dist_cache'):
            self._dest_dist_cache.clear()


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    from pathlib import Path
    import hyperparameters as H

    try:
        print("=" * 80)
        print("Initializing Multicast-Aware GNN Env...")
        print("=" * 80)

        topo = H.TOPOLOGY_MATRIX if hasattr(H, 'TOPOLOGY_MATRIX') else np.eye(14)
        dc_nodes = H.DC_NODES if hasattr(H, 'DC_NODES') else [1, 4]
        caps = H.CAPACITIES if hasattr(H, 'CAPACITIES') else {
            'bandwidth': 100, 'cpu': 100, 'memory': 100
        }

        env = SFC_HIRL_Env_GNN(H.INPUT_DIR, topo, dc_nodes, caps, use_gnn=True)

        print("\n" + "=" * 80)
        print("Testing state extraction...")
        print("=" * 80)

        req, state = env.reset_request()

        if req is not None:
            x, ei, ea, r = state
            print(f"\n✅ Success!")
            print(f"  Node features: {x.shape} (expected: [{env.n}, {env.node_feat_dim}])")
            print(f"  Edge index: {ei.shape}")
            print(f"  Edge attributes: {ea.shape} (expected: [{ei.shape[1]}, {env.edge_feat_dim}])")
            print(f"  Request vector: {r.shape} (expected: [{env.request_dim}])")
            print(f"\n  Request info:")
            print(f"    - ID: {req['id']}")
            print(f"    - Source: {req['source']}")
            print(f"    - Destinations: {req.get('dest', [])}")
            print(f"    - VNF chain: {req.get('vnf', [])}")
        else:
            print("❌ No request available")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()