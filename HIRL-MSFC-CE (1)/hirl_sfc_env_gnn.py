#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hirl_sfc_env_gnn.py - Final Integrated Version
通过继承机制完美复用原有逻辑，仅重写 GNN 状态提取部分。
"""
import logging
import numpy as np
import torch
from gym import spaces
from typing import Dict, List, Tuple, Optional, Any

# 导入原环境 (确保 hirl_sfc_env.py 在同一目录下)
from hirl_sfc_env import SFC_HIRL_Env

logger = logging.getLogger(__name__)


class SFC_HIRL_Env_GNN(SFC_HIRL_Env):
    """
    GNN 适配版环境
    继承自 SFC_HIRL_Env，自动获得所有核心逻辑 (step, reset, backup policy等)
    仅重写与状态观测相关的接口。
    """

    def __init__(self, input_dir, topo, dc_nodes, capacities, use_gnn=True):
        # 1. 调用父类初始化 (加载数据、初始化Expert、BackupPolicy、Logger等)
        super().__init__(input_dir, topo, dc_nodes, capacities)

        self.use_gnn = use_gnn
        self.topo = topo

        # 2. GNN 专用初始化
        if self.use_gnn:
            self._build_edge_index()
            # 特征维度定义 (需与模型匹配)
            self.node_feat_dim = 6 + self.K_vnf  # [CPU, Mem, DC, Src, Dst, InTree] + VNF_OneHot
            self.edge_feat_dim = 3  # [BW, InTree, Hop]
            self.request_dim = 10  # Request Vector

            # 覆盖 Observation Space (虽然 GNN 不用这个，但为了规范)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
            )

        logger.info(f"SFC_HIRL_Env_GNN initialized. Mode: {'GNN' if use_gnn else 'Flat'}")

    def _build_edge_index(self):
        """
        构建 edge_index 并建立物理链路映射 (Critical Fix)
        """
        rows, cols = np.where(self.topo > 0)

        # 建立物理链路 ID (1-based from expert) 到图边索引 (0-based) 的映射
        self.link_id_to_edge_idx = {}

        # 优先尝试从 expert 获取映射
        use_expert_map = hasattr(self.expert, 'link_map')

        edge_list = []
        for idx, (u, v) in enumerate(zip(rows, cols)):
            edge_list.append([u, v])

            phys_id = -1
            if use_expert_map:
                # 尝试有向 (u+1, v+1)
                phys_id = self.expert.link_map.get((u + 1, v + 1))
                if phys_id is None:  # 尝试无向 (v+1, u+1)
                    phys_id = self.expert.link_map.get((v + 1, u + 1))

            # 如果找到了物理ID (转0-based索引)
            if phys_id is not None and phys_id > 0:
                self.link_id_to_edge_idx[phys_id - 1] = idx
            elif not use_expert_map:
                # 回退策略：假设 topo 遍历顺序与物理 ID 顺序一致
                if idx < self.L:
                    self.link_id_to_edge_idx[idx] = idx

        self.edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)
        # 缓存边的跳数信息
        self.edge_hops = torch.tensor([float(self.topo[u, v]) for u, v in zip(rows, cols)], dtype=torch.float32)

        logger.info(f"GNN Graph built: {self.edge_index.shape[1]} edges.")

    def get_state(self):
        """统一状态获取接口"""
        if self.use_gnn:
            return self._get_graph_state()
        return super()._get_flat_state()

    def _get_graph_state(self):
        """
        生成 (x, edge_index, edge_attr, req_vec)
        """
        # 1. Node Features [N, 6 + K_vnf]
        node_feats = []
        src = self.current_request['source'] if self.current_request else -1
        dests = set(self.current_request.get('dest', [])) if self.current_request else set()
        tree_set = self.nodes_on_tree

        for i in range(self.n):
            nid = i + 1
            # 基础特征
            feat = [
                1.0 - self.C[i] / max(1, self.C_cap),  # CPU Util
                1.0 - self.M[i] / max(1, self.M_cap),  # Mem Util
                1.0 if nid in self.expert.DC else 0.0,  # Is DC
                1.0 if nid == src else 0.0,  # Is Source
                1.0 if nid in dests else 0.0,  # Is Dest
                1.0 if nid in tree_set else 0.0  # In Tree
            ]
            # VNF 状态 (归一化计数)
            feat.extend((self.hvt_all[i] / 10.0).tolist())
            node_feats.append(feat)

        x = torch.tensor(node_feats, dtype=torch.float32)

        # 2. Edge Features [E, 3]
        num_edges = self.edge_index.shape[1]
        edge_attrs = torch.zeros((num_edges, self.edge_feat_dim), dtype=torch.float32)

        tree_links = self.current_tree.get('tree', np.zeros(self.L)) if self.current_tree else np.zeros(self.L)

        # 填充边特征
        for phys_idx in range(self.L):
            if phys_idx in self.link_id_to_edge_idx:
                edge_idx = self.link_id_to_edge_idx[phys_idx]

                bw_util = 1.0 - self.B[phys_idx] / max(1, self.B_cap)
                in_tree = 1.0 if tree_links[phys_idx] > 0 else 0.0
                hop = self.edge_hops[edge_idx].item()

                edge_attrs[edge_idx] = torch.tensor([bw_util, in_tree, hop])

        # 3. Req Vec (复用父类逻辑)
        flat = super()._get_flat_state()
        req_vec = flat[self.dim_network:]  # 截取后半部分请求特征

        return x, self.edge_index, edge_attrs, req_vec

    # 重写 reset 和 step 以适配新接口 (虽然父类有，但我们需要确保返回的是 get_state())
    def reset_request(self):
        req, _ = super().reset_request()
        return req, self.get_state()

    def step_low_level(self, goal, action):
        # 调用父类的核心逻辑
        _, reward, sub_done, req_done = super().step_low_level(goal, action)
        # 返回新的 GNN 状态
        return self.get_state(), reward, sub_done, req_done


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    from pathlib import Path
    import hyperparameters as H

    # 简单的冒烟测试
    try:
        print("Initializing GNN Env...")
        # 假设 hyperparameters.py 里有这些配置，或者手动指定
        topo = H.TOPOLOGY_MATRIX if hasattr(H, 'TOPOLOGY_MATRIX') else np.eye(14)
        dc_nodes = H.DC_NODES if hasattr(H, 'DC_NODES') else [1, 4]
        caps = H.CAPACITIES if hasattr(H, 'CAPACITIES') else {'bandwidth': 100, 'cpu': 100, 'memory': 100}

        env = SFC_HIRL_Env_GNN(H.INPUT_DIR, topo, dc_nodes, caps, use_gnn=True)
        req, state = env.reset_request()

        x, ei, ea, r = state
        print(f"Success! State shapes: X={x.shape}, EI={ei.shape}, EA={ea.shape}, R={r.shape}")

    except Exception as e:
        print(f"Test failed: {e}")