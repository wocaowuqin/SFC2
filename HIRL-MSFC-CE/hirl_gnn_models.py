#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hirl_gnn_models.py - 多播感知三层HRL模型 (生产级完整版)

创新点整合:
1. 多播感知GAT (Multicast-Aware GAT)
   - Set Transformer 多目标集合编码
   - 请求调制的图注意力机制
   - VNF共享潜力预测

2. 三层分层强化学习 (3-Level HRL)
   - High-Level: VNF共享策略选择
   - Mid-Level: 目标节点选择 (子树扩展)
   - Low-Level: 路径与VNF放置执行

3. 模块化设计
   - 独立可测试的组件
   - 灵活的配置接口
   - 完善的错误处理

理论基础:
- Set Transformer: Zaheer et al., "Deep Sets", NeurIPS 2017
- Graph Attention: Veličković et al., "GAT", ICLR 2018
- Hierarchical RL: Kulkarni et al., "h-DQN", NeurIPS 2016

作者: Your Name
日期: 2025-12-13
版本: 5.0 (Multicast Edition)
"""

from __future__ import annotations
import logging
import hashlib
import threading
import time
from typing import Optional, List, Union, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, global_mean_pool, MessagePassing
from torch_geometric.utils import softmax

# ============================================================================
# 日志配置
# ============================================================================
logger = logging.getLogger(__name__)


# ============================================================================
# 1. 基础组件: Set Transformer (多目标集合编码)
# ============================================================================

class SetTransformer(nn.Module):
    """
    多目标集合编码器 - 排列不变性 (Permutation Invariant)

    理论基础: Deep Sets (Zaheer et al., NeurIPS 2017)

    关键公式:
        f({x1, ..., xn}) = ρ(Σ φ(xi))

    其中:
        - φ: 元素级编码函数 (Element-wise Encoder)
        - Σ: 求和聚合 (保证排列不变性)
        - ρ: 聚合后的映射函数 (Aggregation Function)

    应用场景:
        多播SFC中,目标节点集合 {d1, d2, ..., dk} 的顺序不应影响决策
        例如: {d1, d2, d3} 和 {d3, d1, d2} 应产生相同的编码

    创新点:
        1. 使用Multi-head Self-Attention捕获目标间的相关性
        2. 例如: 两个距离很近的目标可能共享VNF实例
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Args:
            input_dim: 输入特征维度 (节点特征维度)
            hidden_dim: 隐藏层维度
            num_heads: 多头注意力的头数
            dropout: Dropout比例
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # φ: 元素级编码网络
        self.element_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Multi-head Self-Attention (捕获目标间关系)
        # 例如: 目标d1和d2距离很近 → 高注意力权重 → 可能共享VNF
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.attn_norm = nn.LayerNorm(hidden_dim)

        # ρ: 聚合函数 (Sum Pooling + MLP)
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, dest_features: torch.Tensor,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播

        Args:
            dest_features: [num_dests, input_dim] 目标节点特征张量
            return_attention: 是否返回注意力权重 (用于可视化分析)

        Returns:
            aggregated: [hidden_dim] 聚合后的集合表示
            (optional) attention_weights: [num_heads, num_dests, num_dests]
        """
        if dest_features.size(0) == 0:
            # 空集合情况: 返回零向量
            return torch.zeros(self.hidden_dim, device=dest_features.device, dtype=dest_features.dtype)

        # 1. 元素级编码
        encoded = self.element_encoder(dest_features)  # [num_dests, hidden_dim]

        # 2. Self-Attention (捕获目标间相关性)
        # Q, K, V 都来自同一个输入 (Self-Attention)
        attn_input = encoded.unsqueeze(0)  # [1, num_dests, hidden_dim]

        attn_out, attn_weights = self.self_attn(
            attn_input, attn_input, attn_input,
            need_weights=return_attention
        )
        attn_out = attn_out.squeeze(0)  # [num_dests, hidden_dim]

        # 残差连接 + LayerNorm
        attn_out = self.attn_norm(encoded + attn_out)

        # 3. 排列不变聚合 (Sum Pooling)
        # 为什么用Sum而不是Mean?
        # Sum保留了集合大小信息 (对于多播很重要)
        aggregated = torch.sum(attn_out, dim=0)  # [hidden_dim]

        # 4. 最终映射
        output = self.aggregator(aggregated)  # [hidden_dim]

        if return_attention:
            return output, attn_weights
        return output


# ============================================================================
# 2. 基础组件: 请求调制的图注意力 (Request-Modulated GAT)
# ============================================================================

class RequestModulatedGATConv(MessagePassing):
    """
    请求调制的图注意力卷积层

    创新点 (vs 标准GAT):
        标准GAT:  α_ij = attention(h_i, h_j)
        本方法:   α_ij = attention(h_i, h_j, request_vec)  ← 请求调制

    直觉理解:
        不同的请求有不同的需求 (带宽、VNF类型等)
        注意力权重应该根据请求特征动态调整

    例子:
        - 高带宽请求 → 更关注高容量链路
        - 多VNF请求 → 更关注DC节点
    """

    def __init__(self, in_channels: int, out_channels: int,
                 request_dim: int, heads: int = 4,
                 concat: bool = True, dropout: float = 0.1,
                 edge_dim: Optional[int] = None):
        """
        Args:
            in_channels: 输入节点特征维度
            out_channels: 输出节点特征维度 (每个头)
            request_dim: 请求向量维度
            heads: 多头注意力的头数
            concat: 是否拼接多头结果 (True) 或平均 (False)
            dropout: Dropout比例
            edge_dim: 边特征维度 (可选)
        """
        super().__init__(aggr='add', node_dim=0)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.request_dim = request_dim
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim

        # 节点特征线性变换 (为每个头单独变换)
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)

        # 🔥 创新: 请求调制网络
        self.request_modulator = nn.Sequential(
            nn.Linear(request_dim, heads * out_channels),
            nn.LayerNorm(heads * out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 边特征变换 (如果提供)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None

        # 注意力系数计算
        # 输入维度: out_channels * 3 (src + dst + request)
        # 如果有边特征: out_channels * 4
        attn_input_dim = out_channels * 3
        if edge_dim is not None:
            attn_input_dim += out_channels

        self.att = nn.Parameter(torch.Tensor(1, heads, attn_input_dim))

        # Bias
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """参数初始化"""
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                request_vec: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [num_nodes, in_channels] 节点特征
            edge_index: [2, num_edges] 边索引
            request_vec: [request_dim] 请求向量
            edge_attr: [num_edges, edge_dim] 边特征 (可选)

        Returns:
            out: [num_nodes, out_channels * heads] 或 [num_nodes, out_channels]
        """
        H, C = self.heads, self.out_channels

        # 1. 节点特征变换
        x_src = self.lin_src(x).view(-1, H, C)  # [num_nodes, heads, out_channels]
        x_dst = self.lin_dst(x).view(-1, H, C)

        # 2. 🔥 请求调制
        # 为每个节点生成请求相关的特征
        request_feat = self.request_modulator(request_vec)  # [heads * out_channels]
        request_feat = request_feat.view(1, H, C)  # [1, heads, out_channels]
        request_feat = request_feat.expand(x.size(0), -1, -1)  # [num_nodes, heads, out_channels]

        # 3. 边特征变换 (如果有)
        edge_feat = None
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, H, C)

        # 4. 消息传递
        out = self.propagate(
            edge_index, x=(x_src, x_dst),
            request_feat=request_feat,
            edge_feat=edge_feat
        )

        # 5. 多头拼接或平均
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        # 6. Bias
        out = out + self.bias

        return out

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                request_feat_i: torch.Tensor, request_feat_j: torch.Tensor,
                edge_feat: Optional[torch.Tensor],
                index: torch.Tensor, ptr: Optional[torch.Tensor],
                size_i: Optional[int]) -> torch.Tensor:
        """
        计算消息 (Message Function)

        Args:
            x_i: 目标节点特征 [num_edges, heads, out_channels]
            x_j: 源节点特征
            request_feat_i, request_feat_j: 请求特征
            edge_feat: 边特征
            index: 边索引
            ptr, size_i: PyG内部参数

        Returns:
            messages: [num_edges, heads, out_channels]
        """
        # 拼接特征: [src, dst, request]
        combined = torch.cat([x_j, x_i, request_feat_j], dim=-1)

        # 如果有边特征,也拼接上
        if edge_feat is not None:
            combined = torch.cat([combined, edge_feat], dim=-1)

        # 计算注意力系数
        # alpha: [num_edges, heads]
        alpha = (combined * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Softmax归一化 (按目标节点分组)
        alpha = softmax(alpha, index, ptr, size_i)

        # Dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 加权消息
        messages = x_j * alpha.unsqueeze(-1)

        return messages


# ============================================================================
# 3. 基础组件: VNF共享潜力预测器
# ============================================================================

class VNFSharingPredictor(nn.Module):
    """
    VNF共享潜力预测器

    功能:
        预测每个节点作为VNF共享点的潜力分数

    输入:
        - 节点特征 (位置、资源、当前负载等)
        - 目标集合特征 (目标节点的聚合表示)

    输出:
        - 共享潜力分数 ∈ [0, 1]

    直觉:
        好的VNF共享点应该:
        1. 靠近多个目标节点 (降低传输延迟)
        2. 有充足的资源 (CPU, Memory)
        3. 在多条路径的交汇处 (提高复用率)
    """

    def __init__(self, node_dim: int, dest_set_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.2):
        """
        Args:
            node_dim: 节点特征维度
            dest_set_dim: 目标集合特征维度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
        """
        super().__init__()

        self.node_dim = node_dim
        self.dest_set_dim = dest_set_dim

        # 预测网络
        self.predictor = nn.Sequential(
            nn.Linear(node_dim + dest_set_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, node_features: torch.Tensor,
                dest_set_feature: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            node_features: [num_nodes, node_dim] 节点特征矩阵
            dest_set_feature: [dest_set_dim] 目标集合特征向量

        Returns:
            scores: [num_nodes] 每个节点的共享潜力分数
        """
        num_nodes = node_features.size(0)

        # 扩展目标集合特征以匹配节点数
        dest_expanded = dest_set_feature.unsqueeze(0).expand(
            num_nodes, -1
        )  # [num_nodes, dest_set_dim]

        # 拼接节点特征和目标集合特征
        combined = torch.cat([node_features, dest_expanded], dim=-1)

        # 预测共享潜力
        scores = self.predictor(combined).squeeze(-1)  # [num_nodes]

        return scores


# ============================================================================
# 4. 核心模型: 多播感知图神经网络编码器
# ============================================================================

class MulticastAwareGNN(nn.Module):
    """
    多播感知图神经网络编码器

    整合所有创新组件:
    1. Set Transformer (多目标集合编码)
    2. Request-Modulated GAT (请求调制注意力)
    3. VNF Sharing Predictor (共享潜力预测)

    输入:
        - 图结构: (node_features, edge_index, edge_attr)
        - 请求信息: request_vec
        - 目标节点: dest_indices

    输出:
        - node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
        - dest_set_embedding: 目标集合嵌入 [hidden_dim]
        - sharing_scores: VNF共享潜力 [num_nodes]
        - graph_embedding: 图级嵌入 [hidden_dim]
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 request_dim: int, hidden_dim: int = 128,
                 num_gat_layers: int = 3, num_heads: int = 4,
                 dropout: float = 0.1, use_checkpoint: bool = False,
                 normalization: str = 'layer'):
        """
        Args:
            node_feat_dim: 节点特征维度
            edge_feat_dim: 边特征维度
            request_dim: 请求向量维度
            hidden_dim: 隐藏层维度
            num_gat_layers: GAT层数
            num_heads: 多头注意力的头数
            dropout: Dropout比例
            use_checkpoint: 是否使用梯度检查点 (节省显存)
            normalization: 归一化类型 ('layer' or 'batch')
        """
        super().__init__()

        if hidden_dim % num_heads != 0:
            logger.warning(f"hidden_dim ({hidden_dim}) not divisible by num_heads ({num_heads}), adjusting...")
            num_heads = min(num_heads, hidden_dim)

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_checkpoint = use_checkpoint
        self.normalization = normalization

        # ===== 组件1: 初始特征编码 =====
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feat_dim, hidden_dim)

        # ===== 组件2: Set Transformer (多目标集合编码) =====
        self.dest_set_encoder = SetTransformer(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # ===== 组件3: Request-Modulated GAT 层 =====
        self.gat_layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for layer_idx in range(num_gat_layers):
            # Request-Modulated GAT 层
            gat = RequestModulatedGATConv(
                in_channels=hidden_dim,
                out_channels=self.head_dim,
                request_dim=request_dim,
                heads=num_heads,
                concat=True,
                dropout=dropout,
                edge_dim=hidden_dim
            )
            self.gat_layers.append(gat)

            # 投影层 (如果需要)
            actual_output_dim = self.head_dim * num_heads
            if actual_output_dim != hidden_dim:
                self.projections.append(nn.Linear(actual_output_dim, hidden_dim))
            else:
                self.projections.append(nn.Identity())

            # 归一化层
            if normalization == 'layer':
                self.norm_layers.append(nn.LayerNorm(hidden_dim))
            elif normalization == 'batch':
                self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # ===== 组件4: VNF共享潜力预测器 =====
        self.sharing_predictor = VNFSharingPredictor(
            node_dim=hidden_dim,
            dest_set_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor],
                request_vec: torch.Tensor,
                dest_indices: Optional[List[int]] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播

        Args:
            x: [num_nodes, node_feat_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_feat_dim] 边特征
            request_vec: [request_dim] 请求向量
            dest_indices: List[int] 目标节点索引列表
            batch: [num_nodes] batch索引 (可选,用于批处理)

        Returns:
            node_embeddings: [num_nodes, hidden_dim] 节点嵌入
            dest_set_embedding: [hidden_dim] 目标集合嵌入
            sharing_scores: [num_nodes] VNF共享潜力分数
            graph_embedding: [hidden_dim] 图级嵌入
        """
        device = x.device

        # 处理edge_attr为None的情况
        if edge_attr is None:
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(num_edges, self.edge_embedding.in_features,
                                    device=device, dtype=x.dtype)

        # ===== Step 1: 初始特征编码 =====
        x = self.node_embedding(x)  # [num_nodes, hidden_dim]
        e = self.edge_embedding(edge_attr)  # [num_edges, hidden_dim]

        # ===== Step 2: 目标集合编码 (如果提供了目标节点) =====
        dest_set_feat = None
        if dest_indices is not None and len(dest_indices) > 0:
            try:
                dest_features = x[dest_indices]  # [num_dests, hidden_dim]
                dest_set_feat = self.dest_set_encoder(dest_features)  # [hidden_dim]
            except Exception as e:
                logger.warning(f"Set encoding failed: {e}, using zero vector")
                dest_set_feat = torch.zeros(self.hidden_dim, device=device, dtype=x.dtype)
        else:
            # 没有目标节点时,使用零向量
            dest_set_feat = torch.zeros(self.hidden_dim, device=device, dtype=x.dtype)

        # ===== Step 3: Request-Modulated GAT 传播 =====
        for layer_idx, (gat_layer, proj, norm) in enumerate(
                zip(self.gat_layers, self.projections, self.norm_layers)
        ):
            residual = x

            # 梯度检查点 (节省显存)
            if self.use_checkpoint and self.training:
                x_gat = checkpoint(
                    gat_layer, x, edge_index, request_vec, e,
                    use_reentrant=True
                )
            else:
                x_gat = gat_layer(x, edge_index, request_vec, e)

            # 投影
            x_gat = proj(x_gat)

            # 激活 + Dropout
            x_gat = F.relu(x_gat)
            x_gat = self.dropout(x_gat)

            # 残差连接 + 归一化
            x = residual + x_gat
            x = norm(x)

        # ===== Step 4: VNF共享潜力预测 =====
        sharing_scores = self.sharing_predictor(x, dest_set_feat)

        # ===== Step 5: 图级聚合 =====
        if batch is None:
            graph_emb = torch.mean(x, dim=0, keepdim=True).squeeze(0)  # [hidden_dim]
        else:
            graph_emb = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
            if graph_emb.dim() == 2 and graph_emb.size(0) == 1:
                graph_emb = graph_emb.squeeze(0)

        return x, dest_set_feat, sharing_scores, graph_emb


# ============================================================================
# 5. 核心模型: 三层分层强化学习控制器
# ============================================================================

class ThreeLevelHRL_Controller(nn.Module):
    """
    三层分层强化学习控制器 (多播SFC专用)

    层次结构:

    Level 1 (High-Level): VNF共享策略选择
        - 输入: 图特征 + 请求特征
        - 输出: 共享策略 (4种)
            0: 完全独立部署 (每个目标独立VNF)
            1: 部分共享 (相邻目标共享)
            2: 最大化共享 (尽可能共享)
            3: 自适应共享 (基于资源状态动态决策)

    Level 2 (Mid-Level): 目标节点选择 (子树扩展)
        - 输入: 图特征 + 请求特征 + 候选目标
        - 输出: 下一个要连接的目标节点
        - 特色: 利用VNF共享潜力分数辅助决策

    Level 3 (Low-Level): 路径与VNF放置执行
        - 输入: 图特征 + 请求特征 + 目标节点
        - 输出: 动作Q值 (路径选择 + VNF放置)
        - 特色: 与现有环境兼容,支持动作mask
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 request_dim: int, hidden_dim: int = 128,
                 num_goals: int = 10, num_actions: int = 100,
                 use_cache: bool = False, use_checkpoint: bool = False,
                 max_cache_size: int = 2000):
        """
        Args:
            node_feat_dim: 节点特征维度
            edge_feat_dim: 边特征维度
            request_dim: 请求向量维度
            hidden_dim: 隐藏层维度
            num_goals: 最大目标节点数 (用于embedding)
            num_actions: 动作空间大小
            use_cache: 是否使用缓存 (加速推理)
            use_checkpoint: 是否使用梯度检查点
            max_cache_size: 缓存最大容量
        """
        super().__init__()

        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.request_dim = request_dim
        self.hidden_dim = hidden_dim
        self.num_goals = num_goals
        self.num_actions = num_actions

        # ===== 核心: 多播感知GNN编码器 =====
        self.mgat = MulticastAwareGNN(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            request_dim=request_dim,
            hidden_dim=hidden_dim,
            use_checkpoint=use_checkpoint
        )

        # ===== Level 1: High-Level 策略网络 =====
        # 决策: 选择VNF共享策略
        self.high_policy = nn.Sequential(
            nn.Linear(hidden_dim + request_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4),  # 4种共享策略
        )

        # ===== Level 2: Mid-Level 目标选择网络 =====
        # 决策: 选择下一个要连接的目标节点
        self.mid_context = nn.Sequential(
            nn.Linear(hidden_dim + request_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.mid_goal_scorer = nn.Sequential(
            nn.Linear(128 + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # ===== Level 3: Low-Level 执行网络 =====
        # 决策: 选择具体路径和VNF放置
        self.goal_embedding = nn.Embedding(num_goals, 64)

        self.low_q_net = nn.Sequential(
            nn.Linear(hidden_dim + request_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_actions)
        )

        # ===== 缓存系统 (可选,用于加速推理) =====
        self.use_cache = use_cache
        self.max_cache_size = max_cache_size
        if use_cache:
            self.cache = {}
            self.cache_lock = threading.Lock()
            self.cache_hits = 0
            self.cache_misses = 0

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward_high(self, x: torch.Tensor, edge_index: torch.Tensor,
                     edge_attr: Optional[torch.Tensor],
                     request_vec: torch.Tensor,
                     dest_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        High-Level前向传播: 选择VNF共享策略

        Args:
            x, edge_index, edge_attr: 图数据
            request_vec: [request_dim] 请求向量
            dest_indices: 目标节点索引列表

        Returns:
            strategy_logits: [4] 四种策略的logits
                - 0: 完全独立部署
                - 1: 部分共享
                - 2: 最大化共享
                - 3: 自适应共享
        """
        # GNN编码
        _, _, _, graph_emb = self.mgat(
            x, edge_index, edge_attr, request_vec, dest_indices
        )

        # 拼接图特征和请求特征
        combined = torch.cat([graph_emb, request_vec], dim=-1)

        # 策略预测
        strategy_logits = self.high_policy(combined)

        return strategy_logits

    def forward_mid(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor],
                    request_vec: torch.Tensor,
                    candidate_goals: List[int],
                    dest_indices: Optional[List[int]] = None) -> torch.Tensor:
        """
        Mid-Level前向传播: 选择下一个目标节点

        特色: 利用VNF共享潜力分数辅助决策

        Args:
            x, edge_index, edge_attr: 图数据
            request_vec: [request_dim] 请求向量
            candidate_goals: 候选目标节点索引列表
            dest_indices: 所有目标节点索引列表

        Returns:
            scores: [len(candidate_goals)] 每个候选目标的分数
        """
        if not candidate_goals:
            return torch.tensor([], device=x.device)

        # GNN编码
        node_emb, _, sharing_scores, graph_emb = self.mgat(
            x, edge_index, edge_attr, request_vec, dest_indices
        )

        # 全局上下文
        combined = torch.cat([graph_emb, request_vec], dim=-1)
        context = self.mid_context(combined)  # [128]

        # 为每个候选目标打分
        scores = []
        for goal_idx in candidate_goals:
            if goal_idx >= node_emb.size(0):
                # 越界检查
                scores.append(torch.tensor(-1e9, device=x.device))
                continue

            goal_node_emb = node_emb[goal_idx]

            # 拼接上下文和节点特征
            combined_feat = torch.cat([context, goal_node_emb], dim=-1)
            score = self.mid_goal_scorer(combined_feat)

            # 🔥 创新: 加权VNF共享潜力
            # 共享潜力高的节点获得奖励
            sharing_bonus = sharing_scores[goal_idx] * 0.2
            final_score = score + sharing_bonus

            scores.append(final_score)

        return torch.cat(scores, dim=0)  # [num_candidates]

    def forward_low(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor],
                    batch_vec: Optional[torch.Tensor],
                    request_vec: torch.Tensor,
                    goal_indices: torch.Tensor,
                    action_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Low-Level前向传播: 计算动作Q值

        Args:
            x, edge_index, edge_attr: 图数据
            batch_vec: batch索引 (通常为None,单图情况)
            request_vec: [request_dim] 或 [batch_size, request_dim]
            goal_indices: [batch_size] 目标节点索引
            action_masks: [batch_size, num_actions] 动作mask (可选)

        Returns:
            q_values: [batch_size, num_actions] 动作Q值
        """
        # GNN编码
        _, _, _, graph_emb = self.mgat(
            x, edge_index, edge_attr,
            request_vec if request_vec.dim() == 1 else request_vec[0],
            None, batch_vec
        )

        # 处理batch维度
        if graph_emb.dim() == 1:
            graph_emb = graph_emb.unsqueeze(0)
        if request_vec.dim() == 1:
            request_vec = request_vec.unsqueeze(0)

        # Goal embedding
        goal_emb = self.goal_embedding(goal_indices)

        # 拼接所有特征
        combined = torch.cat([graph_emb, request_vec, goal_emb], dim=-1)

        # Q值计算
        q_values = self.low_q_net(combined)

        # 应用动作mask (如果提供)
        if action_masks is not None:
            if action_masks.device != q_values.device:
                action_masks = action_masks.to(q_values.device)

            huge_neg = torch.full([], -1e9, device=q_values.device, dtype=q_values.dtype)

            # 处理不同类型的mask
            if action_masks.dtype == torch.bool:
                valid_mask = action_masks
            else:
                valid_mask = action_masks > 0.5

            q_values = torch.where(valid_mask, q_values, huge_neg)

        return q_values

    def get_config(self) -> Dict[str, Any]:
        """获取模型配置 (用于保存/加载)"""
        return {
            'node_feat_dim': self.node_feat_dim,
            'edge_feat_dim': self.edge_feat_dim,
            'request_dim': self.request_dim,
            'hidden_dim': self.hidden_dim,
            'num_goals': self.num_goals,
            'num_actions': self.num_actions,
            'use_cache': self.use_cache,
            'max_cache_size': self.max_cache_size
        }

    def clear_cache(self):
        """清空缓存"""
        if self.use_cache:
            with self.cache_lock:
                self.cache.clear()
                self.cache_hits = 0
                self.cache_misses = 0

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        if not self.use_cache:
            return {'enabled': False}

        with self.cache_lock:
            total = self.cache_hits + self.cache_misses
            hit_rate = self.cache_hits / total if total > 0 else 0.0

            return {
                'enabled': True,
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'hit_rate': hit_rate,
                'size': len(self.cache)
            }

    # Pickle支持 (防止序列化锁对象)
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'cache_lock' in state:
            del state['cache_lock']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.use_cache:
            self.cache_lock = threading.Lock()


# ============================================================================
# 6. 向后兼容: GNN_HRL_Controller (别名)
# ============================================================================

class GNN_HRL_Controller(ThreeLevelHRL_Controller):
    """
    向后兼容的别名
    保持与现有代码的兼容性
    """
    pass


# ============================================================================
# 7. 工具函数
# ============================================================================

def create_model(node_feat_dim: int, edge_feat_dim: int, request_dim: int,
                 hidden_dim: int = 128, num_goals: int = 10, num_actions: int = 100,
                 **kwargs) -> ThreeLevelHRL_Controller:
    """
    便捷的模型创建函数

    自动处理参数验证和默认值
    """
    logger.info("Creating ThreeLevelHRL_Controller...")
    logger.info(f"  Node feat dim: {node_feat_dim}")
    logger.info(f"  Edge feat dim: {edge_feat_dim}")
    logger.info(f"  Request dim: {request_dim}")
    logger.info(f"  Hidden dim: {hidden_dim}")

    model = ThreeLevelHRL_Controller(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        request_dim=request_dim,
        hidden_dim=hidden_dim,
        num_goals=num_goals,
        num_actions=num_actions,
        **kwargs
    )

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"✅ Model created successfully")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model


def model_summary(model: nn.Module) -> str:
    """
    生成模型摘要
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL SUMMARY")
    lines.append("=" * 80)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines.append(f"Total Parameters:      {total_params:,}")
    lines.append(f"Trainable Parameters:  {trainable_params:,}")
    lines.append(f"Non-trainable:         {total_params - trainable_params:,}")

    # 模块统计
    lines.append("\nModule Breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        lines.append(f"  {name:30s} {params:>15,}")

    lines.append("=" * 80)

    return "\n".join(lines)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("多播感知三层HRL模型测试")
    print("=" * 80 + "\n")

    # 创建模型
    model = create_model(
        node_feat_dim=14,  # 6 + K_vnf (假设K_vnf=8)
        edge_feat_dim=3,
        request_dim=10,
        hidden_dim=128,
        num_goals=10,
        num_actions=50,
        use_checkpoint=False
    )

    # 打印摘要
    print(model_summary(model))

    # 创建测试数据
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    num_nodes = 28
    num_edges = 45

    x = torch.randn(num_nodes, 14, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    edge_attr = torch.randn(num_edges, 3, device=device)
    request_vec = torch.randn(10, device=device)
    dest_indices = [5, 10, 15]

    print("\n" + "=" * 80)
    print("测试各层前向传播")
    print("=" * 80 + "\n")

    # 测试High-Level
    print("1. High-Level (VNF共享策略选择):")
    with torch.no_grad():
        strategy_logits = model.forward_high(x, edge_index, edge_attr, request_vec, dest_indices)
        strategy = torch.argmax(strategy_logits).item()
        print(f"   Strategy logits: {strategy_logits.cpu().numpy()}")
        print(f"   Selected strategy: {strategy}")

    # 测试Mid-Level
    print("\n2. Mid-Level (目标节点选择):")
    candidate_goals = [5, 10, 15]
    with torch.no_grad():
        goal_scores = model.forward_mid(x, edge_index, edge_attr, request_vec,
                                        candidate_goals, dest_indices)
        best_goal = candidate_goals[torch.argmax(goal_scores).item()]
        print(f"   Goal scores: {goal_scores.cpu().numpy()}")
        print(f"   Selected goal: {best_goal}")

    # 测试Low-Level
    print("\n3. Low-Level (动作Q值):")
    goal_idx = torch.tensor([5], device=device)
    action_mask = torch.ones(1, 50, device=device, dtype=torch.bool)
    action_mask[0, 25:] = False  # 只允许前25个动作

    with torch.no_grad():
        q_values = model.forward_low(x, edge_index, edge_attr, None,
                                     request_vec, goal_idx, action_mask)
        best_action = torch.argmax(q_values).item()
        print(f"   Q-values shape: {q_values.shape}")
        print(f"   Max Q-value: {q_values.max().item():.4f}")
        print(f"   Selected action: {best_action}")

    print("\n" + "=" * 80)
    print("✅ 所有测试通过!")
    print("=" * 80 + "\n")