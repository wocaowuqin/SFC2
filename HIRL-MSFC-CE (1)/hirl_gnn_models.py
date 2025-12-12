# hirl_gnn_models.py
from __future__ import annotations  # [Fix 2] 解决类型注解的前向引用问题
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv, global_mean_pool
import hashlib
import logging
import threading
import time
from typing import Optional, List, Union, Tuple  # 类型提示

# 配置 Logger
logger = logging.getLogger(__name__)


class MulticastAwareGNN(nn.Module):
    """
    GNN Encoder: 提取网络状态的图嵌入
    生产级特性:
    - 细粒度梯度检查点 (只针对 GAT 层)
    - 分离 GAT 和 Projection 层
    """

    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim=128, num_layers=3,
                 heads=4, dropout=0.1, normalization='pre', use_checkpoint=False):
        super(MulticastAwareGNN, self).__init__()

        # 参数验证
        if heads <= 0: raise ValueError(f"heads must be positive, got {heads}")
        if hidden_dim < heads:
            logger.warning(f"hidden_dim ({hidden_dim}) < heads ({heads}). Adjusting heads to {hidden_dim}.")
            heads = hidden_dim

        self.heads = heads
        self.head_dim = hidden_dim // heads
        if self.head_dim == 0: self.head_dim = 1
        self.hidden_dim = hidden_dim

        self.normalization = normalization
        self.dropout_layer = nn.Dropout(dropout)
        self.use_checkpoint = use_checkpoint

        # 初始特征映射
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_feat_dim, hidden_dim)

        self.gat_convs = nn.ModuleList()
        self.projections = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for _ in range(num_layers):
            gat = GATConv(hidden_dim, self.head_dim, heads=self.heads,
                          edge_dim=hidden_dim, concat=True)
            self.gat_convs.append(gat)

            actual_output_dim = self.head_dim * self.heads
            if actual_output_dim != hidden_dim:
                self.projections.append(nn.Linear(actual_output_dim, hidden_dim))
            else:
                self.projections.append(nn.Identity())

            self.norm_layers.append(nn.LayerNorm(hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor], batch: Optional[torch.Tensor] = None):
        # 处理edge_attr为None的情况
        if edge_attr is None:
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(num_edges, self.edge_embedding.in_features,
                                    device=x.device, dtype=x.dtype)

        # 维度检查
        if edge_attr.dim() != 2 or edge_attr.shape[1] != self.edge_embedding.in_features:
            raise AssertionError(
                f"Edge attr dim mismatch. Expected {self.edge_embedding.in_features}, got {edge_attr.shape[1]}")

        x = self.node_embedding(x)
        e = self.edge_embedding(edge_attr)

        for i, gat in enumerate(self.gat_convs):
            residual = x

            norm = self.norm_layers[i]
            proj = self.projections[i]

            if self.normalization == 'pre':
                x_in = norm(x)
                if self.use_checkpoint and self.training:
                    # use_reentrant=True 兼容性更好
                    out = checkpoint(gat, x_in, edge_index, e, use_reentrant=True)
                else:
                    out = gat(x_in, edge_index, e)

                out = proj(out)
                out = F.relu(out)
                out = self.dropout_layer(out)
                x = residual + out
            else:
                if self.use_checkpoint and self.training:
                    out = checkpoint(gat, x, edge_index, e, use_reentrant=True)
                else:
                    out = gat(x, edge_index, e)

                out = proj(out)
                out = F.relu(out)
                out = self.dropout_layer(out)
                x = residual + out
                x = norm(x)

        if batch is None:
            graph_emb = torch.mean(x, dim=0, keepdim=True)
        else:
            graph_emb = global_mean_pool(x, batch)

        return x, graph_emb


class GNN_HRL_Controller(nn.Module):
    """
    包含 Meta-Controller (高层) 和 Low-Level Agent (低层) 的完整模型
    生产级特性:
    - 修正初始化参数验证顺序
    - 基于内存地址的高效缓存键生成
    - 增强的文档和类型注解
    - 完整的元数据序列化
    - [Fix] 支持 pickle/deepcopy (解决 _thread.lock 错误)
    """

    def __init__(self, node_feat_dim, edge_feat_dim, request_dim,
                 hidden_dim=128, num_goals=10, num_actions=100,
                 use_cache=False, use_checkpoint=False, max_cache_size=2000):
        super(GNN_HRL_Controller, self).__init__()

        # 初始化前的参数验证
        self._validate_init_params(node_feat_dim, edge_feat_dim, request_dim,
                                   hidden_dim, num_goals, num_actions,
                                   use_cache, use_checkpoint, max_cache_size)

        self.request_dim = request_dim
        self.max_cache_size = max_cache_size

        self.gnn = MulticastAwareGNN(node_feat_dim, edge_feat_dim, hidden_dim,
                                     use_checkpoint=use_checkpoint)

        # 缓存配置
        self.use_cache = use_cache
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_lock = threading.Lock()

        # Meta-Controller
        self.meta_context = nn.Sequential(
            nn.Linear(hidden_dim + request_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.meta_scorer = nn.Sequential(
            nn.Linear(128 + hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Low-Level Agent
        self.goal_embedding = nn.Embedding(num_goals, 64)
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim + request_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

        self._init_weights()
        self.validate_config()

    # =========================================================================
    # [关键修复] 添加 pickle 支持，忽略线程锁
    # =========================================================================
    def __getstate__(self):
        """序列化时移除锁对象"""
        state = self.__dict__.copy()
        if 'cache_lock' in state:
            del state['cache_lock']
        return state

    def __setstate__(self, state):
        """反序列化时恢复锁对象"""
        self.__dict__.update(state)
        self.cache_lock = threading.Lock()

    # =========================================================================

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def _validate_init_params(self, node_feat_dim, edge_feat_dim, request_dim,
                              hidden_dim, num_goals, num_actions, use_cache,
                              use_checkpoint, max_cache_size):
        """初始化前的参数验证"""
        issues = []
        if node_feat_dim <= 0: issues.append("node_feat_dim must be positive")
        if edge_feat_dim <= 0: issues.append("edge_feat_dim must be positive")
        if request_dim <= 0: issues.append("request_dim must be positive")
        if hidden_dim <= 0: issues.append("hidden_dim must be positive")
        if num_goals <= 0: issues.append("num_goals must be positive")
        if num_actions <= 0: issues.append("num_actions must be positive")

        # 使用传入的参数进行检查
        if use_cache and max_cache_size <= 0:
            issues.append("max_cache_size must be positive")

        if issues:
            raise ValueError(f"Invalid init params: {', '.join(issues)}")

    def validate_config(self):
        """二次验证：确保所有层创建成功"""
        issues = []

        if not hasattr(self, 'gnn') or self.gnn.node_embedding.in_features <= 0:
            issues.append("GNN not properly initialized")
        if not hasattr(self, 'goal_embedding') or self.goal_embedding.num_embeddings <= 0:
            issues.append("Goal embedding not properly initialized")
        if not hasattr(self, 'q_net') or self.q_net[-1].out_features <= 0:
            issues.append("Q-net not properly initialized")

        if self.use_cache and not hasattr(self, 'max_cache_size'):
            issues.append("Cache enabled but max_cache_size not set")

        if issues:
            raise ValueError(f"Model config issues: {', '.join(issues)}")

        return True

    def get_config(self):
        return {
            'node_feat_dim': self.gnn.node_embedding.in_features,
            'edge_feat_dim': self.gnn.edge_embedding.in_features,
            'request_dim': self.request_dim,
            'hidden_dim': self.gnn.hidden_dim,
            'num_goals': self.goal_embedding.num_embeddings,
            'num_actions': self.q_net[-1].out_features,
            'use_cache': self.use_cache,
            'use_checkpoint': self.gnn.use_checkpoint,
            'max_cache_size': self.max_cache_size,
        }

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state = {k: v for k, v in state.items() if 'cache' not in k}
        state['_metadata'] = {
            'version': '5.2',
            'config': self.get_config(),
            'timestamp': time.time(),
        }
        return state

    def load_state_dict(self, state_dict, strict=True):
        if '_metadata' in state_dict:
            metadata = state_dict.pop('_metadata')
            logger.info(f"Loading model version {metadata.get('version', 'unknown')}")

        result = super().load_state_dict(state_dict, strict)
        self.clear_cache()
        return result

    def train(self, mode=True):
        super().train(mode)
        if mode: self.clear_cache()

    def clear_cache(self):
        with self.cache_lock:
            self.cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0

    def get_cache_stats(self):
        with self.cache_lock:
            total = self.cache_hits + self.cache_misses
            rate = self.cache_hits / total if total > 0 else 0.0
            return {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'total': total,
                'rate': rate,
                'size': len(self.cache)
            }

    def _manage_cache_size(self):
        if len(self.cache) > self.max_cache_size:
            with self.cache_lock:
                if len(self.cache) > self.max_cache_size:
                    keys = list(self.cache.keys())
                    num_to_remove = max(1, int(len(keys) * 0.2))
                    for k in keys[:num_to_remove]:
                        self.cache.pop(k, None)

    def _create_cache_key(self, x, edge_index, edge_attr, batch):
        """
        创建缓存键 - 使用内存地址和元数据
        """
        if not self.use_cache or self.training: return None
        try:
            with torch.no_grad():
                # 使用 shape, stride 和 data_ptr 组合作为指纹
                parts = [
                    f"x_{x.shape}_{x.stride()}_{x.data_ptr()}",
                    f"ei_{edge_index.shape}_{edge_index.stride()}_{edge_index.data_ptr()}"
                ]

                if edge_attr is not None:
                    parts.append(f"ea_{edge_attr.shape}_{edge_attr.stride()}_{edge_attr.data_ptr()}")

                if batch is not None:
                    parts.append(f"b_{batch.shape}_{batch.data_ptr()}")

                key_raw = "|".join(parts)
                return hashlib.md5(key_raw.encode("utf-8")).hexdigest()
        except Exception as e:
            logger.warning(f"Cache key generation failed: {e}")
            return None

    def _get_huge_negative(self, tensor):
        if torch.is_floating_point(tensor):
            if tensor.dtype == torch.float32:
                return torch.full([], -1e9, device=tensor.device, dtype=tensor.dtype)
            elif tensor.dtype == torch.float64:
                return torch.full([], -1e18, device=tensor.device, dtype=tensor.dtype)
            elif tensor.dtype == torch.float16:
                return torch.full([], -6.5e4, device=tensor.device, dtype=tensor.dtype)
            else:
                finfo = torch.finfo(tensor.dtype)
                safe_min = float(finfo.min) / 2.0
                return torch.full([], safe_min, device=tensor.device, dtype=tensor.dtype)
        else:
            iinfo = torch.iinfo(tensor.dtype)
            return torch.full([], iinfo.min, device=tensor.device, dtype=tensor.dtype)

    def forward_meta(self,
                     x: torch.Tensor,
                     edge_index: torch.Tensor,
                     edge_attr: Optional[torch.Tensor],
                     req_vecs: torch.Tensor,
                     candidate_indices: Union[List[int], List[List[int]], torch.Tensor],
                     batch: Optional[torch.Tensor] = None,
                     return_all_scores: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        高层前向传播：为候选节点打分
        """

        # 缓存逻辑
        use_cache = self.use_cache and not self.training
        cache_key = self._create_cache_key(x, edge_index, edge_attr, batch) if use_cache else None

        node_embs, graph_embs = None, None

        if cache_key:
            with self.cache_lock:
                if cache_key in self.cache:
                    self.cache_hits += 1
                    node_embs, graph_embs = self.cache[cache_key]
                    current_device = x.device
                    if node_embs.device != current_device:
                        node_embs = node_embs.to(current_device)
                        graph_embs = graph_embs.to(current_device)
                        self.cache[cache_key] = (node_embs, graph_embs)
                else:
                    self.cache_misses += 1

        if node_embs is None:
            node_embs, graph_embs = self.gnn(x, edge_index, edge_attr, batch)
            if cache_key:
                with self.cache_lock:
                    self.cache[cache_key] = (node_embs.detach().clone(), graph_embs.detach().clone())
                self._manage_cache_size()

        if batch is None:
            # === Single Graph ===
            expected_req_dim = self.meta_context[0].in_features - self.gnn.hidden_dim
            if req_vecs.dim() == 1:
                if req_vecs.size(0) != expected_req_dim:
                    raise ValueError(f"req_vecs dim mismatch: expected {expected_req_dim}, got {req_vecs.size(0)}")
                req_vecs = req_vecs.unsqueeze(0)
            elif req_vecs.dim() == 2:
                if req_vecs.size(1) != expected_req_dim:
                    raise ValueError(f"req_vecs dim mismatch: expected {expected_req_dim}, got {req_vecs.size(1)}")

            if not torch.is_tensor(candidate_indices):
                candidate_indices = torch.tensor(candidate_indices, dtype=torch.long, device=x.device)

            if candidate_indices.max() >= node_embs.size(0) or candidate_indices.min() < 0:
                raise ValueError(f"Candidate indices out of range [0, {node_embs.size(0)})")

            global_input = torch.cat([graph_embs, req_vecs], dim=1)
            context = self.meta_context(global_input)

            cand_embs = node_embs[candidate_indices]
            context_expanded = context.expand(cand_embs.size(0), -1)
            combined = torch.cat([context_expanded, cand_embs], dim=1)
            scores = self.meta_scorer(combined).squeeze(-1)
            return scores

        else:
            # === Batch Processing ===
            batch_size = graph_embs.size(0)

            if torch.is_tensor(candidate_indices):
                if candidate_indices.dim() == 2:
                    cands_list = []
                    for i in range(batch_size):
                        row = candidate_indices[i]
                        valid_mask = row >= 0
                        cands_list.append(row[valid_mask])
                    candidate_indices = cands_list
                else:
                    raise ValueError(f"Unsupported candidate_indices tensor shape: {candidate_indices.shape}")
            elif isinstance(candidate_indices, (list, tuple)):
                if len(candidate_indices) != batch_size:
                    raise ValueError(f"Length mismatch: candidates {len(candidate_indices)} != batch {batch_size}")
            else:
                raise TypeError(f"Unsupported candidate_indices type: {type(candidate_indices)}")

            global_input = torch.cat([graph_embs, req_vecs], dim=1)
            context = self.meta_context(global_input)

            scores_list = []

            for i in range(batch_size):
                curr_cands = candidate_indices[i]

                if curr_cands is None or len(curr_cands) == 0:
                    scores_list.append(torch.tensor([], device=x.device))
                    continue

                if not torch.is_tensor(curr_cands):
                    curr_cands = torch.tensor(curr_cands, dtype=torch.long, device=x.device)

                # 边界检查
                if curr_cands.max() >= node_embs.size(0) or curr_cands.min() < 0:
                    raise ValueError(f"Batch {i}: indices {curr_cands} out of range [0, {node_embs.size(0)})")

                cand_feats = node_embs[curr_cands]
                curr_ctx = context[i].unsqueeze(0).expand(cand_feats.size(0), -1)
                combined = torch.cat([curr_ctx, cand_feats], dim=1)
                curr_scores = self.meta_scorer(combined).squeeze(-1)
                scores_list.append(curr_scores)

            return scores_list

    def forward_low(self, x: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: Optional[torch.Tensor], batch_vec: Optional[torch.Tensor],
                    req_vecs: torch.Tensor, goal_indices: torch.Tensor,
                    action_masks: Optional[torch.Tensor] = None):
        """
        低层前向传播：计算 Q 值
        """
        _, graph_embs = self.gnn(x, edge_index, edge_attr, batch_vec)
        goal_embs = self.goal_embedding(goal_indices)

        combined = torch.cat([graph_embs, req_vecs, goal_embs], dim=1)
        q_values = self.q_net(combined)

        if action_masks is not None:
            if action_masks.device != q_values.device:
                action_masks = action_masks.to(q_values.device)
            if action_masks.shape != q_values.shape:
                raise ValueError(f"Mask shape {action_masks.shape} != Q shape {q_values.shape}")

            huge_neg = self._get_huge_negative(q_values)

            if action_masks.dtype == torch.bool:
                valid_mask = action_masks
            elif torch.is_floating_point(action_masks):
                valid_mask = action_masks > 0.5
            elif torch.is_integer(action_masks) or not torch.is_complex(action_masks):
                valid_mask = action_masks > 0
            else:
                raise TypeError(f"Unsupported mask dtype: {action_masks.dtype}")

            q_values = torch.where(valid_mask, q_values, huge_neg)

        return q_values