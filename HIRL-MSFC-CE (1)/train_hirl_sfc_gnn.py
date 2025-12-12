#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_hirl_sfc_gnn_UNIVERSAL.py
通用版训练脚本 - 自动适配不同的模型参数

特性:
1. ✅ 自动检测并适配模型参数
2. ✅ 修复所有已知Bug
3. ✅ 支持继承版GNN环境
4. ✅ 详细日志输出
"""
from __future__ import annotations
import os
import sys
import logging
import csv
import random
import inspect
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib

# 强制使用非交互式后端
os.environ['MPLBACKEND'] = 'Agg'
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# 导入项目模块
import hyperparameters as H
from hirl_sfc_env_gnn import SFC_HIRL_Env_GNN
from hirl_gnn_models import GNN_HRL_Controller
from hirl_sfc_agent_gnn import Agent_SFC_GNN

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(H.OUTPUT_DIR / "training_universal.log", mode='w', encoding='utf-8')
        if H.OUTPUT_DIR.exists() else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 智能模型初始化
# ==========================================
def create_gnn_model(node_feat_dim, edge_feat_dim, request_dim,
                     hidden_dim, num_actions, max_nodes=None):
    """
    智能创建GNN模型 - 自动适配不同的参数签名
    """
    # 获取模型的__init__参数
    sig = inspect.signature(GNN_HRL_Controller.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}

    logger.info(f"GNN_HRL_Controller 支持的参数: {sorted(valid_params)}")

    # 构建参数字典
    model_kwargs = {
        'node_feat_dim': node_feat_dim,
        'edge_feat_dim': edge_feat_dim,
        'request_dim': request_dim,
        'hidden_dim': hidden_dim,
        'num_actions': num_actions
    }

    # 可选参数
    optional_params = {
        'max_nodes': max_nodes,
        'use_cache': False,
        'use_checkpoint': True,
        'num_goals': None  # 如果需要
    }

    # 只添加模型支持的参数
    for key, value in optional_params.items():
        if key in valid_params and value is not None:
            model_kwargs[key] = value

    logger.info(f"实际使用的参数: {sorted(model_kwargs.keys())}")

    # 创建模型
    try:
        model = GNN_HRL_Controller(**model_kwargs)
        logger.info("✅ 模型创建成功")
        return model
    except Exception as e:
        logger.error(f"❌ 模型创建失败: {e}")
        logger.error(f"尝试的参数: {model_kwargs}")
        raise


# ==========================================
# Meta-Controller 训练器
# ==========================================
class MetaTrainer:
    """用于训练 GNN Meta-Controller 的辅助类"""

    def __init__(self, model, lr=1e-4, batch_size=32):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.memory = deque(maxlen=5000)
        self.batch_size = batch_size
        self.device = next(model.parameters()).device
        self.update_count = 0

    def store(self, state_tuple, expert_goal_idx):
        """存储专家标签"""
        self.memory.append((state_tuple, expert_goal_idx))

    def train(self):
        """训练Meta Controller"""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        loss_val = 0.0
        self.optimizer.zero_grad()

        for (state, target) in batch:
            x, ei, ea, req = state

            # 准备输入
            x_d = x.to(self.device)
            ei_d = ei.to(self.device)
            ea_d = ea.to(self.device) if ea is not None else None

            # 处理req_vec
            if isinstance(req, np.ndarray):
                req_d = torch.from_numpy(req).float().to(self.device).unsqueeze(0)
            else:
                req_d = torch.tensor(req, dtype=torch.float32, device=self.device).unsqueeze(0)

            # 检查模型是否有forward_meta方法
            if not hasattr(self.model, 'forward_meta'):
                # 如果没有，跳过Meta训练
                return 0.0

            # 获取候选目标
            num_goals = getattr(self.model, 'goal_embedding', None)
            if num_goals is None:
                return 0.0

            num_goals = num_goals.num_embeddings
            candidates = list(range(num_goals))

            # 前向传播
            scores = self.model.forward_meta(x_d, ei_d, ea_d, req_d, candidates)

            # 计算Loss
            target_t = torch.tensor([target], dtype=torch.long, device=self.device)
            loss = F.cross_entropy(scores.unsqueeze(0), target_t)
            loss.backward()
            loss_val += loss.item()

        self.optimizer.step()
        self.update_count += 1
        return loss_val / self.batch_size


# ==========================================
# 绘图工具
# ==========================================
def save_tracking_data(tracking_data: dict, out_dir: Path):
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(tracking_data)
        df.to_csv(out_dir / "training_metrics.csv", index=False)

        for col in ['reward', 'acceptance_rate', 'blocking_rate', 'meta_loss', 'low_loss']:
            if col in df.columns and len(df[col].dropna()) > 0:
                fig = Figure(figsize=(10, 6))
                ax = fig.add_subplot(111)
                ax.plot(df[col].values)
                ax.set_title(col)
                ax.set_xlabel('Episode / 10')
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
                FigureCanvas(fig).print_figure(str(out_dir / f"{col}.png"))
    except Exception as e:
        logger.error(f"Plotting failed: {e}")


# ==========================================
# 主训练循环
# ==========================================
def main():
    if not H.INPUT_DIR.exists():
        logger.error(f"Input directory not found: {H.INPUT_DIR}")
        return

    H.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化资源日志
    resource_log_path = H.OUTPUT_DIR / "resource_metrics.csv"
    with open(resource_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "RequestID", "Status", "UnaddedDests", "Links", "VNFs"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting GNN Training on {device}")
    logger.info(f"Output directory: {H.OUTPUT_DIR}")

    # =========================================
    # 1. 初始化环境
    # =========================================
    logger.info("Initializing GNN environment...")

    env = SFC_HIRL_Env_GNN(
        input_dir=H.INPUT_DIR,
        topo=H.TOPOLOGY_MATRIX,
        dc_nodes=H.DC_NODES,
        capacities=H.CAPACITIES,
        use_gnn=True
    )

    NB_GOALS = env.NB_HIGH_LEVEL_GOALS
    NB_ACTIONS = env.NB_LOW_LEVEL_ACTIONS

    logger.info(f"Environment initialized:")
    logger.info(f"  Nodes: {env.n}")
    logger.info(f"  Links: {env.L}")
    logger.info(f"  VNF Types: {env.K_vnf}")
    logger.info(f"  High-level Goals: {NB_GOALS}")
    logger.info(f"  Low-level Actions: {NB_ACTIONS}")

    # 获取状态维度
    test_req, test_state = env.reset_request()
    if test_req is None:
        logger.error("No requests available!")
        return

    x, edge_index, edge_attr, req_vec = test_state
    logger.info(f"State dimensions:")
    logger.info(f"  Node features: {x.shape}")
    logger.info(f"  Edges: {edge_index.shape}")
    logger.info(f"  Edge features: {edge_attr.shape}")
    logger.info(f"  Request: {req_vec.shape}")

    # =========================================
    # 2. 智能创建模型
    # =========================================
    logger.info("Creating GNN model (auto-detecting parameters)...")

    model = create_gnn_model(
        node_feat_dim=x.shape[1],
        edge_feat_dim=edge_attr.shape[1],
        request_dim=len(req_vec),
        hidden_dim=128,
        num_actions=NB_ACTIONS,
        max_nodes=env.n  # 会自动检测是否支持
    )

    # =========================================
    # 3. 初始化 Agents
    # =========================================
    logger.info("Creating agents...")
    low_level_agent = Agent_SFC_GNN(
        model=model,
        n_actions=NB_ACTIONS,
        lr=H.LR,
        gamma=0.99,
        device=device,
        buffer_size=H.EXP_MEMORY,
        batch_size=H.BATCH_SIZE,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=100000
    )

    meta_trainer = MetaTrainer(model, lr=H.LR)

    # =========================================
    # 训练配置
    # =========================================
    EXPERT_RATIO = 0.3
    TRAIN_META_FREQ = 10
    UPDATE_FREQ = 4
    LOG_FREQ = 10
    SAVE_FREQ = 100

    logger.info(f"Training config:")
    logger.info(f"  Expert ratio: {EXPERT_RATIO}")
    logger.info(f"  Episodes: {H.EPISODE_LIMIT}")

    # 跟踪数据
    tracking = {
        'episode': [], 'reward': [], 'steps': [],
        'acceptance_rate': [], 'blocking_rate': [],
        'meta_loss': [], 'low_loss': [],
        'epsilon': [], 'unadded_count': []
    }

    ep_count = 0
    total_steps = 0

    # =========================================
    # 开始训练
    # =========================================
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    while ep_count < H.EPISODE_LIMIT:
        current_request, state = env.reset_request()
        if current_request is None:
            logger.info("No more requests")
            break

        if not isinstance(state, tuple):
            state = env.get_state()

        ep_reward = 0.0
        step_count = 0
        ep_losses = []
        request_done = False
        req_id = current_request.get('id', 'unknown')

        # Episode循环
        while len(env.unadded_dest_indices) > 0:
            # 选择目标
            valid_goals = list(env.unadded_dest_indices)

            if random.random() < EXPERT_RATIO:
                flat_state = env._get_flat_state()
                candidates = env.get_expert_high_level_candidates(flat_state, top_k=5)

                if candidates:
                    goal = candidates[0][0]
                    meta_trainer.store(state, goal)
                else:
                    goal = random.choice(valid_goals)
            else:
                goal = random.choice(valid_goals)

            # 选择动作
            valid_actions = env.get_valid_low_level_actions()
            if not valid_actions:
                logger.warning(f"Ep {ep_count}: No valid actions!")
                break

            action = low_level_agent.select_action(state, goal, valid_actions)

            # 执行
            next_state, reward, sub_done, request_done = env.step_low_level(goal, action)

            # 获取下一状态的有效动作
            if not request_done:
                next_valid_actions = env.get_valid_low_level_actions()
            else:
                next_valid_actions = None

            # 存储 (自动适配Agent版本)
            try:
                low_level_agent.store(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=request_done,
                    goal=goal, next_valid_actions=next_valid_actions
                )
            except TypeError:
                low_level_agent.store(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=request_done, goal=goal
                )

            # 更新
            if step_count % UPDATE_FREQ == 0 and low_level_agent.memory.size() >= low_level_agent.batch_size:
                loss = low_level_agent.update()
                if loss > 0:
                    ep_losses.append(loss)

            state = next_state
            ep_reward += reward
            step_count += 1
            total_steps += 1

            if request_done:
                break

        # Episode结束
        ep_count += 1

        # 判断状态
        if request_done:
            status = "Accepted" if len(env.unadded_dest_indices) == 0 else "Partial"
        else:
            status = "Blocked"

        # 记录
        with open(resource_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            tree_links = int(np.sum(env.current_tree['tree'] > 0)) if env.current_tree else 0
            vnf_cnt = len(current_request.get('vnf', []))
            unadded = len(env.unadded_dest_indices)
            writer.writerow([ep_count, req_id, status, unadded, tree_links, vnf_cnt])

        # 训练Meta
        meta_loss = 0.0
        if ep_count % TRAIN_META_FREQ == 0:
            meta_loss = meta_trainer.train()

        # 统计和日志
        if ep_count % LOG_FREQ == 0:
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0

            total_req = max(1, env.total_requests_seen)
            total_dest = max(1, env.total_dest_seen)
            acc_rate = (env.total_requests_accepted / total_req) * 100.0
            dest_acc_rate = (env.total_dest_accepted / total_dest) * 100.0
            epsilon = low_level_agent.get_epsilon()

            tracking['episode'].append(ep_count)
            tracking['reward'].append(ep_reward)
            tracking['steps'].append(step_count)
            tracking['acceptance_rate'].append(acc_rate)
            tracking['blocking_rate'].append(100 - acc_rate)
            tracking['meta_loss'].append(meta_loss)
            tracking['low_loss'].append(avg_loss)
            tracking['epsilon'].append(epsilon)
            tracking['unadded_count'].append(len(env.unadded_dest_indices))

            logger.info(f"\n{'=' * 60}")
            logger.info(f"Episode {ep_count}/{H.EPISODE_LIMIT}")
            logger.info(f"{'=' * 60}")
            logger.info(f"  Request ID:      {req_id}")
            logger.info(f"  Status:          {status}")
            logger.info(f"  Reward:          {ep_reward:.2f}")
            logger.info(f"  Steps:           {step_count}")
            logger.info(f"  Unadded Dests:   {len(env.unadded_dest_indices)}")
            logger.info(f"  Avg Loss:        {avg_loss:.4f}")
            logger.info(f"  Meta Loss:       {meta_loss:.4f}")
            logger.info(f"  Epsilon:         {epsilon:.4f}")
            logger.info(f"Statistics:")
            logger.info(f"  Request Acc:     {acc_rate:.2f}% ({env.total_requests_accepted}/{env.total_requests_seen})")
            logger.info(f"  Dest Acc:        {dest_acc_rate:.2f}% ({env.total_dest_accepted}/{env.total_dest_seen})")

        # 保存
        if ep_count % SAVE_FREQ == 0:
            low_level_agent.save(str(H.OUTPUT_DIR / f"gnn_agent_ep{ep_count}.pth"))
            torch.save({
                'model': model.state_dict(),
                'meta_optimizer': meta_trainer.optimizer.state_dict(),
                'update_count': meta_trainer.update_count
            }, str(H.OUTPUT_DIR / f"meta_ep{ep_count}.pth"))

            save_tracking_data(tracking, H.OUTPUT_DIR)

            logger.info("\n" + "=" * 60)
            env.print_env_summary()
            logger.info("=" * 60 + "\n")

    # 训练完成
    logger.info("\n" + "=" * 60)
    logger.info("✅ Training Finished!")
    logger.info("=" * 60)

    low_level_agent.save(str(H.OUTPUT_DIR / "gnn_agent_final.pth"))
    torch.save({
        'model': model.state_dict(),
        'meta_optimizer': meta_trainer.optimizer.state_dict(),
        'update_count': meta_trainer.update_count
    }, str(H.OUTPUT_DIR / "meta_final.pth"))

    env.print_env_summary()

    logger.info(f"\nFinal Statistics:")
    logger.info(f"  Total Episodes:  {ep_count}")
    logger.info(f"  Total Steps:     {total_steps}")

    save_tracking_data(tracking, H.OUTPUT_DIR)
    logger.info(f"\nOutput saved to: {H.OUTPUT_DIR}")


if __name__ == "__main__":
    main()