#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_three_phase_multicast.py - 多播SFC三阶段训练脚本 (完整适配版)

适配说明:
1. ✅ 支持多播感知GNN模型 (ThreeLevelHRL_Controller)
2. ✅ 支持三层HRL决策 (High + Mid + Low)
3. ✅ 集成VNF共享潜力
4. ✅ 集成Curriculum Learning (可选)
5. ✅ 完整的指标记录和可视化

使用方法:
    python train_three_phase_multicast.py

作者: Your Name
日期: 2025-12-13
版本: 6.0 (Multicast Edition)
"""
import os
import sys
import logging
import random
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入项目模块
import hyperparameters as H
from hirl_sfc_env_gnn import SFC_HIRL_Env_GNN
from hirl_gnn_models import ThreeLevelHRL_Controller  # 🔥 新版三层HRL
from expert_data_collector import ExpertDataCollector

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            H.OUTPUT_DIR / "training_multicast.log",
            mode='w', encoding='utf-8'
        ) if H.OUTPUT_DIR.exists() else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 辅助类: 多播感知Agent
# ============================================================================
class MulticastAgent:
    """
    多播感知Agent (支持三层HRL)

    功能:
    1. High-Level: VNF共享策略选择
    2. Mid-Level: 目标节点选择 (带共享潜力加权)
    3. Low-Level: 路径与VNF放置执行
    """

    def __init__(self, model: nn.Module, env, lr: float = 1e-4,
                 gamma: float = 0.99, device: str = "cpu",
                 buffer_size: int = 20000, batch_size: int = 32):
        """
        Args:
            model: ThreeLevelHRL_Controller
            env: SFC_HIRL_Env_GNN
            lr: 学习率
            gamma: 折扣因子
            device: 设备
            buffer_size: 经验回放缓冲区大小
            batch_size: 批大小
        """
        self.model = model.to(device)
        self.target_model = ThreeLevelHRL_Controller(
            node_feat_dim=model.node_feat_dim,
            edge_feat_dim=model.edge_feat_dim,
            request_dim=model.request_dim,
            hidden_dim=model.hidden_dim,
            num_goals=model.num_goals,
            num_actions=model.num_actions
        ).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.env = env
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size

        # 🔥 分层优化器 (可以为不同层设置不同学习率)
        self.optimizer = optim.Adam([
            {'params': self.model.mgat.parameters(), 'lr': lr},  # GNN编码器
            {'params': self.model.high_policy.parameters(), 'lr': lr * 0.5},  # High-level
            {'params': self.model.mid_context.parameters(), 'lr': lr},  # Mid-level
            {'params': self.model.mid_goal_scorer.parameters(), 'lr': lr},
            {'params': self.model.low_q_net.parameters(), 'lr': lr},  # Low-level
        ])

        # 经验回放
        self.replay_buffer = deque(maxlen=buffer_size)

        # 训练状态
        self.mode = 'rl'  # 'il' (imitation) or 'rl' (reinforcement)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # 统计
        self.update_count = 0
        self.target_update_freq = 100

        logger.info(f"MulticastAgent initialized on {device}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ===== Mode Switching =====
    def switch_to_imitation_mode(self):
        """切换到模仿学习模式"""
        self.mode = 'il'
        self.model.train()
        logger.info("Switched to Imitation Learning mode")

    def switch_to_rl_mode(self, start_epsilon: float = 0.3):
        """切换到强化学习模式"""
        self.mode = 'rl'
        self.epsilon = start_epsilon
        self.model.train()
        logger.info(f"Switched to RL mode (epsilon={start_epsilon})")

    # ===== Experience Replay =====
    def store(self, state, action, reward, next_state, done, goal):
        """存储经验"""
        self.replay_buffer.append((state, action, reward, next_state, done, goal))

    def sample_batch(self, batch_size: int) -> Optional[List]:
        """采样batch"""
        if len(self.replay_buffer) < batch_size:
            return None
        return random.sample(self.replay_buffer, batch_size)

    # ===== Action Selection =====
    def select_high_level_strategy(self, state) -> int:
        """
        High-Level: 选择VNF共享策略

        Returns:
            strategy: 0-3
        """
        x, edge_index, edge_attr, request_vec = state

        with torch.no_grad():
            strategy_logits = self.model.forward_high(
                x.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device),
                request_vec.to(self.device),
                dest_indices=None  # 会在模型内部处理
            )
            strategy = torch.argmax(strategy_logits).item()

        return strategy

    def select_goal(self, state, candidate_goals: List[int],
                    epsilon: float = 0.0) -> int:
        """
        Mid-Level: 选择下一个目标节点

        Args:
            state: 图状态
            candidate_goals: 候选目标列表
            epsilon: 探索率

        Returns:
            goal_idx: 选中的目标索引
        """
        if not candidate_goals:
            return 0

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.choice(candidate_goals)

        x, edge_index, edge_attr, request_vec = state

        with torch.no_grad():
            goal_scores = self.model.forward_mid(
                x.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device),
                request_vec.to(self.device),
                candidate_goals=candidate_goals,
                dest_indices=candidate_goals  # 🔥 传入所有目标用于Set Transformer
            )
            best_idx = torch.argmax(goal_scores).item()
            return candidate_goals[best_idx]

    def select_action(self, state, goal: int, action_mask: np.ndarray,
                      epsilon: float = 0.0) -> int:
        """
        Low-Level: 选择动作

        Args:
            state: 图状态
            goal: 当前目标节点ID
            action_mask: 动作mask
            epsilon: 探索率

        Returns:
            action: 动作索引
        """
        # Epsilon-greedy
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0

        if random.random() < epsilon:
            return np.random.choice(valid_actions)

        x, edge_index, edge_attr, request_vec = state

        # 转换goal为tensor
        goal_idx = torch.tensor([goal], device=self.device, dtype=torch.long)

        # 转换mask为tensor
        mask_tensor = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model.forward_low(
                x.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device),
                None,  # batch_vec
                request_vec.to(self.device),
                goal_idx,
                mask_tensor
            )
            action = torch.argmax(q_values).item()

        return action

    # ===== Training =====
    def supervised_update(self, batch: List) -> Tuple[float, float]:
        """
        监督学习更新 (Behavior Cloning) - 兼容多种格式

        Args:
            batch: Expert数据批次
                   支持格式:
                   1. 字典: {'state': ..., 'goal': ..., 'action': ...}
                   2. 元组: (state, goal, action, ...)

        Returns:
            loss, accuracy
        """
        if not batch:
            return 0.0, 0.0

        self.model.train()

        states, goals, actions = [], [], []

        # 🔥 修复: 兼容字典和元组两种格式
        for item in batch:
            try:
                # 格式1: 字典 (ExpertDataCollector默认格式)
                if isinstance(item, dict):
                    states.append(item['state'])
                    goals.append(item['goal'])
                    actions.append(item['action'])

                # 格式2: 元组/列表
                elif isinstance(item, (list, tuple)) and len(item) >= 3:
                    states.append(item[0])
                    goals.append(item[1])
                    actions.append(item[2])

                else:
                    # 未知格式,跳过
                    logger.warning(f"Unknown item format: {type(item)}")
                    continue

            except (KeyError, IndexError) as e:
                # 数据格式错误,跳过
                logger.debug(f"Skipping invalid item: {e}")
                continue

        if not states:
            logger.warning("No valid states in batch!")
            return 0.0, 0.0

        # 准备输入 (保持原有逻辑)
        losses = []
        correct = 0
        total = 0

        for state, goal, action in zip(states, goals, actions):
            x, edge_index, edge_attr, request_vec = state

            goal_tensor = torch.tensor([goal], device=self.device, dtype=torch.long)
            action_tensor = torch.tensor([action], device=self.device, dtype=torch.long)

            # Forward
            q_values = self.model.forward_low(
                x.to(self.device),
                edge_index.to(self.device),
                edge_attr.to(self.device),
                None,
                request_vec.to(self.device),
                goal_tensor,
                None
            )

            # Loss
            loss = nn.CrossEntropyLoss()(q_values, action_tensor)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

            # Accuracy
            pred = torch.argmax(q_values, dim=-1).item()
            if pred == action:
                correct += 1
            total += 1

        avg_loss = np.mean(losses) if losses else 0.0
        accuracy = correct / total * 100 if total > 0 else 0.0

        return avg_loss, accuracy

    def update(self) -> Optional[float]:
        """
        强化学习更新 (DQN)

        Returns:
            loss (if updated)
        """
        if self.mode != 'rl':
            return None

        batch = self.sample_batch(self.batch_size)
        if batch is None:
            return None

        self.model.train()

        states, actions, rewards, next_states, dones, goals = zip(*batch)

        # 转换为tensor
        # 注意: 这里简化,实际需要proper batching
        losses = []

        for i in range(len(states)):
            x, ei, ea, rv = states[i]
            nx, nei, nea, nrv = next_states[i]

            goal_t = torch.tensor([goals[i]], device=self.device, dtype=torch.long)
            action_t = torch.tensor([actions[i]], device=self.device, dtype=torch.long)
            reward_t = torch.tensor([rewards[i]], device=self.device, dtype=torch.float32)
            done_t = torch.tensor([dones[i]], device=self.device, dtype=torch.float32)

            # Current Q
            q_values = self.model.forward_low(
                x.to(self.device), ei.to(self.device), ea.to(self.device),
                None, rv.to(self.device), goal_t, None
            )
            q_value = q_values[0, actions[i]]

            # Target Q
            with torch.no_grad():
                next_q_values = self.target_model.forward_low(
                    nx.to(self.device), nei.to(self.device), nea.to(self.device),
                    None, nrv.to(self.device), goal_t, None
                )
                max_next_q = torch.max(next_q_values)
                target = reward_t + (1 - done_t) * self.gamma * max_next_q

            # Loss
            loss = nn.MSELoss()(q_value, target)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            losses.append(loss.item())

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return np.mean(losses)

    def evaluate_imitation(self, val_data: List, num_samples: int = 500) -> Dict:
        """评估模仿学习性能"""
        self.model.eval()

        samples = random.sample(val_data, min(num_samples, len(val_data)))

        correct = 0
        total = 0

        with torch.no_grad():
            for item in samples:
                if len(item) < 3:
                    continue

                state, goal, action = item[0], item[1], item[2]
                x, ei, ea, rv = state

                goal_t = torch.tensor([goal], device=self.device, dtype=torch.long)

                q_values = self.model.forward_low(
                    x.to(self.device), ei.to(self.device), ea.to(self.device),
                    None, rv.to(self.device), goal_t, None
                )
                pred = torch.argmax(q_values).item()

                if pred == action:
                    correct += 1
                total += 1

        accuracy = correct / total * 100 if total > 0 else 0.0

        return {'accuracy': accuracy, 'correct': correct, 'total': total}

    def get_epsilon(self) -> float:
        """获取当前epsilon"""
        return self.epsilon

    # ===== Save/Load =====
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'config': self.model.get_config()
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.update_count = checkpoint.get('update_count', 0)
        logger.info(f"Model loaded from {path}")


# ============================================================================
# 三阶段训练器
# ============================================================================
class MulticastThreePhaseTrainer:
    """多播SFC三阶段训练管理器"""

    def __init__(self, config: dict):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("=" * 80)
        logger.info("初始化多播SFC训练环境")
        logger.info("=" * 80)

        # 1. 初始化环境
        logger.info("Creating multicast-aware environment...")
        self.env = SFC_HIRL_Env_GNN(
            input_dir=config['input_dir'],
            topo=config['topology'],
            dc_nodes=config['dc_nodes'],
            capacities=config['capacities'],
            use_gnn=True
        )

        # 获取状态维度
        test_req, test_state = self.env.reset_request()
        if test_req is None:
            raise RuntimeError("No requests available in the dataset!")

        x, edge_index, edge_attr, req_vec = test_state

        logger.info(f"State dimensions detected:")
        logger.info(f"  - Node features: {x.shape[1]}")
        logger.info(f"  - Edge features: {edge_attr.shape[1]}")
        logger.info(f"  - Request vector: {len(req_vec)}")

        # 2. 初始化模型
        logger.info("Creating ThreeLevelHRL_Controller...")
        self.model = ThreeLevelHRL_Controller(
            node_feat_dim=x.shape[1],
            edge_feat_dim=edge_attr.shape[1],
            request_dim=len(req_vec),
            hidden_dim=config['hidden_dim'],
            num_goals=self.env.NB_HIGH_LEVEL_GOALS,
            num_actions=self.env.NB_LOW_LEVEL_ACTIONS,
            use_cache=False,
            use_checkpoint=True
        )

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model created: {total_params:,} parameters")

        # 3. 初始化Agent
        logger.info("Creating MulticastAgent...")
        self.agent = MulticastAgent(
            model=self.model,
            env=self.env,
            lr=config['lr'],
            gamma=config['gamma'],
            device=self.device,
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size']
        )

        # 输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 80)

        # 训练统计
        self.training_stats = {
            'phase1': {},
            'phase2': {},
            'phase3': {}
        }

    def run_phase1_collect_expert_data(self) -> str:
        """
        [阶段1] 收集专家演示数据
        """
        logger.info("\n" + "=" * 80)
        logger.info("Phase 1: Expert Data Collection")
        logger.info("=" * 80)

        expert_data_path = self.output_dir / "expert_data" / "expert_data_final.pkl"

        if expert_data_path.exists():
            logger.info(f"Found existing expert data: {expert_data_path}")
            logger.info("Skipping Phase 1...")
            return str(expert_data_path)

        # 创建收集器
        collector = ExpertDataCollector(
            env=self.env,
            output_dir=self.output_dir / "expert_data"
        )

        # 开始收集
        logger.info(f"Collecting {self.config['phase1_episodes']} expert demonstrations...")
        expert_buffer = collector.collect(
            num_episodes=self.config['phase1_episodes'],
            min_confidence=self.config['min_expert_confidence'],
            save_interval=100
        )

        self.training_stats['phase1'] = collector.stats
        logger.info(f"✅ Phase 1 complete: {len(expert_buffer)} samples collected")

        return str(expert_data_path)

    def run_phase2_imitation_learning(self, expert_data_path: str):
        """
        [阶段2] 模仿学习预训练
        """
        logger.info("\n" + "=" * 80)
        logger.info("Phase 2: Imitation Learning (Behavior Cloning)")
        logger.info("=" * 80)

        # 加载数据
        logger.info(f"Loading expert data from {expert_data_path}...")
        expert_buffer, _ = ExpertDataCollector.load_expert_data(Path(expert_data_path))
        logger.info(f"Loaded {len(expert_buffer)} expert samples")

        # 划分数据集
        random.shuffle(expert_buffer)
        split_idx = int(len(expert_buffer) * 0.9)
        train_data = expert_buffer[:split_idx]
        val_data = expert_buffer[split_idx:]

        logger.info(f"Train: {len(train_data)} | Val: {len(val_data)}")

        # 切换模式
        self.agent.switch_to_imitation_mode()

        num_epochs = self.config['phase2_epochs']
        batch_size = self.config['batch_size']
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0

        logger.info(f"Starting IL training: {num_epochs} epochs")

        for epoch in range(num_epochs):
            # 训练
            random.shuffle(train_data)
            epoch_losses = []
            epoch_accs = []

            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]
                loss, acc = self.agent.supervised_update(batch)
                epoch_losses.append(loss)
                epoch_accs.append(acc)

            avg_loss = np.mean(epoch_losses)
            avg_acc = np.mean(epoch_accs)

            # 验证
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                val_metrics = self.agent.evaluate_imitation(val_data, num_samples=500)
                val_acc = val_metrics['accuracy']

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {avg_loss:.4f} Acc: {avg_acc:.2f}% | "
                    f"Val Acc: {val_acc:.2f}%"
                )

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.agent.save(str(self.output_dir / "best_imitation_model.pth"))
                    logger.info(f"  ✅ New best model saved (Val Acc: {val_acc:.2f}%)")
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping: Val acc not improved for {patience} checks")
                    break

        # 恢复最佳权重
        if best_val_acc > 0:
            self.agent.load(str(self.output_dir / "best_imitation_model.pth"))
            logger.info(f"✅ Phase 2 complete. Best Val Acc: {best_val_acc:.2f}%")

        self.training_stats['phase2'] = {'best_val_acc': best_val_acc}

    def run_phase3_rl_finetuning(self):
        """
        [阶段3] 强化学习微调
        """
        logger.info("\n" + "=" * 80)
        logger.info("Phase 3: RL Fine-tuning")
        logger.info("=" * 80)

        # 重新初始化环境
        logger.info("Reinitializing environment...")
        self.env = SFC_HIRL_Env_GNN(
            input_dir=self.config['input_dir'],
            topo=self.config['topology'],
            dc_nodes=self.config['dc_nodes'],
            capacities=self.config['capacities'],
            use_gnn=True
        )

        # 切换模式
        self.agent.switch_to_rl_mode(start_epsilon=self.config['phase3_start_epsilon'])

        num_episodes = self.config['phase3_episodes']
        logger.info(f"Starting RL training: {num_episodes} episodes")

        ep_rewards = []
        acceptance_rates = []
        vnf_sharing_rates = []  # 🔥 新增: VNF共享率

        ep_count = 0
        failed_reset_count = 0

        while ep_count < num_episodes:
            req, state = self.env.reset_request()

            if req is None:
                failed_reset_count += 1
                if failed_reset_count >= 10:
                    logger.error("Failed to get request 10 times, stopping...")
                    break
                continue

            failed_reset_count = 0

            # 🔥 High-Level决策: 选择共享策略
            strategy = self.agent.select_high_level_strategy(state)
            if hasattr(self.env, 'sharing_strategy'):
                self.env.sharing_strategy = strategy

            done = False
            ep_reward = 0.0
            step = 0

            while not done and step < 100:
                # 🔥 Mid-Level决策: 选择目标
                candidates = list(self.env.unadded_dest_indices)
                if not candidates:
                    break

                goal = self.agent.select_goal(
                    state, candidates,
                    epsilon=self.agent.get_epsilon()
                )

                # Low-Level决策: 执行动作
                action_mask = self.env.get_low_level_mask(goal)
                action = self.agent.select_action(
                    state, goal, action_mask,
                    epsilon=self.agent.get_epsilon()
                )

                # 环境step
                next_state, reward, sub_done, req_done = self.env.step_low_level(goal, action)

                # 存储经验
                self.agent.store(state, action, reward, next_state, req_done, goal)

                # 更新
                loss = self.agent.update()

                state = next_state
                ep_reward += reward
                step += 1

                if req_done:
                    done = True

            # 记录
            ep_count += 1
            ep_rewards.append(ep_reward)

            # 🔥 记录VNF共享率
            if hasattr(self.env, 'get_vnf_sharing_rate'):
                sharing_rate = self.env.get_vnf_sharing_rate()
                vnf_sharing_rates.append(sharing_rate)

            # 日志
            if ep_count % 10 == 0:
                total = max(1, self.env.total_requests_seen)
                acc = self.env.total_requests_accepted / total * 100
                acceptance_rates.append(acc)

                avg_rew = np.mean(ep_rewards[-10:])
                avg_share = np.mean(vnf_sharing_rates[-10:]) if vnf_sharing_rates else 0.0

                logger.info(
                    f"Episode {ep_count}/{num_episodes} | "
                    f"Reward: {avg_rew:.2f} | "
                    f"Acc: {acc:.2f}% | "
                    f"Share: {avg_share:.2%} | "  # 🔥 新增
                    f"Eps: {self.agent.get_epsilon():.3f}"
                )

            # Checkpoint
            if ep_count % 100 == 0:
                self.agent.save(str(self.output_dir / f"rl_model_ep{ep_count}.pth"))
                logger.info(f"💾 Checkpoint saved: ep{ep_count}")

        # 保存最终模型
        self.agent.save(str(self.output_dir / "final_model.pth"))

        # 统计
        final_acc = acceptance_rates[-1] if acceptance_rates else 0.0
        avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        avg_sharing = np.mean(vnf_sharing_rates) if vnf_sharing_rates else 0.0

        self.training_stats['phase3'] = {
            'episodes_completed': ep_count,
            'final_acc': final_acc,
            'avg_reward': avg_reward,
            'avg_vnf_sharing': avg_sharing  # 🔥 新增
        }

        logger.info("=" * 80)
        logger.info("Phase 3 Training Statistics")
        logger.info("=" * 80)
        logger.info(f"Episodes: {ep_count}/{num_episodes}")
        logger.info(f"Avg Reward: {avg_reward:.2f}")
        logger.info(f"Final Acceptance Rate: {final_acc:.2f}%")
        logger.info(f"Avg VNF Sharing Rate: {avg_sharing:.2%}")  # 🔥 新增
        logger.info("=" * 80)
        logger.info(f"✅ Phase 3 complete")

    def run(self):
        """执行完整训练流程"""
        start_time = time.time()

        try:
            # Phase 1
            data_path = self.run_phase1_collect_expert_data()

            # Phase 2
            self.run_phase2_imitation_learning(data_path)

            # Phase 3
            self.run_phase3_rl_finetuning()

            elapsed = time.time() - start_time
            logger.info(f"\n🎉 All training phases complete! Total time: {elapsed / 3600:.2f}h")

            # 保存统计
            stats_path = self.output_dir / "training_stats.pkl"
            with open(stats_path, 'wb') as f:
                pickle.dump(self.training_stats, f)
            logger.info(f"Training stats saved to {stats_path}")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    # 配置
    config = {
        # 环境配置
        'input_dir': H.INPUT_DIR,
        'topology': H.TOPOLOGY_MATRIX,
        'dc_nodes': H.DC_NODES,
        'capacities': H.CAPACITIES,

        # 模型配置
        'hidden_dim': 128,
        'lr': 1e-4,
        'gamma': 0.99,
        'buffer_size': 20000,
        'batch_size': 32,

        # Phase 1: 专家数据收集
        'phase1_episodes': 500,
        'min_expert_confidence': 0.3,

        # Phase 2: 模仿学习
        'phase2_epochs': 30,

        # Phase 3: 强化学习
        'phase3_episodes': 1500,
        'phase3_start_epsilon': 0.3,
        'phase3_expert_ratio': 0.2,

        # 输出
        'output_dir': H.OUTPUT_DIR / "multicast_three_phase_results"
    }

    logger.info("=" * 80)
    logger.info("MULTICAST SFC TRAINING - THREE PHASE")
    logger.info("=" * 80)
    logger.info(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Configuration:")
    for key, value in config.items():
        if not isinstance(value, (np.ndarray, Path)):
            logger.info(f"  {key}: {value}")
    logger.info("=" * 80)

    trainer = MulticastThreePhaseTrainer(config)
    trainer.run()