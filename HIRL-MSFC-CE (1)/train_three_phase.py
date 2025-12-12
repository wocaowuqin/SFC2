#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_three_phase.py - 修复版
修复问题: 阶段3训练数据耗尽导致立即退出

关键修改:
1. 在阶段3开始时重新初始化环境以重置请求队列
2. 添加详细的调试日志
3. 添加数据耗尽的错误处理

使用方法:
    python train_three_phase_FIXED.py
"""
import os
import sys
import logging
import random
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib

# 强制使用非交互式后端
os.environ['MPLBACKEND'] = 'Agg'
matplotlib.use('Agg')

# 导入项目模块
import hyperparameters as H
from hirl_sfc_env_gnn import SFC_HIRL_Env_GNN
from hirl_gnn_models import GNN_HRL_Controller
from hirl_sfc_agent_gnn import Agent_SFC_GNN
from expert_data_collector import ExpertDataCollector

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            H.OUTPUT_DIR / "training_three_phase.log",
            mode='w', encoding='utf-8'
        ) if H.OUTPUT_DIR.exists() else logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# 三阶段训练器 - 修复版
# ============================================================================
class ThreePhaseTrainer:
    """三阶段训练管理器"""

    def __init__(self, config: dict):
        """
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1. 初始化环境
        logger.info("初始化环境...")
        self.env = SFC_HIRL_Env_GNN(
            input_dir=config['input_dir'],
            topo=config['topology'],
            dc_nodes=config['dc_nodes'],
            capacities=config['capacities'],
            use_gnn=True
        )

        # 获取状态维度 (用于初始化模型)
        test_req, test_state = self.env.reset_request()
        if test_req is None:
            raise RuntimeError("No requests available in the dataset!")

        x, edge_index, edge_attr, req_vec = test_state

        # 2. 初始化模型
        logger.info("创建GNN模型...")
        self.model = self._create_model(
            node_feat_dim=x.shape[1],
            edge_feat_dim=edge_attr.shape[1],
            request_dim=len(req_vec),
            hidden_dim=config['hidden_dim'],
            num_actions=self.env.NB_LOW_LEVEL_ACTIONS
        )

        # 3. 初始化Agent
        logger.info("创建 Agent...")
        self.agent = Agent_SFC_GNN(
            model=self.model,
            n_actions=self.env.NB_LOW_LEVEL_ACTIONS,
            lr=config['lr'],
            gamma=config['gamma'],
            device=self.device,
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=100000
        )

        # 输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 训练统计容器
        self.training_stats = {
            'phase1': {},
            'phase2': {},
            'phase3': {}
        }

    def _create_model(self, node_feat_dim, edge_feat_dim, request_dim,
                      hidden_dim, num_actions):
        """智能创建模型 - 自动适配参数"""
        import inspect
        sig = inspect.signature(GNN_HRL_Controller.__init__)
        valid_params = set(sig.parameters.keys()) - {'self'}

        model_kwargs = {
            'node_feat_dim': node_feat_dim,
            'edge_feat_dim': edge_feat_dim,
            'request_dim': request_dim,
            'hidden_dim': hidden_dim,
            'num_actions': num_actions
        }

        # 添加可选参数
        if 'use_cache' in valid_params:
            model_kwargs['use_cache'] = False
        if 'use_checkpoint' in valid_params:
            model_kwargs['use_checkpoint'] = True

        logger.info(f"模型初始化参数: {list(model_kwargs.keys())}")
        return GNN_HRL_Controller(**model_kwargs)

    def run_phase1_collect_expert_data(self) -> str:
        """
        [阶段1] 收集专家演示数据
        Returns: expert_data_path
        """
        logger.info("\n" + "=" * 80)
        logger.info("阶段1: 收集专家演示数据 (Expert Data Collection)")
        logger.info("=" * 80)

        expert_data_path = self.output_dir / "expert_data" / "expert_data_final.pkl"

        # 如果数据已存在，询问是否跳过
        if expert_data_path.exists():
            logger.info(f"发现已存在的专家数据: {expert_data_path}")
            return str(expert_data_path)

        # 创建数据收集器
        collector = ExpertDataCollector(
            env=self.env,
            output_dir=self.output_dir / "expert_data"
        )

        # 开始收集
        expert_buffer = collector.collect(
            num_episodes=self.config['phase1_episodes'],
            min_confidence=self.config['min_expert_confidence'],
            save_interval=100
        )

        self.training_stats['phase1'] = collector.stats
        logger.info(f"✅ 阶段1完成: 收集了 {len(expert_buffer)} 条数据")
        return str(expert_data_path)

    def run_phase2_imitation_learning(self, expert_data_path: str):
        """
        [阶段2] 模仿学习预训练 (Behavior Cloning)
        """
        logger.info("\n" + "=" * 80)
        logger.info("阶段2: 模仿学习预训练 (Imitation Learning)")
        logger.info("=" * 80)

        # 加载数据
        expert_buffer, _ = ExpertDataCollector.load_expert_data(Path(expert_data_path))

        # 划分数据集
        random.shuffle(expert_buffer)
        split_idx = int(len(expert_buffer) * 0.9)
        train_data = expert_buffer[:split_idx]
        val_data = expert_buffer[split_idx:]

        logger.info(f"训练集大小: {len(train_data)} | 验证集大小: {len(val_data)}")

        # 切换模式
        self.agent.switch_to_imitation_mode()

        num_epochs = self.config['phase2_epochs']
        batch_size = self.config['batch_size']
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(num_epochs):
            # --- 训练 ---
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

            # --- 验证 ---
            if epoch % 5 == 0 or epoch == num_epochs - 1:
                val_metrics = self.agent.evaluate_imitation(val_data, num_samples=500)
                val_acc = val_metrics['accuracy']

                logger.info(f"Epoch {epoch + 1}/{num_epochs} | "
                            f"Train Loss: {avg_loss:.4f} Acc: {avg_acc:.2f}% | "
                            f"Val Acc: {val_acc:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    self.agent.save(str(self.output_dir / "best_imitation_model.pth"))
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"早停: 验证集准确率 {patience} 次未提升")
                    break

        # 恢复最佳权重
        if best_val_acc > 0:
            self.agent.load(str(self.output_dir / "best_imitation_model.pth"))
            logger.info(f"✅ 阶段2完成. 加载最佳模型 (Val Acc: {best_val_acc:.2f}%)")

        self.training_stats['phase2'] = {'best_val_acc': best_val_acc}

    def run_phase3_rl_finetuning(self):
        """
        [阶段3] 强化学习微调 (RL Fine-tuning) - 修复版
        """
        logger.info("\n" + "=" * 80)
        logger.info("阶段3: 强化学习微调 (RL Fine-tuning)")
        logger.info("=" * 80)

        # ✅ 关键修复: 重新初始化环境以重置请求队列
        logger.info("🔧 重新初始化环境以加载新数据...")
        self.env = SFC_HIRL_Env_GNN(
            input_dir=self.config['input_dir'],
            topo=self.config['topology'],
            dc_nodes=self.config['dc_nodes'],
            capacities=self.config['capacities'],
            use_gnn=True
        )

        # 检查可用请求数
        if hasattr(self.env, 'requests'):
            logger.info(f"✅ 环境已重置，可用请求数: {len(self.env.requests)}")
        elif hasattr(self.env, 'data_loader'):
            logger.info(f"✅ 环境已重置，使用数据加载器")
        else:
            logger.warning("⚠️  无法确认请求数量")

        # 切换回 RL 模式
        self.agent.switch_to_rl_mode(start_epsilon=self.config['phase3_start_epsilon'])
        logger.info(f"切换到RL模式: epsilon={self.config['phase3_start_epsilon']}")

        num_episodes = self.config['phase3_episodes']
        expert_ratio = self.config['phase3_expert_ratio']

        logger.info(f"开始RL训练: {num_episodes} episodes")
        logger.info(f"专家混合比例: {expert_ratio}")

        ep_rewards = []
        acceptance_rates = []

        ep_count = 0
        failed_reset_count = 0  # 追踪连续失败次数

        while ep_count < num_episodes:
            req, state = self.env.reset_request()

            # ✅ 添加调试信息
            if req is None:
                failed_reset_count += 1
                if failed_reset_count == 1:
                    logger.error(f"❌ Episode {ep_count}: reset_request() 返回 None - 数据已耗尽")
                    logger.error(f"   已完成 episodes: {ep_count}/{num_episodes}")
                    logger.error(f"   提示: 检查数据文件是否包含足够的请求")
                if failed_reset_count >= 10:
                    logger.error(f"❌ 连续10次无法获取请求，终止训练")
                    break
                continue

            failed_reset_count = 0  # 重置计数器

            ep_reward = 0.0
            done = False
            step = 0
            max_steps = 100

            while not done and step < max_steps:
                if not self.env.unadded_dest_indices:
                    break

                # --- 目标选择 ---
                valid_goals = list(self.env.unadded_dest_indices)

                if random.random() < expert_ratio:
                    flat_state = self.env._get_flat_state()
                    candidates = self.env.get_expert_high_level_candidates(flat_state)
                    goal = candidates[0][0] if candidates else random.choice(valid_goals)
                else:
                    goal = random.choice(valid_goals)

                # --- 动作选择 ---
                valid_actions = self.env.get_valid_low_level_actions()
                if not valid_actions:
                    break

                expert_action = None
                if random.random() < expert_ratio:
                    try:
                        expert_action = self.env.expert_low_level_action(goal)
                    except:
                        pass

                action = self.agent.select_action(
                    state, goal, valid_actions,
                    expert_action=expert_action,
                    beta=expert_ratio
                )

                # --- 执行 ---
                next_state, reward, sub_done, req_done = self.env.step_low_level(goal, action)

                # --- 存储与更新 ---
                next_valid = self.env.get_valid_low_level_actions() if not req_done else None

                try:
                    self.agent.store(state, action, reward, next_state, req_done, goal, next_valid)
                except TypeError:
                    self.agent.store(state, action, reward, next_state, req_done, goal)

                self.agent.update()

                state = next_state
                ep_reward += reward
                step += 1

                if req_done:
                    done = True

            # 记录与日志
            ep_count += 1
            ep_rewards.append(ep_reward)

            if ep_count % 10 == 0:
                total = max(1, self.env.total_requests_seen)
                acc = self.env.total_requests_accepted / total * 100
                acceptance_rates.append(acc)
                avg_rew = np.mean(ep_rewards[-10:])
                logger.info(
                    f"Episode {ep_count}/{num_episodes} | "
                    f"Reward: {avg_rew:.2f} | "
                    f"Acc: {acc:.2f}% | "
                    f"Eps: {self.agent.get_epsilon():.3f}")

            if ep_count % 100 == 0:
                self.agent.save(str(self.output_dir / f"rl_model_ep{ep_count}.pth"))
                logger.info(f"💾 Checkpoint saved: rl_model_ep{ep_count}.pth")

        # 保存最终模型
        self.agent.save(str(self.output_dir / "final_model.pth"))

        # 统计
        final_acc = acceptance_rates[-1] if acceptance_rates else 0.0
        avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0

        self.training_stats['phase3'] = {
            'episodes_completed': ep_count,
            'final_acc': final_acc,
            'avg_reward': avg_reward
        }

        logger.info("=" * 80)
        logger.info("阶段3训练完成统计")
        logger.info("=" * 80)
        logger.info(f"完成Episodes: {ep_count}/{num_episodes}")
        logger.info(f"平均Reward: {avg_reward:.2f}")
        logger.info(f"最终成功率: {final_acc:.2f}%")
        logger.info("=" * 80)
        logger.info(f"✅ 阶段3完成. 最终模型已保存.")

    def run(self):
        """执行完整流程"""
        # 1. 收集数据
        data_path = self.run_phase1_collect_expert_data()
        # 2. 模仿学习
        self.run_phase2_imitation_learning(data_path)
        # 3. RL 微调
        self.run_phase3_rl_finetuning()

        logger.info("\n🎉 所有训练阶段完成！")


# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    # 配置
    config = {
        'input_dir': H.INPUT_DIR,
        'topology': H.TOPOLOGY_MATRIX,
        'dc_nodes': H.DC_NODES,
        'capacities': H.CAPACITIES,

        'hidden_dim': 128,
        'lr': 1e-4,
        'gamma': 0.99,
        'buffer_size': 20000,
        'batch_size': 32,

        'phase1_episodes': 500,
        'min_expert_confidence': 0.1,

        'phase2_epochs': 30,

        'phase3_episodes': 1500,
        'phase3_start_epsilon': 0.3,
        'phase3_expert_ratio': 0.2,

        'output_dir': H.OUTPUT_DIR / "three_phase_results"
    }

    trainer = ThreePhaseTrainer(config)
    trainer.run()