# -*- coding: utf-8 -*-
# @File     : train_hierarchical.py
# @Author   : AI Assistant
# @Desc     : Training script for Hierarchical RL in SFC Mapping (完整修复版)

import sys
import time
import pickle
import logging
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import xml.etree.ElementTree as ET
import networkx as nx

# 导入修复后的模块
from config import Config
from env import SFCEnv
from log import MyLog


# ==================== 分层DQN Agent实现 ====================
class GoalNetwork(nn.Module):
    """Meta-Controller：选择抽象目标（Goal）"""
    def __init__(self, state_dim, goal_dim, hidden_dim=256):
        super(GoalNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, goal_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        goal = torch.tanh(self.fc3(x))  # 限制goal在[-1, 1]
        return goal


class SubPolicyNetwork(nn.Module):
    """Sub-Controller：根据goal选择具体动作"""
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=256):
        super(SubPolicyNetwork, self).__init__()
        # 拼接state和goal
        input_dim = state_dim + goal_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state, goal):
        # 拼接state和goal
        x = torch.cat([state, goal], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class HierarchicalDQNAgent:
    """
    分层DQN代理
    - Meta-Controller：选择目标（goal）
    - Sub-Controller：根据goal选择动作
    """
    def __init__(self, config, num_nodes):
        self.config = config
        self.num_nodes = num_nodes
        self.device = config.DEVICE
        
        # 状态维度（简化为节点数）
        self.state_dim = num_nodes
        self.goal_dim = 32  # goal的维度
        self.action_dim = num_nodes
        
        # Meta-Controller（选择goal）
        self.goal_net = GoalNetwork(self.state_dim, self.goal_dim).to(self.device)
        self.goal_target_net = GoalNetwork(self.state_dim, self.goal_dim).to(self.device)
        self.goal_target_net.load_state_dict(self.goal_net.state_dict())
        self.goal_optimizer = optim.Adam(self.goal_net.parameters(), lr=config.LR)
        
        # Sub-Controller（根据goal选择动作）
        self.sub_net = SubPolicyNetwork(self.state_dim, self.goal_dim, self.action_dim).to(self.device)
        self.sub_target_net = SubPolicyNetwork(self.state_dim, self.goal_dim, self.action_dim).to(self.device)
        self.sub_target_net.load_state_dict(self.sub_net.state_dict())
        self.sub_optimizer = optim.Adam(self.sub_net.parameters(), lr=config.LR)
        
        # 经验回放缓冲
        self.meta_buffer = deque(maxlen=10000)
        self.sub_buffer = deque(maxlen=50000)
        
        # 训练计数
        self.meta_step = 0
        self.sub_step = 0
        
        # 超参数
        self.gamma = config.DISCOUNT
        self.batch_size = config.BATCH_SIZE
        self.tau = config.TAU
        
        self.is_training = True
        
    def extract_state_features(self, state_tensor):
        """
        从SFCEnv的状态张量中提取特征向量
        输入: (1, 7, N, N)
        输出: (state_dim,) 即 (N,)
        """
        # 简化：对每个节点计算平均特征
        # state_tensor: (1, 7, N, N)
        if state_tensor.dim() == 4:
            state_tensor = state_tensor.squeeze(0)  # (7, N, N)
        
        # 提取对角线元素（节点自身特征）
        # 通道0: CPU, 通道3: 部署状态, 通道6: 进度
        cpu_features = state_tensor[0].diag()  # (N,)
        deploy_features = state_tensor[3].diag()
        progress_features = state_tensor[6].diag()
        
        # 组合特征（可以更复杂）
        node_features = (cpu_features + deploy_features + progress_features) / 3.0
        
        return node_features.to(self.device)
    
    def select_goal(self, state, epsilon=0.1):
        """Meta-Controller选择goal"""
        if np.random.rand() < epsilon and self.is_training:
            # 随机探索
            goal = torch.randn(self.goal_dim).to(self.device)
        else:
            # 策略选择
            with torch.no_grad():
                state_features = self.extract_state_features(state).unsqueeze(0)  # (1, N)
                goal = self.goal_net(state_features).squeeze(0)  # (goal_dim,)
        
        return goal
    
    def select_action(self, state, goal, epsilon, valid_actions):
        """Sub-Controller根据goal选择动作"""
        if np.random.rand() < epsilon and self.is_training:
            # 从有效动作中随机选择
            return np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_features = self.extract_state_features(state).unsqueeze(0)  # (1, N)
                goal_tensor = goal.unsqueeze(0)  # (1, goal_dim)
                q_values = self.sub_net(state_features, goal_tensor).squeeze(0)  # (action_dim,)
                
                # 只考虑有效动作
                valid_q_values = {a: q_values[a].item() for a in valid_actions}
                action = max(valid_q_values, key=valid_q_values.get)
        
        return action
    
    def compute_intrinsic_reward(self, goal, action, next_state):
        """
        计算内在奖励（goal达成度）
        使用余弦相似度衡量next_state与goal的接近程度
        """
        with torch.no_grad():
            next_features = self.extract_state_features(next_state)  # (N,)
            
            # 将goal投影到状态空间（简化方法）
            goal_projection = goal[:self.state_dim]  # 取前N维
            
            # 余弦相似度
            similarity = F.cosine_similarity(
                next_features.unsqueeze(0),
                goal_projection.unsqueeze(0),
                dim=1
            ).item()
            
            intrinsic_reward = similarity * 0.5  # 缩放到合理范围
        
        return intrinsic_reward
    
    def push_meta(self, state, goal, reward, next_state, done):
        """存储Meta-level transition"""
        state_features = self.extract_state_features(state)
        next_features = self.extract_state_features(next_state) if next_state is not None else None
        
        self.meta_buffer.append({
            'state': state_features.cpu(),
            'goal': goal.cpu(),
            'reward': reward,
            'next_state': next_features.cpu() if next_features is not None else None,
            'done': done
        })
    
    def push_sub(self, state, goal, action, reward, next_state, done):
        """存储Sub-level transition"""
        state_features = self.extract_state_features(state)
        next_features = self.extract_state_features(next_state) if next_state is not None else None
        
        self.sub_buffer.append({
            'state': state_features.cpu(),
            'goal': goal.cpu(),
            'action': action,
            'reward': reward,
            'next_state': next_features.cpu() if next_features is not None else None,
            'done': done
        })
    
    def train_meta(self):
        """训练Meta-Controller"""
        if len(self.meta_buffer) < self.batch_size:
            return None
        
        # 采样batch
        indices = np.random.choice(len(self.meta_buffer), self.batch_size, replace=False)
        batch = [self.meta_buffer[i] for i in indices]
        
        states = torch.stack([b['state'] for b in batch]).to(self.device)
        goals = torch.stack([b['goal'] for b in batch]).to(self.device)
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([b['next_state'] for b in batch if b['next_state'] is not None]).to(self.device)
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.float32).to(self.device)
        
        # 处理终止状态
        non_final_mask = torch.tensor([b['next_state'] is not None for b in batch], dtype=torch.bool)
        
        # 当前goal的Q值（简化：使用goal与state的相似度作为Q值）
        current_q = torch.sum(states * goals[:, :self.state_dim], dim=1)
        
        # 目标Q值
        next_q = torch.zeros(self.batch_size).to(self.device)
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                next_goals = self.goal_target_net(next_states)
                next_q[non_final_mask] = torch.sum(next_states * next_goals[:, :self.state_dim], dim=1)
        
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.goal_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.goal_net.parameters(), 1.0)
        self.goal_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.goal_net, self.goal_target_net)
        
        self.meta_step += 1
        return loss.item()
    
    def train_sub(self):
        """训练Sub-Controller"""
        if len(self.sub_buffer) < self.batch_size:
            return None
        
        # 采样batch
        indices = np.random.choice(len(self.sub_buffer), self.batch_size, replace=False)
        batch = [self.sub_buffer[i] for i in indices]
        
        states = torch.stack([b['state'] for b in batch]).to(self.device)
        goals = torch.stack([b['goal'] for b in batch]).to(self.device)
        actions = torch.tensor([b['action'] for b in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([b['reward'] for b in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([b['next_state'] for b in batch if b['next_state'] is not None]).to(self.device)
        dones = torch.tensor([b['done'] for b in batch], dtype=torch.float32).to(self.device)
        
        non_final_mask = torch.tensor([b['next_state'] is not None for b in batch], dtype=torch.bool)
        
        # 当前Q值
        current_q = self.sub_net(states, goals).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值（DDQN）
        next_q = torch.zeros(self.batch_size).to(self.device)
        if non_final_mask.sum() > 0:
            with torch.no_grad():
                # 使用当前网络选择动作
                next_actions = self.sub_net(next_states, goals[non_final_mask]).argmax(1)
                # 使用目标网络评估
                next_q[non_final_mask] = self.sub_target_net(
                    next_states, goals[non_final_mask]
                ).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.sub_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sub_net.parameters(), 1.0)
        self.sub_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.sub_net, self.sub_target_net)
        
        self.sub_step += 1
        return loss.item()
    
    def _soft_update(self, source_net, target_net):
        """软更新目标网络"""
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'goal_net': self.goal_net.state_dict(),
            'goal_target_net': self.goal_target_net.state_dict(),
            'sub_net': self.sub_net.state_dict(),
            'sub_target_net': self.sub_target_net.state_dict(),
            'goal_optimizer': self.goal_optimizer.state_dict(),
            'sub_optimizer': self.sub_optimizer.state_dict(),
            'meta_step': self.meta_step,
            'sub_step': self.sub_step,
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.goal_net.load_state_dict(checkpoint['goal_net'])
        self.goal_target_net.load_state_dict(checkpoint['goal_target_net'])
        self.sub_net.load_state_dict(checkpoint['sub_net'])
        self.sub_target_net.load_state_dict(checkpoint['sub_target_net'])
        self.goal_optimizer.load_state_dict(checkpoint['goal_optimizer'])
        self.sub_optimizer.load_state_dict(checkpoint['sub_optimizer'])
        self.meta_step = checkpoint.get('meta_step', 0)
        self.sub_step = checkpoint.get('sub_step', 0)
    
    def eval(self):
        """设置为评估模式"""
        self.is_training = False
        self.goal_net.eval()
        self.sub_net.eval()
    
    def train_mode(self):
        """设置为训练模式"""
        self.is_training = True
        self.goal_net.train()
        self.sub_net.train()


# ==================== 训练器类 ====================
class HierarchicalSFCTrainer:
    """分层强化学习训练器"""

    def __init__(self, config: Config, env: SFCEnv, name: str = "HRL_SFC"):
        self.config = config
        self.env = env
        self.name = name

        # 日志系统
        self.mylog = MyLog(Path(__file__), filesave=True, consoleprint=True, name=name)
        self.logger = self.mylog.logger

        # 创建分层代理
        self.agent = HierarchicalDQNAgent(config, num_nodes=len(env.nodes))

        # TensorBoard
        self.writer = SummaryWriter(f"./runs/HRL_{name}")

        # 训练参数
        self.meta_epsilon_start = 1.0
        self.meta_epsilon_end = 0.1
        self.meta_epsilon_decay = config.E_GREEDY_EPISODES // 2

        self.sub_epsilon_start = 1.0
        self.sub_epsilon_end = 0.01
        self.sub_epsilon_decay = config.E_GREEDY_EPISODES

        # 统计信息
        self.episode_rewards = []
        self.meta_losses = []
        self.sub_losses = []

        # Goal持续时间
        self.goal_horizon = 3

    def get_epsilon(self, episode: int, is_meta: bool = False) -> float:
        """计算epsilon"""
        if is_meta:
            start, end, decay = self.meta_epsilon_start, self.meta_epsilon_end, self.meta_epsilon_decay
        else:
            start, end, decay = self.sub_epsilon_start, self.sub_epsilon_end, self.sub_epsilon_decay

        if episode >= decay:
            return end

        epsilon = end + (start - end) * np.exp(-episode / decay)
        return epsilon

    def train_episode(self, episode: int, pkl_path: str = None):
        """训练单个episode"""
        # 重置环境
        if pkl_path is not None:
            self.env.read_pickle_and_modify(pkl_path)

        state, _ = self.env.reset()
        episode_reward = 0
        step_count = 0

        # Episode级别的goal
        current_goal = None
        goal_start_vnf = 0

        # Meta-level状态
        meta_state = state.clone()
        meta_cumulative_reward = 0

        # 训练循环
        while True:
            # Meta-level决策
            if current_goal is None or (self.env.current_vnf_index - goal_start_vnf) >= self.goal_horizon:
                # 保存上一个meta transition
                if current_goal is not None:
                    self.agent.push_meta(
                        meta_state,
                        current_goal,
                        meta_cumulative_reward,
                        state.clone(),
                        False
                    )

                # 训练Meta-Controller
                meta_loss = self.agent.train_meta()
                if meta_loss is not None:
                    self.meta_losses.append(meta_loss)
                    self.writer.add_scalar('Loss/Meta', meta_loss, self.agent.meta_step)

                # 选择新goal
                meta_epsilon = self.get_epsilon(episode, is_meta=True)
                current_goal = self.agent.select_goal(state, epsilon=meta_epsilon)

                # 重置meta状态
                meta_state = state.clone()
                meta_cumulative_reward = 0
                goal_start_vnf = self.env.current_vnf_index

            # Sub-level决策
            sub_epsilon = self.get_epsilon(episode, is_meta=False)
            valid_actions = self.env.get_valid_actions()

            if not valid_actions:
                self.logger.warning(f"Episode {episode}: No valid actions at step {step_count}")
                break

            action = self.agent.select_action(state, current_goal, sub_epsilon, valid_actions)

            # 环境交互
            next_state, extrinsic_reward, done, info = self.env.step(action)

            # 计算内在奖励
            intrinsic_reward = self.agent.compute_intrinsic_reward(current_goal, action, next_state)

            # 保存sub transition
            self.agent.push_sub(
                state.clone(),
                current_goal.clone(),
                action,
                intrinsic_reward,
                next_state.clone() if not done else None,
                done
            )

            # 训练Sub-Controller
            sub_loss = self.agent.train_sub()
            if sub_loss is not None:
                self.sub_losses.append(sub_loss)
                self.writer.add_scalar('Loss/Sub', sub_loss, self.agent.sub_step)

            # 更新统计
            episode_reward += extrinsic_reward
            meta_cumulative_reward += extrinsic_reward
            step_count += 1

            # 日志
            if step_count % 5 == 0:
                self.logger.info(
                    f"[Ep {episode}][Step {step_count}] "
                    f"VNF {self.env.current_vnf_index}/{len(self.env.vnfs)} | "
                    f"Reward: {extrinsic_reward:.4f} | "
                    f"Intrinsic: {intrinsic_reward:.4f} | "
                    f"Node: {action} | "
                    f"Tree: {len(self.env.tree_nodes)} nodes"
                )

            # 检查终止
            if done:
                # 保存最后的meta transition
                self.agent.push_meta(
                    meta_state,
                    current_goal,
                    meta_cumulative_reward,
                    next_state.clone(),
                    True
                )

                meta_loss = self.agent.train_meta()
                if meta_loss is not None:
                    self.meta_losses.append(meta_loss)

                self.logger.info(
                    f"[Episode {episode} Done] "
                    f"Reward: {episode_reward:.4f} | "
                    f"Steps: {step_count} | "
                    f"Status: {info.get('termination', 'unknown')}"
                )
                break

            state = next_state

        return episode_reward, step_count

    def train(self, num_episodes: int = 2000):
        """完整训练流程"""
        start_time = time.time()

        # 获取pkl文件列表
        pkl_files = list(self._get_pkl_files())
        if not pkl_files:
            self.logger.warning("未找到pkl文件，使用静态环境")
            pkl_files = [None] * num_episodes

        for episode in range(num_episodes):
            pkl_path = pkl_files[episode % len(pkl_files)]

            # 训练一个episode
            episode_reward, steps = self.train_episode(episode, pkl_path)

            # 记录
            self.episode_rewards.append(episode_reward)
            self.writer.add_scalar('Reward/Episode', episode_reward, episode)
            self.writer.add_scalar('Steps/Episode', steps, episode)
            self.writer.add_scalar('Epsilon/Meta', self.get_epsilon(episode, True), episode)
            self.writer.add_scalar('Epsilon/Sub', self.get_epsilon(episode, False), episode)

            # 定期保存
            if episode % 100 == 0 and episode > 0:
                self.save_checkpoint(episode)
                self._log_statistics(episode)

            # 定期评估
            if episode % 500 == 0 and episode > 0:
                self.evaluate(num_eval_episodes=10)

        # 最终保存
        self.save_checkpoint('final')
        self._log_statistics(num_episodes)

        elapsed_time = time.time() - start_time
        self.logger.info(f"训练完成，耗时 {elapsed_time / 3600:.2f} 小时")

    def evaluate(self, num_eval_episodes: int = 10):
        """评估模型性能"""
        self.logger.info(f"开始评估 {num_eval_episodes} episodes...")

        eval_rewards = []
        eval_success = []
        eval_qos = {'bw': [], 'delay': [], 'loss': [], 'length': []}

        self.agent.eval()

        for ep in range(num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            current_goal = None
            goal_start_vnf = 0

            while True:
                # Meta-level决策（贪婪）
                if current_goal is None or (self.env.current_vnf_index - goal_start_vnf) >= self.goal_horizon:
                    current_goal = self.agent.select_goal(state, epsilon=0.0)
                    goal_start_vnf = self.env.current_vnf_index

                # Sub-level决策（贪婪）
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break

                action = self.agent.select_action(state, current_goal, epsilon=0.0, valid_actions=valid_actions)
                next_state, reward, done, info = self.env.step(action)

                episode_reward += reward

                if done:
                    if info.get('termination') == 'sfc_completed':
                        eval_success.append(1)
                        bw, delay, loss, length = self.env.get_sfc_qos_params()
                        eval_qos['bw'].append(bw)
                        eval_qos['delay'].append(delay)
                        eval_qos['loss'].append(loss)
                        eval_qos['length'].append(length)
                    else:
                        eval_success.append(0)
                        eval_qos['bw'].append(0.0)
                        eval_qos['delay'].append(1.0)
                        eval_qos['loss'].append(1.0)
                        eval_qos['length'].append(0)
                    break

                state = next_state

            eval_rewards.append(episode_reward)

        self.agent.train_mode()

        # 统计结果
        avg_reward = np.mean(eval_rewards)
        success_rate = np.mean(eval_success)
        avg_bw = np.mean(eval_qos['bw'])
        avg_delay = np.mean(eval_qos['delay'])
        avg_loss = np.mean(eval_qos['loss'])
        avg_length = np.mean(eval_qos['length'])

        self.logger.info("=" * 60)
        self.logger.info(f"评估结果 ({num_eval_episodes} episodes):")
        self.logger.info(f"  平均奖励: {avg_reward:.4f}")
        self.logger.info(f"  成功率: {success_rate * 100:.2f}%")
        self.logger.info(f"  平均带宽: {avg_bw:.4f}")
        self.logger.info(f"  平均延迟: {avg_delay:.4f}")
        self.logger.info(f"  平均丢包: {avg_loss:.6f}")
        self.logger.info(f"  平均长度: {avg_length:.2f}")
        self.logger.info("=" * 60)

        return avg_reward, success_rate

    def save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint_dir = Path('./checkpoints') / self.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'hrl_episode_{episode}.pt'
        self.agent.save(str(checkpoint_path))

        stats = {
            'episode_rewards': self.episode_rewards,
            'meta_losses': self.meta_losses,
            'sub_losses': self.sub_losses,
        }
        stats_path = checkpoint_dir / f'stats_episode_{episode}.pkl'
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        self.logger.info(f"检查点已保存: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        self.agent.load(checkpoint_path)

        stats_path = Path(checkpoint_path).parent / f"stats_{Path(checkpoint_path).stem}.pkl"
        if stats_path.exists():
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
                self.episode_rewards = stats.get('episode_rewards', [])
                self.meta_losses = stats.get('meta_losses', [])
                self.sub_losses = stats.get('sub_losses', [])
            self.logger.info(f"统计信息已加载: {stats_path}")

    def _log_statistics(self, episode: int):
        """记录统计信息"""
        if len(self.episode_rewards) == 0:
            return

        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)

        self.logger.info("=" * 60)
        self.logger.info(f"训练统计 (Episode {episode}):")
        self.logger.info(f"  近100回合平均奖励: {avg_reward:.4f}")
        self.logger.info(f"  总回合数: {len(self.episode_rewards)}")
        self.logger.info(f"  Meta缓冲区大小: {len(self.agent.meta_buffer)}")
        self.logger.info(f"  Sub缓冲区大小: {len(self.agent.sub_buffer)}")
        self.logger.info(f"  Meta训练步数: {self.agent.meta_step}")
        self.logger.info(f"  Sub训练步数: {self.agent.sub_step}")
        if self.meta_losses:
            self.logger.info(f"  近期Meta损失: {np.mean(self.meta_losses[-100:]):.6f}")
        if self.sub_losses:
            self.logger.info(f"  近期Sub损失: {np.mean(self.sub_losses[-100:]):.6f}")
        self.logger.info("=" * 60)

    def _get_pkl_files(self):
        """获取pkl文件列表"""
        pkl_dir = Path(self.config.pkl_weight_path)
        if not pkl_dir.exists():
            return []

        pkl_files = list(pkl_dir.glob('*.pkl'))
        return sorted(pkl_files)[:self.config.PKL_CUT_NUM]


# ==================== 主函数 ====================
def parse_xml_topology(xml_path):
    """解析XML拓扑文件"""
    if not Path(xml_path).exists():
        # 创建默认拓扑
        print(f"警告: '{xml_path}' 不存在，创建模拟拓扑")
        graph = nx.Graph()
        for i in range(1, 6):
            graph.add_node(i, cpu=100.0, cpu_total=100.0)
        
        links = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 5), (2, 4)]
        for u, v in links:
            graph.add_edge(u, v, bandwidth=1000.0, delay=10.0, loss=0.01)
        
        return graph
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    topo_element = root.find("topology")
    graph = nx.Graph()

    for child in topo_element.iter():
        if child.tag == 'node':
            node_id = int(child.get('id'))
            cpu_total = float(child.get('cpu_total', 100.0))
            graph.add_node(node_id, cpu=cpu_total, cpu_total=cpu_total)
        elif child.tag == 'link':
            from_node = int(child.find('from').get('node'))
            to_node = int(child.find('to').get('node'))
            graph.add_edge(from_node, to_node,
                         bandwidth=float(child.get('bw', 1000.0)),
                         delay=float(child.get('delay', 10.0)),
                         loss=float(child.get('loss', 0.01)))

    return graph


def main():
    """主训练函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical RL for SFC Mapping")
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'eval', 'test'],
                       help='运行模式')
    parser.add_argument('--episodes', type=int, default=2000, 
                       help='训练回合数')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='检查点路径')
    parser.add_argument('--name', type=str, default='HRL_SFC', 
                       help='实验名称')
    parser.add_argument('--goal_horizon', type=int, default=3, 
                       help='Goal持续时间（每个goal用于多少个VNF）')
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "=" * 70)
    print("  分层强化学习 (HRL) SFC映射训练系统")
    print("=" * 70)
    print(f"  模式: {args.mode}")
    print(f"  实验名称: {args.name}")
    print(f"  Goal Horizon: {args.goal_horizon}")
    if args.checkpoint:
        print(f"  加载检查点: {args.checkpoint}")
    print("=" * 70 + "\n")

    # 初始化配置
    config = Config()

    # 解析拓扑
    graph = parse_xml_topology(config.xml_topology_path)
    print(f"✓ 拓扑加载完成: {len(graph.nodes)} 节点, {len(graph.edges)} 边")

    # 初始化环境
    vnfs = getattr(config, 'vnfs', [{'cpu': 10}, {'cpu': 5}, {'cpu': 7}])
    env = SFCEnv(graph, vnfs, config)
    print(f"✓ 环境初始化完成: {len(vnfs)} VNFs")

    # 创建训练器
    trainer = HierarchicalSFCTrainer(config, env, name=args.name)
    trainer.goal_horizon = args.goal_horizon
    print(f"✓ 训练器初始化完成\n")

    # 加载检查点
    if args.checkpoint:
        try:
            trainer.load_checkpoint(args.checkpoint)
            print(f"✓ 检查点加载成功: {args.checkpoint}\n")
        except Exception as e:
            print(f"✗ 检查点加载失败: {e}\n")
            return

    # 执行操作
    if args.mode == 'train':
        print(f"🚀 开始分层强化学习训练 ({args.episodes} episodes)...\n")
        try:
            trainer.train(num_episodes=args.episodes)
            print("\n✓ 训练完成!")
        except KeyboardInterrupt:
            print("\n\n⚠ 训练被用户中断")
            trainer.save_checkpoint('interrupted')
            print("✓ 中断状态已保存")
        except Exception as e:
            print(f"\n✗ 训练出错: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'eval':
        print("📊 开始评估...\n")
        try:
            avg_reward, success_rate = trainer.evaluate(num_eval_episodes=50)
            print(f"\n✓ 评估完成!")
            print(f"  最终平均奖励: {avg_reward:.4f}")
            print(f"  最终成功率: {success_rate * 100:.2f}%")
        except Exception as e:
            print(f"\n✗ 评估出错: {e}")
            import traceback
            traceback.print_exc()

    elif args.mode == 'test':
        print("🧪 运行测试模式...\n")
        print("测试不同Goal Horizon的影响:\n")
        
        results = []
        for horizon in [1, 2, 3, 5, 10]:
            print(f"--- 测试 Goal Horizon = {horizon} ---")
            trainer.goal_horizon = horizon
            try:
                avg_reward, success_rate = trainer.evaluate(num_eval_episodes=20)
                results.append({
                    'horizon': horizon,
                    'reward': avg_reward,
                    'success_rate': success_rate
                })
                print(f"  奖励: {avg_reward:.4f}")
                print(f"  成功率: {success_rate * 100:.2f}%\n")
            except Exception as e:
                print(f"  ✗ 测试失败: {e}\n")
        
        # 输出最佳配置
        if results:
            best_by_reward = max(results, key=lambda x: x['reward'])
            best_by_success = max(results, key=lambda x: x['success_rate'])
            
            print("=" * 70)
            print("测试总结:")
            print(f"  最佳奖励配置: Goal Horizon = {best_by_reward['horizon']} "
                  f"(Reward = {best_by_reward['reward']:.4f})")
            print(f"  最佳成功率配置: Goal Horizon = {best_by_success['horizon']} "
                  f"(Success Rate = {best_by_success['success_rate'] * 100:.2f}%)")
            print("=" * 70)

    print("\n" + "=" * 70)
    print("  程序结束")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
