# -*- coding: utf-8 -*-
# @File    : hierarchical_rl.py
# @Author  : AI Assistant
# @Desc    : Hierarchical Reinforcement Learning for SFC Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import namedtuple, deque
from typing import Dict, List, Tuple, Optional
import logging

# ==================== 经验元组定义 ====================
MetaTransition = namedtuple('MetaTransition',
                            ('state', 'goal', 'meta_reward', 'next_state', 'done'))

SubTransition = namedtuple('SubTransition',
                           ('state', 'goal', 'action', 'intrinsic_reward', 'next_state', 'done'))


# ==================== 高层策略网络（Meta-Controller）====================
class MetaController(nn.Module):
    """
    高层控制器：负责为每个VNF选择部署目标（goal）
    Goal可以是：节点区域、资源偏好、QoS约束等抽象策略
    """

    def __init__(self, state_channels: int, num_nodes: int, goal_dim: int = 16):
        super(MetaController, self).__init__()
        self.goal_dim = goal_dim
        self.num_nodes = num_nodes

        # 状态编码器（共享）
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.ReLU()
        )

        # 全局池化 + 全连接
        self.fc_input_size = 64 * num_nodes * num_nodes

        # Goal生成网络（输出goal向量）
        self.goal_generator = nn.Sequential(
            nn.Linear(self.fc_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, goal_dim)
        )

        # 价值网络（用于meta-level Q-learning）
        self.value_net = nn.Sequential(
            nn.Linear(self.fc_input_size + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播生成goal
        Args:
            state: (batch, channels, H, W)
        Returns:
            goal: (batch, goal_dim)
        """
        batch_size = state.shape[0]
        x = self.state_encoder(state)
        x = x.view(batch_size, -1)
        goal = self.goal_generator(x)
        return goal

    def compute_value(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """计算状态-目标对的价值"""
        batch_size = state.shape[0]
        x = self.state_encoder(state)
        x = x.view(batch_size, -1)
        combined = torch.cat([x, goal], dim=1)
        value = self.value_net(combined)
        return value


# ==================== 低层策略网络（Sub-Controller）====================
class SubController(nn.Module):
    """
    低层控制器：根据高层goal选择具体动作（节点选择）
    使用goal-conditioned policy
    """

    def __init__(self, state_channels: int, num_nodes: int, goal_dim: int = 16):
        super(SubController, self).__init__()
        self.num_nodes = num_nodes

        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(1, 1)),
            nn.ReLU()
        )

        self.fc_input_size = 64 * num_nodes * num_nodes

        # Goal-conditioned Q网络
        self.q_network = nn.Sequential(
            nn.Linear(self.fc_input_size + goal_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes)  # 每个节点的Q值
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算Q值
        Args:
            state: (batch, channels, H, W)
            goal: (batch, goal_dim)
        Returns:
            q_values: (batch, num_nodes)
        """
        batch_size = state.shape[0]
        x = self.state_encoder(state)
        x = x.view(batch_size, -1)
        combined = torch.cat([x, goal], dim=1)
        q_values = self.q_network(combined)
        return q_values


# ==================== 目标嵌入网络（Goal Embedding）====================
class GoalEmbedding(nn.Module):
    """
    将离散的节点选择映射到goal空间
    用于计算intrinsic reward
    """

    def __init__(self, num_nodes: int, goal_dim: int = 16):
        super(GoalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_nodes, goal_dim)

    def forward(self, node_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_ids: (batch,) 节点ID
        Returns:
            embeddings: (batch, goal_dim)
        """
        return self.embedding(node_ids)


# ==================== 分层经验回放 ====================
class HierarchicalReplayBuffer:
    """分层经验回放池"""

    def __init__(self, meta_capacity: int = 5000, sub_capacity: int = 50000):
        self.meta_buffer = deque(maxlen=meta_capacity)
        self.sub_buffer = deque(maxlen=sub_capacity)

    def push_meta(self, *args):
        self.meta_buffer.append(MetaTransition(*args))

    def push_sub(self, *args):
        self.sub_buffer.append(SubTransition(*args))

    def sample_meta(self, batch_size: int):
        import random
        return random.sample(self.meta_buffer, min(batch_size, len(self.meta_buffer)))

    def sample_sub(self, batch_size: int):
        import random
        return random.sample(self.sub_buffer, min(batch_size, len(self.sub_buffer)))

    def __len__(self):
        return len(self.meta_buffer), len(self.sub_buffer)


# ==================== 分层DQN Agent ====================
class HierarchicalDQNAgent:
    """
    分层DQN代理
    实现两层决策：Meta-Controller + Sub-Controller
    """

    def __init__(self, config, num_nodes: int):
        self.config = config
        self.num_nodes = num_nodes
        self.goal_dim = 16
        self.device = config.DEVICE

        # 初始化网络
        self.meta_controller = MetaController(
            config.NUM_STATES, num_nodes, self.goal_dim
        ).to(self.device)

        self.sub_controller = SubController(
            config.NUM_STATES, num_nodes, self.goal_dim
        ).to(self.device)

        self.goal_embedding = GoalEmbedding(
            num_nodes, self.goal_dim
        ).to(self.device)

        # 目标网络
        self.target_meta = MetaController(
            config.NUM_STATES, num_nodes, self.goal_dim
        ).to(self.device)
        self.target_meta.load_state_dict(self.meta_controller.state_dict())

        self.target_sub = SubController(
            config.NUM_STATES, num_nodes, self.goal_dim
        ).to(self.device)
        self.target_sub.load_state_dict(self.sub_controller.state_dict())

        # 优化器
        self.meta_optimizer = torch.optim.Adam(
            self.meta_controller.parameters(), lr=config.LR * 0.1
        )
        self.sub_optimizer = torch.optim.Adam(
            self.sub_controller.parameters(), lr=config.LR
        )

        # 经验回放
        self.replay_buffer = HierarchicalReplayBuffer()

        # 训练参数
        self.gamma = config.DISCOUNT
        self.batch_size = config.BATCH_SIZE
        self.tau = config.TAU
        self.update_target_freq = config.UPDATE_FREQUENCY

        # 统计信息
        self.meta_step = 0
        self.sub_step = 0
        self.current_goal = None

        # Logger
        self.logger = logging.getLogger('HierarchicalDQN')

    def select_goal(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        Meta-controller选择goal
        Args:
            state: 当前状态
            epsilon: 探索率
        Returns:
            goal: goal向量
        """
        if np.random.rand() < epsilon:
            # 随机goal
            goal = torch.randn(1, self.goal_dim, device=self.device)
        else:
            with torch.no_grad():
                goal = self.meta_controller(state)

        return goal

    def select_action(self, state: torch.Tensor, goal: torch.Tensor,
                      epsilon: float = 0.0, valid_actions: Optional[List[int]] = None) -> int:
        """
        Sub-controller根据goal选择动作
        Args:
            state: 当前状态
            goal: 当前goal
            epsilon: 探索率
            valid_actions: 有效动作列表
        Returns:
            action: 选择的动作（节点ID）
        """
        if valid_actions is None:
            valid_actions = list(range(self.num_nodes))

        if np.random.rand() < epsilon:
            action = np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                q_values = self.sub_controller(state, goal)
                # 只考虑有效动作
                mask = torch.zeros(self.num_nodes, device=self.device)
                mask[valid_actions] = 1
                q_values = q_values * mask - (1 - mask) * 1e9
                action = q_values.argmax(dim=1).item()

        return action

    def compute_intrinsic_reward(self, goal: torch.Tensor,
                                 action: int, next_state: torch.Tensor) -> float:
        """
        计算内在奖励（goal达成度）
        使用cosine similarity衡量action是否符合goal
        """
        with torch.no_grad():
            # 获取动作的goal嵌入
            action_tensor = torch.tensor([action], device=self.device)
            action_goal = self.goal_embedding(action_tensor)

            # 计算相似度
            similarity = F.cosine_similarity(goal, action_goal, dim=1)
            intrinsic_reward = similarity.item()

        return intrinsic_reward

    def train_meta(self):
        """训练Meta-Controller"""
        if len(self.replay_buffer.meta_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample_meta(self.batch_size)
        batch = MetaTransition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        goal_batch = torch.cat(batch.goal).to(self.device)
        reward_batch = torch.tensor(batch.meta_reward, device=self.device).unsqueeze(1)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float).unsqueeze(1)

        # 计算当前Q值
        current_q = self.meta_controller.compute_value(state_batch, goal_batch)

        # 计算目标Q值
        with torch.no_grad():
            next_goals = self.target_meta(next_state_batch)
            next_q = self.target_meta.compute_value(next_state_batch, next_goals)
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 优化
        self.meta_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_controller.parameters(), 1.0)
        self.meta_optimizer.step()

        self.meta_step += 1

        # 软更新目标网络
        if self.meta_step % self.update_target_freq == 0:
            self._soft_update(self.target_meta, self.meta_controller)

        return loss.item()

    def train_sub(self):
        """训练Sub-Controller"""
        if len(self.replay_buffer.sub_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample_sub(self.batch_size)
        batch = SubTransition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        goal_batch = torch.cat(batch.goal).to(self.device)
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.intrinsic_reward, device=self.device).unsqueeze(1)
        next_state_batch = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float).unsqueeze(1)

        # 计算当前Q值
        current_q = self.sub_controller(state_batch, goal_batch).gather(1, action_batch.long())

        # 计算目标Q值（DDQN）
        with torch.no_grad():
            next_q_policy = self.sub_controller(next_state_batch, goal_batch[:len(next_state_batch)])
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)
            next_q_target = self.target_sub(next_state_batch, goal_batch[:len(next_state_batch)])
            next_q = next_q_target.gather(1, next_actions)
            target_q = reward_batch + (1 - done_batch[:len(next_state_batch)]) * self.gamma * next_q

        # 计算损失
        loss = F.mse_loss(current_q, target_q)

        # 优化
        self.sub_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.sub_controller.parameters(), 1.0)
        self.sub_optimizer.step()

        self.sub_step += 1

        # 软更新目标网络
        if self.sub_step % self.update_target_freq == 0:
            self._soft_update(self.target_sub, self.sub_controller)

        return loss.item()

    def _soft_update(self, target_net, policy_net):
        """软更新目标网络"""
        for target_param, param in zip(target_net.parameters(), policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'meta_controller': self.meta_controller.state_dict(),
            'sub_controller': self.sub_controller.state_dict(),
            'goal_embedding': self.goal_embedding.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'sub_optimizer': self.sub_optimizer.state_dict(),
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_controller.load_state_dict(checkpoint['meta_controller'])
        self.sub_controller.load_state_dict(checkpoint['sub_controller'])
        self.goal_embedding.load_state_dict(checkpoint['goal_embedding'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer'])
        self.sub_optimizer.load_state_dict(checkpoint['sub_optimizer'])

        # 同步目标网络
        self.target_meta.load_state_dict(self.meta_controller.state_dict())
        self.target_sub.load_state_dict(self.sub_controller.state_dict())

        self.logger.info(f"Model loaded from {path}")


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 创建配置
    class DummyConfig:
        NUM_STATES = 7
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LR = 1e-4
        DISCOUNT = 0.95
        BATCH_SIZE = 64
        TAU = 0.01
        UPDATE_FREQUENCY = 100


    config = DummyConfig()
    num_nodes = 14

    # 创建分层代理
    agent = HierarchicalDQNAgent(config, num_nodes)

    # 模拟训练循环
    dummy_state = torch.randn(1, 7, 14, 14)

    # Meta-level决策
    goal = agent.select_goal(dummy_state, epsilon=0.1)
    print(f"Selected goal shape: {goal.shape}")

    # Sub-level决策
    action = agent.select_action(dummy_state, goal, epsilon=0.1)
    print(f"Selected action: {action}")

    # 计算内在奖励
    intrinsic_reward = agent.compute_intrinsic_reward(goal, action, dummy_state)
    print(f"Intrinsic reward: {intrinsic_reward:.4f}")

    print("✅ Hierarchical RL framework initialized successfully!")