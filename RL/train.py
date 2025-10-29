# -*- coding: utf-8 -*-
# @File     : train_hierarchical.py
# @Author   : AI Assistant
# @Desc     : Training script for Hierarchical RL in SFC Mapping
# @Fixes    : Corrected class structure and logical flow

import sys
import time
import pickle
import logging
from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


# -------------------------------------------
# ⚠️ 假设您的原有模块已正确导入
from config import Config
from env import SFCEnv
from rl import HierarchicalDQNAgent
from log import MyLog
# -------------------------------------------

# Placeholder classes for running the script
# (Replace with your actual imports)
class Config:
    E_GREEDY_EPISODES = 1000
    xml_topology_path = "topology.xml"  # 假设的拓扑文件
    pkl_weight_path = "./pkl_files"
    PKL_CUT_NUM = 100
    vnfs = [{'cpu': 10}, {'cpu': 5}, {'cpu': 7}]


class SFCEnv:
    def __init__(self, graph, vnfs, config):
        self.nodes = graph.nodes()
        self.vnfs = vnfs
        self.config = config
        self.current_vnf_index = 0
        self.tree_nodes = []
        print("Mock SFCEnv initialized.")

    def reset(self):
        self.current_vnf_index = 0
        self.tree_nodes = []
        return torch.rand(10), {}  # 返回一个虚拟状态

    def read_pickle_and_modify(self, pkl_path):
        pass  # 模拟

    def get_valid_actions(self):
        return [1, 2, 3]  # 模拟

    def step(self, action):
        self.current_vnf_index += 1
        self.tree_nodes.append(action)
        done = self.current_vnf_index >= len(self.vnfs)
        info = {'termination': 'sfc_completed' if done else 'in_progress'}
        reward = np.random.rand()
        next_state = torch.rand(10)
        return next_state, reward, done, info

    def get_sfc_qos_params(self):
        return 0.9, 0.5, 0.01, 5  # bw, delay, loss, length


class HierarchicalDQNAgent:
    def __init__(self, config, num_nodes):
        self.meta_step = 0
        self.sub_step = 0
        self.replay_buffer = self  # Mock replay buffer
        self.meta_buffer = deque(maxlen=1000)
        self.sub_buffer = deque(maxlen=10000)
        print("Mock HierarchicalDQNAgent initialized.")

    def select_goal(self, state, epsilon):
        return torch.rand(5)  # 模拟

    def select_action(self, state, goal, epsilon, valid_actions):
        return np.random.choice(valid_actions)  # 模拟

    def compute_intrinsic_reward(self, goal, action, next_state):
        return np.random.rand() * 0.1  # 模拟

    def push_meta(self, s, g, r, ns, d):
        self.meta_buffer.append((s, g, r, ns, d))

    def push_sub(self, s, g, a, r, ns, d):
        self.sub_buffer.append((s, g, a, r, ns, d))

    def train_meta(self):
        if len(self.meta_buffer) < 100: return None
        self.meta_step += 1
        return np.random.rand() * 0.1  # 模拟

    def train_sub(self):
        if len(self.sub_buffer) < 100: return None
        self.sub_step += 1
        return np.random.rand() * 0.5  # 模拟

    def save(self, path):
        pass

    def load(self, path):
        pass

    def eval(self):
        pass

    def train_mode(self):
        pass


class MyLog:
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(kwargs.get("name", "HRL_SFC"))
        print("Mock MyLog initialized.")


# (End of placeholder classes)


class HierarchicalSFCTrainer:
    """
    分层强化学习训练器（适配SFC映射）
    """

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

        # Goal持续时间（每个goal用于多少步）
        self.goal_horizon = 3  # 每3个VNF重新选择goal

    def get_epsilon(self, episode: int, is_meta: bool = False) -> float:
        """计算epsilon（epsilon-greedy策略）"""
        if is_meta:
            start, end, decay = self.meta_epsilon_start, self.meta_epsilon_end, self.meta_epsilon_decay
        else:
            start, end, decay = self.sub_epsilon_start, self.sub_epsilon_end, self.sub_epsilon_decay

        if episode >= decay:
            return end

        epsilon = end + (start - end) * np.exp(-episode / decay)
        return epsilon

    def train_episode(self, episode: int, pkl_path: str = None):
        """
        训练单个episode（分层决策）
        (FIXED: 这是 `train_episode` 的正确逻辑)
        """
        # 重置环境
        if pkl_path is not None:
            self.env.read_pickle_and_modify(pkl_path)

        state, _ = self.env.reset()
        episode_reward = 0
        step_count = 0

        # Episode级别的goal（初始化）
        current_goal = None
        goal_start_vnf = 0  # 当前goal从哪个VNF开始

        # 用于meta-level transition的缓存
        meta_state = state.clone()
        meta_cumulative_reward = 0

        # 训练循环
        while True:
            # ============== Meta-level决策 ==============
            if current_goal is None or (self.env.current_vnf_index - goal_start_vnf) >= self.goal_horizon:
                # 1. 保存上一个meta transition（如果有）
                if current_goal is not None:
                    self.agent.replay_buffer.push_meta(
                        meta_state,
                        current_goal,
                        meta_cumulative_reward,
                        state.clone(),
                        False  # is_done=False
                    )

                # 2. 训练Meta-Controller
                meta_loss = self.agent.train_meta()
                if meta_loss is not None:
                    self.meta_losses.append(meta_loss)
                    self.writer.add_scalar('Loss/Meta', meta_loss, self.agent.meta_step)

                # 3. 选择新goal (epsilon-greedy)
                meta_epsilon = self.get_epsilon(episode, is_meta=True)
                current_goal = self.agent.select_goal(state, epsilon=meta_epsilon)

                # 4. 重置meta状态
                meta_state = state.clone()
                meta_cumulative_reward = 0
                goal_start_vnf = self.env.current_vnf_index

            # ============== Sub-level决策 ==============
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

            # 组合奖励（可调整权重）
            # combined_reward = 0.5 * extrinsic_reward + 0.5 * intrinsic_reward

            # 保存sub transition (Sub-controller使用内在奖励)
            self.agent.replay_buffer.push_sub(
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
            meta_cumulative_reward += extrinsic_reward  # Meta-controller使用累积的外在奖励
            step_count += 1

            # 日志记录
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
                # 保存最后的meta transition (is_done=True)
                self.agent.replay_buffer.push_meta(
                    meta_state,
                    current_goal,
                    meta_cumulative_reward,
                    next_state.clone(),
                    True
                )

                # 最后训练一次
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
        """
        完整训练流程
        (FIXED: 已移入class)
        """
        start_time = time.time()

        # 获取pkl文件列表
        pkl_files = list(self._get_pkl_files())
        if not pkl_files:
            self.logger.warning("未找到pkl文件，使用静态环境")
            pkl_files = [None] * num_episodes

        for episode in range(num_episodes):
            # 选择pkl文件（循环使用）
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
        self.logger.info(f"Training completed in {elapsed_time / 3600:.2f} hours")

    def evaluate(self, num_eval_episodes: int = 10):
        """
        评估模型性能（无探索）
        (FIXED: 已补全并移入class)
        """
        self.logger.info(f"Starting evaluation for {num_eval_episodes} episodes...")

        eval_rewards = []
        eval_success = []
        eval_qos = {'bw': [], 'delay': [], 'loss': [], 'length': []}

        self.agent.eval()  # 设置为评估模式

        for ep in range(num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            current_goal = None
            goal_start_vnf = 0

            while True:
                # Meta-level决策（贪婪策略）
                if current_goal is None or (self.env.current_vnf_index - goal_start_vnf) >= self.goal_horizon:
                    current_goal = self.agent.select_goal(state, epsilon=0.0)
                    goal_start_vnf = self.env.current_vnf_index

                # Sub-level决策（贪婪策略）
                valid_actions = self.env.get_valid_actions()
                if not valid_actions:
                    break  # 失败

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

        self.agent.train_mode()  # 恢复训练模式

        # 统计结果
        avg_reward = np.mean(eval_rewards)
        success_rate = np.mean(eval_success)
        avg_bw = np.mean(eval_qos['bw'])
        avg_delay = np.mean(eval_qos['delay'])
        avg_loss = np.mean(eval_qos['loss'])
        avg_length = np.mean(eval_qos['length'])

        self.logger.info("=" * 60)
        self.logger.info(f"Evaluation Results ({num_eval_episodes} episodes):")
        self.logger.info(f"  Average Reward: {avg_reward:.4f}")
        self.logger.info(f"  Success Rate: {success_rate * 100:.2f}%")
        self.logger.info(f"  Average BW: {avg_bw:.4f}")
        self.logger.info(f"  Average Delay: {avg_delay:.4f}")
        self.logger.info(f"  Average Loss: {avg_loss:.6f}")
        self.logger.info(f"  Average Length: {avg_length:.2f}")
        self.logger.info("=" * 60)

        return avg_reward, success_rate

    def save_checkpoint(self, episode):
        """保存检查点"""
        checkpoint_dir = Path('./checkpoints') / self.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'hrl_episode_{episode}.pt'
        self.agent.save(str(checkpoint_path))

        # 保存训练统计
        stats = {
            'episode_rewards': self.episode_rewards,
            'meta_losses': self.meta_losses,
            'sub_losses': self.sub_losses,
        }
        stats_path = checkpoint_dir / f'stats_episode_{episode}.pkl'
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        self.agent.load(checkpoint_path)

        # 尝试加载统计信息
        stats_path = Path(checkpoint_path).parent / f"stats_{Path(checkpoint_path).stem}.pkl"
        if stats_path.exists():
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
                self.episode_rewards = stats.get('episode_rewards', [])
                self.meta_losses = stats.get('meta_losses', [])
                self.sub_losses = stats.get('sub_losses', [])
            self.logger.info(f"Statistics loaded from {stats_path}")

    def _log_statistics(self, episode: int):
        """记录训练统计"""
        if len(self.episode_rewards) == 0:
            return

        recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        avg_reward = np.mean(recent_rewards)

        self.logger.info("=" * 60)
        self.logger.info(f"Training Statistics (Episode {episode}):")
        self.logger.info(f"  Recent 100 Avg Reward: {avg_reward:.4f}")
        self.logger.info(f"  Total Episodes: {len(self.episode_rewards)}")
        self.logger.info(f"  Meta Buffer Size: {len(self.agent.replay_buffer.meta_buffer)}")
        self.logger.info(f"  Sub Buffer Size: {len(self.agent.replay_buffer.sub_buffer)}")
        self.logger.info(f"  Meta Training Steps: {self.agent.meta_step}")
        self.logger.info(f"  Sub Training Steps: {self.agent.sub_step}")
        if self.meta_losses:
            self.logger.info(f"  Recent Meta Loss: {np.mean(self.meta_losses[-100:]):.6f}")
        if self.sub_losses:
            self.logger.info(f"  Recent Sub Loss: {np.mean(self.sub_losses[-100:]):.6f}")
        self.logger.info("=" * 60)

    def _get_pkl_files(self):
        """获取pkl文件列表"""
        pkl_dir = Path(self.config.pkl_weight_path)
        if not pkl_dir.exists():
            return []

        pkl_files = list(pkl_dir.glob('*.pkl'))
        return sorted(pkl_files)[:self.config.PKL_CUT_NUM]


# ==================== 高级变体：Options框架 ====================
class OptionBasedHRL:
    """
    基于Options的分层RL（Sutton et al., 1999）
    Options定义为时间上的抽象动作（策略片段）
    """

    def __init__(self, config, num_nodes: int):
        self.config = config
        self.num_nodes = num_nodes

        # 定义Options（每个Option是一个子策略）
        self.options = self._define_options()
        self.num_options = len(self.options)

        # Option选择网络
        self.option_selector = self._build_option_selector()

        # 每个Option的Q网络
        self.option_q_networks = {
            opt_id: self._build_option_q_network()
            for opt_id in range(self.num_options)
        }

    def _define_options(self):
        """
        定义Options（策略集合）
        每个Option代表一种部署策略
        """
        options = [
            {'id': 0, 'name': 'greedy_resource', 'desc': '优先选择资源充足的节点'},
            {'id': 1, 'name': 'greedy_qos', 'desc': '优先选择QoS最优的节点'},
            {'id': 2, 'name': 'balanced', 'desc': '平衡资源和QoS'},
            {'id': 3, 'name': 'exploration', 'desc': '探索性部署'},
        ]
        return options

    def _build_option_selector(self):
        """构建Option选择器"""
        # 简化示例
        return None

    def _build_option_q_network(self):
        """构建Option级别的Q网络"""
        # 简化示例
        return None


# ==================== 主函数 ====================
def main():
    """主训练函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Hierarchical RL for SFC Mapping")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'test'])
    parser.add_argument('--episodes', type=int, default=2000, help='Number of training episodes')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path to load')
    parser.add_argument('--name', type=str, default='HRL_SFC', help='Experiment name')
    parser.add_argument('--goal_horizon', type=int, default=3, help='Goal horizon (VNFs per goal)')
    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 初始化配置
    # from config import Config
    config = Config()

    # 解析拓扑
    import xml.etree.ElementTree as ET
    import networkx as nx

    # --- Mock Topology Generation (if XML not found) ---
    if not Path(config.xml_topology_path).exists():
        print(f"Warning: '{config.xml_topology_path}' not found. Creating a mock graph.")
        graph = nx.Graph()
        graph.add_node(1, cpu=100.0, cpu_total=100.0)
        graph.add_node(2, cpu=100.0, cpu_total=100.0)
        graph.add_node(3, cpu=100.0, cpu_total=100.0)
        graph.add_edge(1, 2, bandwidth=1000.0, delay=10.0, loss=0.01)
        graph.add_edge(2, 3, bandwidth=1000.0, delay=10.0, loss=0.01)
    else:
        tree = ET.parse(config.xml_topology_path)
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

    # 初始化环境
    # from env import SFCEnv
    vnfs = getattr(config, 'vnfs', [{'cpu': 10}, {'cpu': 5}, {'cpu': 7}])
    env = SFCEnv(graph, vnfs, config)

    # 创建训练器
    trainer = HierarchicalSFCTrainer(config, env, name=args.name)
    trainer.goal_horizon = args.goal_horizon

    # 加载检查点
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        print(f"✅ Checkpoint loaded: {args.checkpoint}")

    # 执行操作
    if args.mode == 'train':
        print(f"🚀 Starting Hierarchical RL training for {args.episodes} episodes...")
        trainer.train(num_episodes=args.episodes)
        print("✅ Training completed!")

    elif args.mode == 'eval':
        print("📊 Starting evaluation...")
        trainer.evaluate(num_eval_episodes=50)
        print("✅ Evaluation completed!")

    elif args.mode == 'test':
        print("🧪 Running test mode...")
        # 测试不同goal_horizon的影响
        for horizon in [1, 2, 3, 5]:
            print(f"\n--- Testing with goal_horizon={horizon} ---")
            trainer.goal_horizon = horizon
            avg_reward, success_rate = trainer.evaluate(num_eval_episodes=20)
            print(f"Goal Horizon {horizon}: Reward={avg_reward:.4f}, Success={success_rate * 100:.2f}%")


if __name__ == "__main__":
    main()