# hirl_sfc_agent_gnn.py
"""
Final production-ready Agent for HIRL-SFC with GNN-based policy.
Integrated with Imitation Learning (Behavior Cloning) capabilities.

Features:
- Double DQN + Prioritized Experience Replay (PER)
- Imitation Learning (Supervised Update & Evaluation) [New]
- Robust Action Masking
- Mode Switching (RL <-> Imitation)
"""

import copy
import logging
import random
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


class GraphPrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.buffer = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self._min_priority = 1e-6
        self._max_priority = 1e6

    def add(self, state, action, reward, next_state, done, goal, next_valid_mask):
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        max_prio = np.clip(max_prio, self._min_priority, self._max_priority)

        data = (state, int(action), float(reward), next_state, bool(done), int(goal), next_valid_mask)

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def size(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int, beta: float = 0.4):
        current_len = len(self.buffer)
        if current_len == 0: return None, None, None

        priorities = self.priorities[:current_len]
        probs = np.power(priorities + 1e-8, self.alpha)
        probs /= (probs.sum() + 1e-10)

        indices = np.random.choice(current_len, batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = np.power(current_len * probs[indices], -beta)
        weights /= (weights.max() + 1e-10)

        states, actions, rewards, next_states, dones, goals, next_masks = zip(*samples)
        return (states, actions, rewards, next_states, dones, goals, next_masks), weights.astype(np.float32), indices

    def update_priorities(self, indices, priorities):
        for idx, p in zip(indices, priorities):
            if 0 <= idx < self.capacity:
                self.priorities[idx] = np.clip(p + 1e-6, self._min_priority, self._max_priority)

    def clear(self):
        self.buffer.clear()
        self.priorities.fill(0.0)
        self.pos = 0


class Agent_SFC_GNN:
    def __init__(self, model, n_actions: int, lr: float = 1e-4, gamma: float = 0.99,
                 buffer_size: int = 10000, batch_size: int = 32, device: str = 'cuda',
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: int = 10000,
                 prioritized_alpha: float = 0.6, prioritized_beta0: float = 0.4):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_actions = n_actions
        self.lr = lr  # 保存基础学习率
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        self._training = True

        self.policy_net = model.to(self.device)
        self.target_net = copy.deepcopy(model).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # 学习率调度器
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        self.memory = GraphPrioritizedReplayBuffer(buffer_size, alpha=prioritized_alpha)
        self.prioritized_beta0 = prioritized_beta0
        self.update_count = 0

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.steps_done / max(1, self.epsilon_decay))

    def select_action(self, state, goal_idx, valid_actions, epsilon=None, expert_action=None, beta=0.0):
        if not valid_actions: return 0

        self.steps_done += 1

        # 1. DAgger / Expert Guidance
        if expert_action is not None and expert_action in valid_actions:
            if random.random() < beta:
                return expert_action

        # 2. Epsilon-Greedy
        if epsilon is None: epsilon = self.get_epsilon()
        if random.random() < epsilon:
            return int(random.choice(valid_actions))

        # 3. RL Policy
        x, ei, ea, req = state

        # Prepare inputs
        x_d = x.to(self.device)
        ei_d = ei.to(self.device)
        ea_d = ea.to(self.device) if ea is not None else None

        if isinstance(req, np.ndarray):
            req_d = torch.from_numpy(req).float().to(self.device).unsqueeze(0)
        else:
            req_d = torch.tensor(req, dtype=torch.float32, device=self.device).unsqueeze(0)

        goal_d = torch.tensor([goal_idx], dtype=torch.long, device=self.device)
        batch_d = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

        # Action Mask
        mask = torch.zeros(1, self.n_actions, dtype=torch.bool, device=self.device)
        mask[0, valid_actions] = True

        with torch.no_grad():
            q = self.policy_net.forward_low(x_d, ei_d, ea_d, batch_d, req_d, goal_d, action_masks=mask)

        q_np = q.cpu().numpy().flatten()

        valid_q = q_np[valid_actions]
        if valid_q.size == 0 or np.all(valid_q <= -1e8):
            chosen = int(random.choice(valid_actions))
        else:
            best_idx_in_valid = np.argmax(valid_q)
            chosen = int(valid_actions[best_idx_in_valid])

        return chosen

    def store(self, state, action, reward, next_state, done, goal, next_valid_actions=None):
        """
        Store experience.
        Args:
            next_valid_actions: List of valid actions for next state (for Double DQN masking)
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        if next_valid_actions:
            valid = [a for a in next_valid_actions if 0 <= a < self.n_actions]
            mask[valid] = True
        else:
            mask[:] = True

        self.memory.add(state, action, reward, next_state, done, goal, mask)

    def _build_batch(self, states):
        data_list, reqs = [], []
        for x, ei, ea, r in states:
            x_cpu = x.cpu() if x.device.type != 'cpu' else x
            ei_cpu = ei.cpu() if ei.device.type != 'cpu' else ei
            if ea is None:
                ea_cpu = torch.zeros(ei_cpu.size(1), 1)
            else:
                ea_cpu = ea.cpu() if ea.device.type != 'cpu' else ea
            data_list.append(Data(x=x_cpu, edge_index=ei_cpu, edge_attr=ea_cpu))
            reqs.append(r)

        batch_graph = Batch.from_data_list(data_list).to(self.device)
        batch_req = torch.tensor(np.array(reqs), dtype=torch.float32, device=self.device)
        return batch_graph, batch_req

    def update(self, perform_logging=False):
        if self.memory.size() < self.batch_size: return 0.0

        beta = min(1.0, self.prioritized_beta0 + (self.update_count * (1.0 - self.prioritized_beta0) / 100000.0))
        batch, weights, idx = self.memory.sample(self.batch_size, beta)
        if batch is None: return 0.0

        states, acts, rews, next_states, dones, goals, next_masks = batch

        bg, br = self._build_batch(states)
        nbg, nbr = self._build_batch(next_states)

        at = torch.tensor(acts, dtype=torch.long, device=self.device)
        rt = torch.tensor(rews, dtype=torch.float32, device=self.device)
        dt = torch.tensor(dones, dtype=torch.float32, device=self.device)
        wt = torch.tensor(weights, dtype=torch.float32, device=self.device)
        gt = torch.tensor(goals, dtype=torch.long, device=self.device)
        nmt = torch.tensor(np.array(next_masks), dtype=torch.bool, device=self.device)

        # 1. Current Q
        q = self.policy_net.forward_low(bg.x, bg.edge_index, bg.edge_attr, bg.batch, br, gt)
        q_act = q.gather(1, at.unsqueeze(1)).squeeze(1)

        # 2. Next Q (Double DQN with Mask)
        with torch.no_grad():
            qn_online = self.policy_net.forward_low(nbg.x, nbg.edge_index, nbg.edge_attr, nbg.batch, nbr, gt,
                                                    action_masks=nmt)
            next_acts = qn_online.argmax(dim=1)

            qn_target = self.target_net.forward_low(nbg.x, nbg.edge_index, nbg.edge_attr, nbg.batch, nbr, gt)
            q_next = qn_target.gather(1, next_acts.unsqueeze(1)).squeeze(1)

            target = rt + (1 - dt) * self.gamma * q_next

        loss = (wt * F.smooth_l1_loss(q_act, target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        td_errors = (q_act - target).abs().detach().cpu().numpy()
        self.memory.update_priorities(idx, td_errors)

        self.update_count += 1
        if self.update_count % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    # =========================================================================
    # [扩展功能] 模仿学习模块 (Imitation Learning Extension)
    # =========================================================================

    def supervised_update(self, batch_data: List[Dict]) -> Tuple[float, float]:
        """
        使用专家数据进行监督学习更新 (Behavior Cloning)。
        Args:
            batch_data: List[Dict], 每个元素包含 'state', 'goal', 'action', 'valid_actions'
        Returns:
            avg_loss (float), accuracy (float)
        """
        if not batch_data:
            return 0.0, 0.0

        self.policy_net.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        correct_predictions = 0

        # 梯度累积 (Gradient Accumulation)
        # 注意：为了效率，也可以组装成Batch，但这里为了兼容 diverse input (valid_actions)
        # 我们采用逐样本 forward + accum 的方式，或者手动 batch 处理。
        # 下面使用逐样本累积梯度的方式，确保逻辑正确。

        for item in batch_data:
            state = item['state']
            goal = item['goal']
            expert_action = item['action']
            valid_actions = item['valid_actions']

            # Unpack state
            x, edge_index, edge_attr, req_vec = state

            # Prepare tensors
            x_d = x.to(self.device)
            ei_d = edge_index.to(self.device)
            ea_d = edge_attr.to(self.device) if edge_attr is not None else None

            if isinstance(req_vec, np.ndarray):
                req_d = torch.from_numpy(req_vec).float().to(self.device).unsqueeze(0)
            else:
                req_d = torch.tensor(req_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

            goal_d = torch.tensor([goal], dtype=torch.long, device=self.device)
            batch_d = torch.zeros(x_d.size(0), dtype=torch.long, device=self.device)

            # Action Mask
            mask = torch.zeros(1, self.n_actions, dtype=torch.bool, device=self.device)
            valid_indices = [a for a in valid_actions if 0 <= a < self.n_actions]
            if valid_indices:
                mask[0, valid_indices] = True

            # Forward
            q_values = self.policy_net.forward_low(
                x_d, ei_d, ea_d, batch_d, req_d, goal_d, action_masks=mask
            )

            # Loss: CrossEntropy (treating Q-values as logits for classification)
            target = torch.tensor([expert_action], dtype=torch.long, device=self.device)
            loss = F.cross_entropy(q_values, target)

            # Backward
            loss.backward()
            total_loss += loss.item()

            # Accuracy check
            pred_action = q_values.argmax(dim=1).item()
            if pred_action == expert_action:
                correct_predictions += 1

        # Gradient Clipping & Step
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        avg_loss = total_loss / len(batch_data)
        accuracy = (correct_predictions / len(batch_data)) * 100.0

        return avg_loss, accuracy

    def evaluate_imitation(self, eval_data: List[Dict], num_samples: int = 500) -> Dict:
        """
        评估模仿学习效果。
        """
        self.policy_net.eval()

        if len(eval_data) > num_samples:
            eval_samples = random.sample(eval_data, num_samples)
        else:
            eval_samples = eval_data

        correct = 0
        total = 0
        q_diffs = []

        with torch.no_grad():
            for item in eval_samples:
                # Same preprocessing as supervised_update
                x, ei, ea, req = item['state']
                goal = item['goal']
                expert_action = item['action']
                valid = item['valid_actions']

                x_d = x.to(self.device)
                ei_d = ei.to(self.device)
                ea_d = ea.to(self.device) if ea is not None else None

                if isinstance(req, np.ndarray):
                    req_d = torch.from_numpy(req).float().to(self.device).unsqueeze(0)
                else:
                    req_d = torch.tensor(req, dtype=torch.float32, device=self.device).unsqueeze(0)

                goal_d = torch.tensor([goal], dtype=torch.long, device=self.device)
                batch_d = torch.zeros(x_d.size(0), dtype=torch.long, device=self.device)

                mask = torch.zeros(1, self.n_actions, dtype=torch.bool, device=self.device)
                v_idxs = [a for a in valid if 0 <= a < self.n_actions]
                if v_idxs: mask[0, v_idxs] = True

                q = self.policy_net.forward_low(x_d, ei_d, ea_d, batch_d, req_d, goal_d, action_masks=mask)

                # Check accuracy
                agent_action = q.argmax(dim=1).item()
                if agent_action == expert_action:
                    correct += 1

                # Q-value gap
                expert_q = q[0, expert_action].item()
                max_q = q.max().item()
                q_diffs.append(max_q - expert_q)

                total += 1

        self.policy_net.train()  # Revert to train mode

        return {
            'accuracy': (correct / total * 100.0) if total > 0 else 0.0,
            'avg_q_gap': np.mean(q_diffs) if q_diffs else 0.0,
            'correct_count': correct,
            'total_samples': total
        }

    def switch_to_imitation_mode(self):
        """
        切换到模仿学习模式：使用更高的学习率，保存原优化器。
        """
        # 防止重复切换覆盖 rl_optimizer
        if not hasattr(self, 'rl_optimizer'):
            self.rl_optimizer = self.optimizer

        # BC 通常使用比 RL 更大的学习率
        self.imitation_lr = self.lr * 5.0
        self.bc_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.imitation_lr
        )
        self.optimizer = self.bc_optimizer
        logger.info(f"Switched to Imitation Mode (LR={self.imitation_lr})")

    def switch_to_rl_mode(self, start_epsilon: float = 0.3):
        """
        切换回 RL 模式：恢复优化器，重置探索率。
        """
        if hasattr(self, 'rl_optimizer'):
            self.optimizer = self.rl_optimizer
            logger.info("Restored RL Optimizer")
        else:
            # Fallback if switch_to_imitation wasn't called
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # 设置 RL 阶段的探索率
        # 通常 BC 后不需要很高的探索率
        self.steps_done = 0  # 重置步数以便 epsilon 从 start_epsilon 开始衰减
        self.epsilon_start = start_epsilon
        self.epsilon_end = 0.01
        self.epsilon_decay = 50000

        logger.info(f"Switched to RL Mode (Start Eps={start_epsilon})")

    def save(self, path):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'update_count': self.update_count
        }, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path):
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt['policy_net'])
            self.target_net.load_state_dict(ckpt['target_net'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.steps_done = ckpt.get('steps_done', 0)
            self.update_count = ckpt.get('update_count', 0)
            logger.info(f"Agent loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load agent: {e}")

    def clear_memory(self):
        self.memory.clear()