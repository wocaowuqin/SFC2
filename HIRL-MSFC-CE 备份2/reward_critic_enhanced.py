# reward_critic_enhanced.py
import numpy as np
from typing import Dict, Optional
from collections import defaultdict


class RewardCritic:
    """
    增强版HRL奖励系统 for 多播VNF映射
    - HRL高层/低层奖励分离
    - DAgger专家一致性奖励
    - BackupPolicy多级回退感知
    - Phase-aware训练阶段适配
    - QoS违反惩罚
    - 自适应惩罚权重
    """

    def __init__(self, training_phase: int = 2, epoch: int = 0, max_epochs: int = 1200):
        self.phase = training_phase
        self.epoch = epoch
        self.max_epochs = max_epochs

        # 折扣因子
        self.gamma_high = 0.99
        self.gamma_low = 0.95

        # 自适应backup惩罚追踪
        self.backup_success_rate = defaultdict(lambda: 0.5)
        self.backup_usage_count = defaultdict(int)

        # 奖励归一化范围
        self.reward_min = -10.0
        self.reward_max = 10.0

    # ----------------------------------------------------------------
    # 高层奖励（Meta-Controller）
    # ----------------------------------------------------------------
    def high_level_reward(self,
                          goal_reached: bool,
                          total_cost: float,
                          qos_satisfied: bool,
                          blocking_rate: float = 0.0) -> float:
        """
        全局任务奖励
        Args:
            goal_reached: 是否完成所有目的节点连接
            total_cost: 归一化总成本 [0, 1]
            qos_satisfied: 是否满足QoS约束
            blocking_rate: 当前网络阻塞率
        """
        reward = 0.0

        # 主要目标奖励
        if goal_reached and qos_satisfied:
            reward += 10.0
            # 效率bonus（成本越低越好）
            reward += 2.0 * (1.0 - total_cost)
        elif goal_reached:
            reward += 5.0

        # 全局cost惩罚
        reward -= 2.5 * total_cost

        # 网络健康度奖励（降低阻塞率）
        if blocking_rate < 0.1:
            reward += 1.0
        elif blocking_rate > 0.5:
            reward -= 2.0

        return self._normalize_reward(reward)

    # ----------------------------------------------------------------
    # 低层奖励（Worker-Controller）
    # ----------------------------------------------------------------
    def low_level_reward(self,
                         sub_task_completed: bool,
                         step_cost: float,
                         request_failed: bool,
                         progress_to_goal: float,
                         backup_used: bool,
                         backup_level: str,
                         qos_violations: Optional[Dict[str, float]] = None,
                         failure_reason: Optional[str] = None) -> float:
        """
        单步奖励
        Args:
            sub_task_completed: 是否完成子任务（连接到一个目的节点）
            step_cost: 当前步骤成本 [0, 1]
            request_failed: 是否失败
            progress_to_goal: 进度指标 [-1, 1]
            backup_used: 是否使用了备份策略
            backup_level: 备份级别
            qos_violations: QoS违反字典 {"delay": 0.2, "bandwidth": 0.1}
            failure_reason: 失败原因
        """
        reward = 0.0

        # ========== Phase-aware 基础奖励 ==========
        if self.phase == 1:  # 模仿学习阶段
            if sub_task_completed:
                reward += 5.0
            reward -= 0.3 * step_cost
            if request_failed:
                reward -= 8.0
        else:  # Phase 2: 混合RL训练
            # 完成奖励
            if sub_task_completed:
                # 效率bonus
                efficiency = max(0, 1.0 - step_cost)
                reward += 3.0 + efficiency

            # 进度shaping（关键！）
            reward += 1.5 * progress_to_goal

            # 分段cost惩罚
            if step_cost < 0.3:
                reward -= 0.5 * step_cost
            elif step_cost < 0.7:
                reward -= 1.5 * step_cost
            else:
                reward -= 3.0 * step_cost  # 高成本重罚

            # 动态失败惩罚（随训练进度递减）
            if request_failed:
                base_penalty = self._compute_failure_penalty(failure_reason)
                decay_factor = 1.0 - (self.epoch / self.max_epochs)
                reward -= base_penalty * decay_factor

        # ========== BackupPolicy惩罚 ==========
        if backup_used:
            penalty = self._get_adaptive_backup_penalty(backup_level)
            reward -= penalty
        else:
            reward += 0.5  # 主路径成功bonus

        # ========== QoS违反惩罚 ==========
        if qos_violations:
            qos_penalty = self._compute_qos_penalty(qos_violations)
            reward -= qos_penalty

        return self._normalize_reward(reward, clip_range=(-10.0, 6.0))

    # ----------------------------------------------------------------
    # DAgger增强奖励
    # ----------------------------------------------------------------
    def dagger_augmented_reward(self,
                                base_reward: float,
                                agent_action: int,
                                expert_action: int,
                                state_novelty: float,
                                expert_confidence: float = 1.0) -> float:
        """
        专家-学习者差异奖励
        Args:
            base_reward: 基础奖励
            agent_action: 智能体选择的动作
            expert_action: 专家推荐的动作
            state_novelty: 状态新颖度 [0, 1]
            expert_confidence: 专家对该动作的置信度 [0, 1]
        """
        reward = base_reward

        # 只在专家有信心时奖励一致性
        if expert_confidence > 0.7:
            if agent_action == expert_action:
                # 在熟悉状态下跟随专家
                consistency_bonus = 1.2 * (1.0 - state_novelty) * expert_confidence
                reward += consistency_bonus
            else:
                # 在高信心状态下偏离专家要轻微惩罚
                reward -= 0.3 * expert_confidence
        else:
            # 专家不确定时，鼓励探索
            if base_reward > 0 and state_novelty > 0.7:
                reward += 0.8  # 在新状态成功探索

        return self._normalize_reward(reward, clip_range=(-10.0, 8.0))

    # ----------------------------------------------------------------
    #  辅助方法
    # ----------------------------------------------------------------
    def _compute_failure_penalty(self, reason: Optional[str]) -> float:
        """根据失败原因返回差异化惩罚"""
        penalty_map = {
            "resource_exhausted": 2.0,  # 资源不足（可接受）
            "timeout": 3.0,  # 超时
            "routing_deadlock": 5.0,  # 路由死锁（严重）
            "qos_violation": 4.0,  # QoS违反
            "invalid_action": 6.0  # 非法动作
        }
        return penalty_map.get(reason, 4.0)

    def _compute_qos_penalty(self, violations: Dict[str, float]) -> float:
        """计算QoS违反的总惩罚"""
        penalty = 0.0
        weight_map = {
            "delay": 1.5,
            "bandwidth": 2.0,
            "jitter": 1.0,
            "packet_loss": 3.0
        }
        for metric, violation_ratio in violations.items():
            weight = weight_map.get(metric, 1.0)
            penalty += weight * violation_ratio
        return penalty

    def _get_adaptive_backup_penalty(self, backup_level: str) -> float:
        """
        自适应backup惩罚：基于历史成功率
        成功率越低，说明该级别越不靠谱，惩罚越重
        """
        base_penalty = {
            "primary": 0.0,
            "resource_aware": 0.3,
            "smart_greedy": 0.8,
            "minimal": 1.5,
            "never_fail": 2.5
        }

        # 根据历史成功率调整
        success_rate = self.backup_success_rate.get(backup_level, 0.5)
        adaptive_factor = 1.0 + (1.0 - success_rate)  # 成功率低则放大惩罚

        return base_penalty.get(backup_level, 1.0) * adaptive_factor

    def update_backup_stats(self, backup_level: str, success: bool):
        """更新backup策略统计（由环境调用）"""
        self.backup_usage_count[backup_level] += 1
        count = self.backup_usage_count[backup_level]

        # 指数移动平均
        alpha = 0.1
        if success:
            self.backup_success_rate[backup_level] = \
                alpha * 1.0 + (1 - alpha) * self.backup_success_rate[backup_level]
        else:
            self.backup_success_rate[backup_level] = \
                alpha * 0.0 + (1 - alpha) * self.backup_success_rate[backup_level]

    def _normalize_reward(self, reward: float,
                          clip_range: tuple = None) -> float:
        """奖励归一化和裁剪"""
        if clip_range is None:
            clip_range = (self.reward_min, self.reward_max)
        return float(np.clip(reward, clip_range[0], clip_range[1]))

    # ----------------------------------------------------------------
    # 主入口（环境调用）
    # ----------------------------------------------------------------
    def criticize(self,
                  # 低层参数
                  sub_task_completed: bool,
                  cost: float,
                  request_failed: bool,
                  progress_to_goal: float,
                  backup_used: bool,
                  backup_level: str,
                  qos_violations: Optional[Dict[str, float]] = None,
                  failure_reason: Optional[str] = None,
                  # DAgger参数
                  agent_action: int = -1,
                  expert_action: int = -1,
                  state_novelty: float = 0.5,
                  expert_confidence: float = 1.0) -> float:
        """
        统一奖励计算入口
        """
        # 计算基础低层奖励
        base_reward = self.low_level_reward(
            sub_task_completed=sub_task_completed,
            step_cost=cost,
            request_failed=request_failed,
            progress_to_goal=progress_to_goal,
            backup_used=backup_used,
            backup_level=backup_level,
            qos_violations=qos_violations,
            failure_reason=failure_reason
        )

        # 如果提供了专家动作，增加DAgger奖励
        if expert_action >= 0 and agent_action >= 0:
            final_reward = self.dagger_augmented_reward(
                base_reward=base_reward,
                agent_action=agent_action,
                expert_action=expert_action,
                state_novelty=state_novelty,
                expert_confidence=expert_confidence
            )
        else:
            final_reward = base_reward

        return final_reward

    def set_training_phase(self, phase: int, epoch: int = 0):
        """更新训练阶段（由训练循环调用）"""
        self.phase = phase
        self.epoch = epoch