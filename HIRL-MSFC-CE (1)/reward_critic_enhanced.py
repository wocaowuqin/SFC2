# reward_critic_enhanced_fixed.py
"""
修复版 HRL 奖励系统 for 多播 VNF 映射

修复内容:
1. 降低 Phase 1 失败惩罚 (8.0 -> 5.0)
2. 降低 Backup 惩罚，让 Agent 不害怕使用 backup
3. 添加 backup 成功奖励（而不是只惩罚）
4. 优化 progress shaping 的稳定性
5. 添加更多调试信息
"""

import logging
from typing import Dict, Optional, Tuple, Any
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class RewardCritic:
    """
    增强版 HRL 奖励系统 (修复版)
    """

    def __init__(self,
                 training_phase: int = 2,
                 epoch: int = 0,
                 max_epochs: int = 1200,
                 params: Optional[Dict[str, Any]] = None):

        self.phase = int(training_phase)
        self.epoch = int(epoch)
        self.max_epochs = int(max_epochs)

        # ============================================
        # 🔧 修复1: 调整默认参数
        # ============================================
        default_params = {
            # high-level
            "high_goal_reward": 10.0,
            "high_goal_partial": 5.0,
            "high_cost_weight": 2.5,
            "high_blocking_bonus": 1.0,
            "high_blocking_penalty": 2.0,

            # low-level
            "low_subtask_reward": 3.0,
            "low_phase1_subtask": 5.0,
            "low_progress_weight": 1.0,  # 🔧 从1.5降到1.0，减少噪声影响

            # piecewise cost weights
            "low_cost_weights": (0.3, 1.0, 2.0),  # 🔧 降低惩罚强度

            # 🔧 修复2: 大幅降低失败惩罚
            "low_request_failed_phase1_penalty": 5.0,  # 从8.0降到5.0
            "low_failure_base": 3.0,  # 从4.0降到3.0

            # 🔧 修复3: 大幅降低backup惩罚
            "backup_penalties": {
                "primary": 0.0,
                "resource_aware": 0.1,  # 从0.3降到0.1
                "smart_greedy": 0.2,  # 从0.8降到0.2
                "minimal": 0.4,  # 从1.5降到0.4
                "never_fail": 0.8,  # 从2.5降到0.8
                "unknown": 0.5  # 从1.5降到0.5
            },

            # 🔧 新增: backup成功奖励
            "backup_success_bonus": {
                "resource_aware": 0.3,
                "smart_greedy": 0.2,
                "minimal": 0.1,
                "never_fail": 0.05,
                "unknown": 0.1
            },

            # qos weights
            "qos_weights": {"delay": 1.5, "bandwidth": 2.0, "jitter": 1.0, "packet_loss": 3.0},

            # reward clipping range
            "reward_clip": (-8.0, 8.0),  # 🔧 收窄范围

            # dagger tuning
            "dagger_consistency_scale": 1.0,  # 🔧 从1.2降到1.0
            "dagger_divergence_penalty": 0.2,  # 🔧 从0.3降到0.2

            # smoothing alpha
            "backup_alpha": 0.1
        }

        self.params = default_params if params is None else {**default_params, **params}

        # discount factors
        self.gamma_high = 0.99
        self.gamma_low = 0.95

        # backup stats
        self.backup_success_rate = defaultdict(lambda: 0.5)
        self.backup_usage_count = defaultdict(int)

        # debug
        self.debug = False
        self.logger = logger

        # quick refs
        self.reward_min, self.reward_max = self.params["reward_clip"]

        # 🔧 新增: 奖励统计（用于调试）
        self.reward_history = []
        self.component_history = []

    # -------------------------
    # Public helper / config
    # -------------------------
    def set_training_phase(self, phase: int, epoch: int = 0, max_epochs: Optional[int] = None):
        self.phase = int(phase)
        self.epoch = int(epoch)
        if max_epochs is not None:
            self.max_epochs = int(max_epochs)

    def set_params(self, **kwargs):
        self.params.update(kwargs)
        if "reward_clip" in kwargs:
            self.reward_min, self.reward_max = self.params["reward_clip"]

    def set_debug(self, debug_on: bool):
        self.debug = bool(debug_on)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "epoch": self.epoch,
            "backup_success_rate": dict(self.backup_success_rate),
            "backup_usage_count": dict(self.backup_usage_count),
            "params_snapshot": {k: self.params[k] for k in ("reward_clip", "backup_penalties")},
            "reward_history_len": len(self.reward_history),
            "avg_reward": np.mean(self.reward_history[-100:]) if self.reward_history else 0.0
        }

    # -------------------------
    # Input sanitation
    # -------------------------
    def _clamp(self, x: Any, lo: float, hi: float) -> float:
        try:
            return float(np.clip(float(x), lo, hi))
        except Exception:
            return float(lo)

    # -------------------------
    # High-level reward
    # -------------------------
    def high_level_reward(self,
                          goal_reached: bool,
                          total_cost: float,
                          qos_satisfied: bool,
                          blocking_rate: float = 0.0) -> float:
        total_cost = self._clamp(total_cost, 0.0, 1.0)
        blocking_rate = self._clamp(blocking_rate, 0.0, 1.0)

        p = self.params
        reward = 0.0

        if goal_reached and qos_satisfied:
            reward += p["high_goal_reward"]
            reward += 2.0 * (1.0 - total_cost)
        elif goal_reached:
            reward += p["high_goal_partial"]

        reward -= p["high_cost_weight"] * total_cost

        if blocking_rate < 0.1:
            reward += p["high_blocking_bonus"]
        elif blocking_rate > 0.5:
            reward -= p["high_blocking_penalty"]

        return self._normalize_reward(reward)

    # -------------------------
    # 🔧 修复后的 Low-level reward
    # -------------------------
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
        修复后的低层奖励函数
        """
        # sanitize
        step_cost = self._clamp(step_cost, 0.0, 1.0)
        progress_to_goal = self._clamp(progress_to_goal, -1.0, 1.0)
        backup_level = str(backup_level) if backup_level is not None else "unknown"
        qos_violations = qos_violations or {}

        p = self.params
        reward = 0.0
        components = {}  # 用于调试

        # ============================================
        # Phase-aware base reward
        # ============================================
        if self.phase == 1:
            # Phase 1: 简单奖励，鼓励完成
            if sub_task_completed:
                reward += p["low_phase1_subtask"]
                components["subtask_bonus"] = p["low_phase1_subtask"]

            # 轻微的cost惩罚
            cost_penalty = 0.2 * step_cost
            reward -= cost_penalty
            components["cost_penalty"] = -cost_penalty

            if request_failed:
                # 🔧 修复: 降低失败惩罚
                fail_penalty = p["low_request_failed_phase1_penalty"]
                reward -= fail_penalty
                components["fail_penalty"] = -fail_penalty
        else:
            # Phase 2+: 更复杂的奖励
            if sub_task_completed:
                efficiency = max(0.0, 1.0 - step_cost)
                subtask_bonus = p["low_subtask_reward"] + efficiency
                reward += subtask_bonus
                components["subtask_bonus"] = subtask_bonus

            # 🔧 修复: 更稳定的 progress shaping
            # 只在progress明显时给奖励，避免噪声
            if abs(progress_to_goal) > 0.1:
                progress_reward = p["low_progress_weight"] * progress_to_goal
                reward += progress_reward
                components["progress"] = progress_reward

            # piecewise cost penalty
            w1, w2, w3 = p["low_cost_weights"]
            if step_cost < 0.3:
                cost_penalty = w1 * step_cost
            elif step_cost < 0.7:
                cost_penalty = w2 * step_cost
            else:
                cost_penalty = w3 * step_cost
            reward -= cost_penalty
            components["cost_penalty"] = -cost_penalty

            # failure penalty with decay
            if request_failed:
                base_penalty = self._compute_failure_penalty(failure_reason)
                decay_factor = max(0.3, 1.0 - (self.epoch / max(1, self.max_epochs)))
                fail_penalty = base_penalty * decay_factor
                reward -= fail_penalty
                components["fail_penalty"] = -fail_penalty

        # ============================================
        # 🔧 修复4: Backup 奖励/惩罚重新设计
        # ============================================
        if backup_used:
            if sub_task_completed:
                # Backup 成功：给小奖励而不是惩罚！
                bonus_map = p.get("backup_success_bonus", {})
                bonus = bonus_map.get(backup_level, 0.1)
                reward += bonus
                components["backup_bonus"] = bonus
            else:
                # Backup 也失败：轻微惩罚
                penalty = self._get_adaptive_backup_penalty(backup_level) * 0.5
                reward -= penalty
                components["backup_penalty"] = -penalty
        else:
            if sub_task_completed:
                # 主路径成功：奖励
                reward += 0.5
                components["primary_bonus"] = 0.5

        # QoS violations
        if qos_violations:
            qos_penalty = self._compute_qos_penalty(qos_violations)
            reward -= qos_penalty
            components["qos_penalty"] = -qos_penalty

        # 记录组件（调试用）
        if self.debug:
            self.component_history.append(components)

        return self._normalize_reward(reward, clip_range=(-8.0, 6.0))

    # -------------------------
    # DAgger augmentation
    # -------------------------
    def dagger_augmented_reward(self,
                                base_reward: float,
                                agent_action: int,
                                expert_action: int,
                                state_novelty: float,
                                expert_confidence: float = 1.0) -> float:
        state_novelty = self._clamp(state_novelty, 0.0, 1.0)
        expert_confidence = self._clamp(expert_confidence, 0.0, 1.0)

        reward = float(base_reward)
        p = self.params

        # 🔧 修复: 只在高置信度时使用DAgger
        if expert_confidence > 0.8:
            if agent_action == expert_action:
                consistency_bonus = p["dagger_consistency_scale"] * (1.0 - state_novelty) * expert_confidence
                reward += consistency_bonus
            else:
                # 🔧 修复: 降低分歧惩罚
                reward -= p["dagger_divergence_penalty"] * expert_confidence * 0.5
        elif base_reward > 0 and state_novelty > 0.7:
            # 鼓励新颖状态下的成功探索
            reward += 0.5

        return self._normalize_reward(reward, clip_range=(-8.0, 8.0))

    # -------------------------
    # helpers
    # -------------------------
    def _compute_failure_penalty(self, reason: Optional[str]) -> float:
        # 🔧 修复: 降低所有失败惩罚
        penalty_map = {
            "resource_exhausted": 1.5,  # 从2.0降
            "timeout": 2.0,  # 从3.0降
            "routing_deadlock": 3.0,  # 从5.0降
            "qos_violation": 2.5,  # 从4.0降
            "invalid_action": 4.0  # 从6.0降
        }
        return float(penalty_map.get(reason, self.params.get("low_failure_base", 3.0)))

    def _compute_qos_penalty(self, violations: Dict[str, float]) -> float:
        penalty = 0.0
        weight_map = self.params.get("qos_weights", {})
        for metric, ratio in violations.items():
            try:
                r = self._clamp(float(ratio), 0.0, 1.0)
            except Exception:
                r = 0.0
            weight = float(weight_map.get(metric, 1.0))
            penalty += weight * r
        return float(penalty)

    def _get_adaptive_backup_penalty(self, backup_level: str) -> float:
        base_penalty_map = self.params.get("backup_penalties", {})
        base_pen = float(base_penalty_map.get(backup_level, base_penalty_map.get("unknown", 0.5)))
        success_rate = float(np.clip(self.backup_success_rate.get(backup_level, 0.5), 0.0, 1.0))
        # 🔧 修复: 成功率高时减少惩罚
        adaptive_factor = 1.0 - 0.5 * success_rate
        return base_pen * adaptive_factor

    def update_backup_stats(self, backup_level: str, success: bool):
        alpha = float(self.params.get("backup_alpha", 0.1))
        prev = float(self.backup_success_rate.get(backup_level, 0.5))
        value = 1.0 if bool(success) else 0.0
        new = alpha * value + (1.0 - alpha) * prev
        self.backup_success_rate[backup_level] = float(new)
        self.backup_usage_count[backup_level] += 1

    # -------------------------
    # normalize / clip
    # -------------------------
    def _normalize_reward(self, reward: float, clip_range: Optional[Tuple[float, float]] = None,
                          scale_to_unit: bool = False) -> float:
        if clip_range is None:
            clip_range = (self.reward_min, self.reward_max)
        lo, hi = float(clip_range[0]), float(clip_range[1])
        r = float(np.clip(float(reward), lo, hi))

        # 记录历史
        self.reward_history.append(r)
        if len(self.reward_history) > 10000:
            self.reward_history = self.reward_history[-5000:]

        if scale_to_unit:
            if hi == lo:
                return r
            return 2.0 * (r - lo) / (hi - lo) - 1.0
        return r

    # -------------------------
    # main unified entry
    # -------------------------
    def criticize(self,
                  sub_task_completed: bool,
                  cost: float,
                  request_failed: bool,
                  progress_to_goal: float,
                  backup_used: bool,
                  backup_level: str,
                  qos_violations: Optional[Dict[str, float]] = None,
                  failure_reason: Optional[str] = None,
                  agent_action: int = -1,
                  expert_action: int = -1,
                  state_novelty: float = 0.5,
                  expert_confidence: float = 1.0) -> float:
        """
        统一入口
        """
        cost = self._clamp(cost, 0.0, 1.0)
        progress_to_goal = self._clamp(progress_to_goal, -1.0, 1.0)
        state_novelty = self._clamp(state_novelty, 0.0, 1.0)
        expert_confidence = self._clamp(expert_confidence, 0.0, 1.0)
        backup_level = str(backup_level) if backup_level is not None else "unknown"

        # compute base low-level reward
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

        # DAgger augmentation
        if expert_action >= 0 and agent_action >= 0:
            final_reward = self.dagger_augmented_reward(
                base_reward=base_reward,
                agent_action=int(agent_action),
                expert_action=int(expert_action),
                state_novelty=state_novelty,
                expert_confidence=expert_confidence
            )
        else:
            final_reward = base_reward

        # 更新 backup 统计
        if backup_used:
            self.update_backup_stats(backup_level, sub_task_completed)

        # debug logging
        if self.debug:
            try:
                self.logger.debug(f"RewardCritic: phase={self.phase}, "
                                  f"completed={sub_task_completed}, cost={cost:.3f}, "
                                  f"backup={backup_used}({backup_level}), "
                                  f"base={base_reward:.3f}, final={final_reward:.3f}")
            except Exception:
                pass

        return float(final_reward)

    # -------------------------
    # 🔧 新增: 诊断方法
    # -------------------------
    def get_reward_diagnostics(self) -> Dict[str, Any]:
        """返回奖励诊断信息"""
        if not self.reward_history:
            return {"status": "no_data"}

        recent = self.reward_history[-100:]
        return {
            "total_rewards": len(self.reward_history),
            "recent_mean": float(np.mean(recent)),
            "recent_std": float(np.std(recent)),
            "recent_min": float(np.min(recent)),
            "recent_max": float(np.max(recent)),
            "positive_ratio": float(np.mean([r > 0 for r in recent])),
            "backup_success_rates": dict(self.backup_success_rate),
            "backup_usage": dict(self.backup_usage_count)
        }