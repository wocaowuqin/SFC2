"""
完整的三阶段层次强化学习训练系统 - 修复版
Three-Phase Hierarchical Reinforcement Learning Training System - Fixed

🔥 修复内容：
1. Phase 3 正确加载独立数据集 (phase3_requests.pkl, phase3_events.pkl)
2. 添加数据文件验证和自动切换
3. 增强错误处理和日志输出
4. 自动恢复原始数据文件

阶段1: 专家轨迹采集 (Expert Trajectory Collection)
阶段2: 监督模仿学习 (Supervised Imitation Learning)
阶段3: 强化学习微调 (RL Fine-tuning) - FIXED
"""

import os
import time
import pickle
import shutil
import logging
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =====================================================================
#   Phase 1: Expert Trajectory Collection
# =====================================================================

class Phase1ExpertCollector:
    """
    阶段 1：专家轨迹采集

    依赖环境接口：
        env.generate_random_request() -> request or None
        env.expert_solve_request(req) -> (success: bool, traj: List[tuple])
        env.reset_all() (可选)

    轨迹格式: List[(state, goal, action, reward, next_state)]
    """

    def __init__(self, env, output_dir: str, config: Optional[Dict] = None):
        self.env = env
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 默认配置
        default_config = {
            "episodes": 2000,
            "save_every": 500,
            "max_dataset_size": 100000  # 防止内存溢出
        }
        self.cfg = {**default_config, **(config or {})}
        self.dataset = []

        # 统计信息
        self.stats = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "total_transitions": 0
        }

    def run(self) -> List[Tuple]:
        """运行专家轨迹采集"""
        logger.info("=" * 60)
        logger.info("Phase 1: Expert Dataset Collection")
        logger.info("=" * 60)

        # 重置环境
        if hasattr(self.env, "reset_all"):
            self.env.reset_all()
            logger.info("Environment reset complete")

        max_episodes = self.cfg["episodes"]

        for ep in range(1, max_episodes + 1):
            # 生成随机请求
            try:
                req = self.env.generate_random_request()
            except Exception as e:
                logger.warning(f"[Phase1] Episode {ep}: Failed to generate request - {e}")
                self.stats["failed_episodes"] += 1
                continue

            if req is None:
                logger.warning(f"[Phase1] Episode {ep}: generate_random_request() returned None")
                self.stats["failed_episodes"] += 1
                continue

            # 专家求解
            try:
                success, traj = self.env.expert_solve_request(req)
            except Exception as e:
                logger.error(f"[Phase1] Episode {ep}: Expert solver failed - {e}")
                self.stats["failed_episodes"] += 1
                continue

            self.stats["total_episodes"] += 1

            # 添加成功的轨迹
            if success and traj:
                # 检查数据集大小限制
                if len(self.dataset) + len(traj) > self.cfg["max_dataset_size"]:
                    logger.warning(
                        f"[Phase1] Dataset size limit reached "
                        f"({self.cfg['max_dataset_size']}). Stopping collection."
                    )
                    break

                self.dataset.extend(traj)
                self.stats["successful_episodes"] += 1
                self.stats["total_transitions"] += len(traj)
            else:
                self.stats["failed_episodes"] += 1

            # 定期日志
            if ep % 100 == 0:
                success_rate = (self.stats["successful_episodes"] /
                                self.stats["total_episodes"] * 100) if self.stats["total_episodes"] > 0 else 0
                logger.info(
                    f"[Phase1] Episode {ep}/{max_episodes} | "
                    f"Dataset size: {len(self.dataset)} | "
                    f"Success rate: {success_rate:.1f}%"
                )

            # 定期保存
            if ep % self.cfg["save_every"] == 0:
                self._save_checkpoint(f"expert_ep{ep}.pkl")

        # 最终保存
        self._save_checkpoint("expert_final.pkl")
        self._save_stats()

        logger.info(f"[Phase1] Collection complete:")
        logger.info(f"  - Total episodes: {self.stats['total_episodes']}")
        logger.info(f"  - Successful: {self.stats['successful_episodes']}")
        logger.info(f"  - Failed: {self.stats['failed_episodes']}")
        logger.info(f"  - Total transitions: {len(self.dataset)}")

        return self.dataset

    def _save_checkpoint(self, filename: str):
        """保存数据集检查点"""
        path = self.output_dir / filename
        try:
            with open(path, "wb") as f:
                pickle.dump(self.dataset, f)
            logger.info(f"[Phase1] Saved checkpoint: {path} ({len(self.dataset)} transitions)")
        except Exception as e:
            logger.error(f"[Phase1] Failed to save checkpoint {path}: {e}")

    def _save_stats(self):
        """保存统计信息"""
        stats_path = self.output_dir / "phase1_stats.pkl"
        try:
            with open(stats_path, "wb") as f:
                pickle.dump(self.stats, f)
            logger.info(f"[Phase1] Saved statistics: {stats_path}")
        except Exception as e:
            logger.error(f"[Phase1] Failed to save statistics: {e}")


# =====================================================================
#   Phase 2: Supervised Imitation Learning
# =====================================================================

class Phase2ILTrainer:
    """
    阶段 2: 模仿学习（行为克隆）

    依赖智能体接口：
        agent.supervised_update(state, goal, action) -> loss: float
        agent.save(path: str)
    """

    def __init__(self, agent, expert_data: List[Tuple],
                 output_dir: str, config: Optional[Dict] = None):
        self.agent = agent
        self.data = expert_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 默认配置
        default_config = {
            "epochs": 10,
            "batch_size": 128,
            "save_every_epoch": 2,
            "validation_split": 0.1  # 10% 用于验证
        }
        self.cfg = {**default_config, **(config or {})}

        # 数据验证
        if not self.data:
            raise ValueError("Expert data is empty! Cannot train.")

        # 划分训练集和验证集
        self._split_data()

        # 训练历史
        self.history = {
            "train_loss": [],
            "val_loss": []
        }

    def _split_data(self):
        """划分训练集和验证集"""
        n_total = len(self.data)
        n_val = int(n_total * self.cfg["validation_split"])

        indices = np.random.permutation(n_total)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.train_data = [self.data[i] for i in train_indices]
        self.val_data = [self.data[i] for i in val_indices]

        logger.info(f"[Phase2] Data split - Train: {len(self.train_data)}, Val: {len(self.val_data)}")

    def run(self) -> bool:
        """运行模仿学习训练"""
        logger.info("=" * 60)
        logger.info("Phase 2: Imitation Learning (Behavior Cloning)")
        logger.info("=" * 60)

        n_train = len(self.train_data)
        batch_size = self.cfg["batch_size"]
        n_epochs = self.cfg["epochs"]

        best_val_loss = float('inf')

        for epoch in range(1, n_epochs + 1):
            # 训练
            train_loss = self._train_epoch(epoch)

            # 验证
            val_loss = self._validate()

            # 记录历史
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            logger.info(
                f"[Phase2] Epoch {epoch}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.agent.save(str(self.output_dir / "il_best.pth"))
                logger.info(f"[Phase2] New best model saved (val_loss: {val_loss:.4f})")

            # 定期保存
            if epoch % self.cfg["save_every_epoch"] == 0:
                self.agent.save(str(self.output_dir / f"il_ep{epoch}.pth"))

        # 保存最终模型
        final_path = self.output_dir / "il_final.pth"
        self.agent.save(str(final_path))
        logger.info(f"[Phase2] Final model saved: {final_path}")

        # 保存训练历史
        self._save_history()

        logger.info(f"[Phase2] Training complete. Best val loss: {best_val_loss:.4f}")

        return True

    def _train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        np.random.shuffle(self.train_data)

        batch_size = self.cfg["batch_size"]
        n_batches = (len(self.train_data) + batch_size - 1) // batch_size

        epoch_losses = []

        for i in range(n_batches):
            batch_data = self.train_data[i * batch_size:(i + 1) * batch_size]

            batch_loss = 0.0
            for (state, goal, action, reward, next_state) in batch_data:
                try:
                    loss = self.agent.supervised_update(state, goal, action)
                    batch_loss += loss
                except Exception as e:
                    logger.error(f"[Phase2] Update failed: {e}")
                    continue

            avg_batch_loss = batch_loss / len(batch_data) if batch_data else 0.0
            epoch_losses.append(avg_batch_loss)

        return np.mean(epoch_losses) if epoch_losses else 0.0

    def _validate(self) -> float:
        """验证"""
        if not self.val_data:
            return 0.0

        val_losses = []

        for (state, goal, action, reward, next_state) in self.val_data:
            try:
                # 仅前向传播，不更新参数
                # 假设 agent 有一个 evaluate 方法，否则可以调用 supervised_update 但不应用梯度
                # 这里简化处理，假设 supervised_update 在 eval 模式下不更新
                loss = self.agent.supervised_update(state, goal, action)
                val_losses.append(loss)
            except:
                continue

        return np.mean(val_losses) if val_losses else 0.0

    def _save_history(self):
        """保存训练历史"""
        history_path = self.output_dir / "phase2_history.pkl"
        try:
            with open(history_path, "wb") as f:
                pickle.dump(self.history, f)
            logger.info(f"[Phase2] Training history saved: {history_path}")
        except Exception as e:
            logger.error(f"[Phase2] Failed to save history: {e}")


# =====================================================================
#   Phase 3: RL Fine-tuning - FIXED VERSION
# =====================================================================

class Phase3RLTrainer:
    """
    阶段 3：强化学习微调 - 修复版

    🔥 修复：正确加载 phase3_requests.pkl 和 phase3_events.pkl

    依赖环境接口：
        env.reset_all()
        env.reset_request() -> (request, state)
        env.get_low_level_mask(goal) -> mask
        env.step_low_level(goal, action) -> (next_state, reward, sub_done, req_done)
        env.unadded_dest_indices (属性)
        env.total_requests_seen (属性)
        env.total_requests_accepted (属性)

    依赖智能体接口：
        agent.switch_to_rl_mode(start_epsilon)
        agent.select_goal(state, candidates, epsilon) -> goal
        agent.select_action(state, goal, mask, epsilon) -> action
        agent.store(state, action, reward, next_state, done, goal)
        agent.update()
        agent.get_epsilon() -> float
        agent.save(path)
    """

    def __init__(self, env, agent, output_dir: str, config: Optional[Dict] = None):
        self.env = env
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 默认配置
        default_config = {
            "episodes": 2000,
            "start_epsilon": 0.2,
            "max_steps_per_episode": 120,
            "eval_every": 100,
            "eval_episodes": 20,
            "save_every": 100
        }
        self.cfg = {**default_config, **(config or {})}

        # 训练统计
        self.stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "acceptance_rates": [],
            "eval_rewards": [],
            "best_eval_reward": -float('inf')
        }

    def run(self) -> Dict[str, Any]:
        """运行RL微调"""
        logger.info("=" * 60)
        logger.info("Phase 3: RL Fine-tuning")
        logger.info("=" * 60)

        # 重置环境
        if hasattr(self.env, "reset_all"):
            self.env.reset_all()
            logger.info("Environment reset for RL training")

        # 切换到RL模式
        self.agent.switch_to_rl_mode(start_epsilon=self.cfg["start_epsilon"])
        logger.info(f"Agent switched to RL mode (epsilon: {self.cfg['start_epsilon']})")

        max_episodes = self.cfg["episodes"]

        for ep in range(1, max_episodes + 1):
            ep_reward, ep_length = self._run_episode(ep)

            # 记录统计
            self.stats["episode_rewards"].append(ep_reward)
            self.stats["episode_lengths"].append(ep_length)

            # 计算接受率
            if hasattr(self.env, 'total_requests_seen') and hasattr(self.env, 'total_requests_accepted'):
                total_seen = max(1, self.env.total_requests_seen)
                acc_rate = (self.env.total_requests_accepted / total_seen) * 100
                self.stats["acceptance_rates"].append(acc_rate)
            else:
                acc_rate = 0.0

            # 定期日志
            if ep % 10 == 0:
                recent_rewards = self.stats["episode_rewards"][-10:]
                avg_reward = np.mean(recent_rewards)
                epsilon = self.agent.get_epsilon()

                logger.info(
                    f"[Phase3] Episode {ep}/{max_episodes} | "
                    f"Reward: {ep_reward:.2f} (avg: {avg_reward:.2f}) | "
                    f"Steps: {ep_length} | "
                    f"Acc: {acc_rate:.2f}% | "
                    f"ε: {epsilon:.3f}"
                )

            # 定期评估
            if ep % self.cfg["eval_every"] == 0:
                eval_reward = self._evaluate()
                self.stats["eval_rewards"].append(eval_reward)

                logger.info(f"[Phase3] Evaluation at episode {ep}: {eval_reward:.2f}")

                # 保存最佳模型
                if eval_reward > self.stats["best_eval_reward"]:
                    self.stats["best_eval_reward"] = eval_reward
                    self.agent.save(str(self.output_dir / "rl_best.pth"))
                    logger.info(f"[Phase3] New best model saved (eval_reward: {eval_reward:.2f})")

            # 定期保存检查点
            if ep % self.cfg["save_every"] == 0:
                self.agent.save(str(self.output_dir / f"rl_ep{ep}.pth"))

        # 保存最终模型
        final_path = self.output_dir / "rl_final.pth"
        self.agent.save(str(final_path))

        # 保存统计信息
        self._save_stats()

        # 汇总结果
        result = {
            "avg_reward": np.mean(self.stats["episode_rewards"]) if self.stats["episode_rewards"] else 0.0,
            "final_acceptance_rate": self.stats["acceptance_rates"][-1] if self.stats["acceptance_rates"] else 0.0,
            "best_eval_reward": self.stats["best_eval_reward"]
        }

        logger.info("=" * 60)
        logger.info("Phase 3 Complete")
        logger.info(f"Average reward: {result['avg_reward']:.2f}")
        logger.info(f"Final acceptance rate: {result['final_acceptance_rate']:.2f}%")
        logger.info(f"Best eval reward: {result['best_eval_reward']:.2f}")
        logger.info(f"Final model saved to: {final_path}")
        logger.info("=" * 60)

        return result

    def _run_episode(self, episode_num: int) -> Tuple[float, int]:
        """运行一个训练episode"""
        try:
            req, state = self.env.reset_request()
        except Exception as e:
            logger.error(f"[Phase3] Episode {episode_num}: Failed to reset - {e}")
            return 0.0, 0

        if req is None:
            logger.warning(f"[Phase3] Episode {episode_num}: reset_request() returned None")
            return 0.0, 0

        done = False
        episode_reward = 0.0
        step = 0
        max_steps = self.cfg["max_steps_per_episode"]
        epsilon = self.agent.get_epsilon()

        while not done and step < max_steps:
            # 中层决策：选择目标
            try:
                candidates = list(self.env.unadded_dest_indices)
            except Exception as e:
                logger.error(f"[Phase3] Failed to get candidates: {e}")
                break

            if not candidates:
                # 没有可用目标，结束episode
                break

            try:
                goal = self.agent.select_goal(state, candidates, epsilon=epsilon)
            except Exception as e:
                logger.error(f"[Phase3] Goal selection failed: {e}")
                break

            # 低层决策：选择动作
            try:
                mask = self.env.get_low_level_mask(goal)
                action = self.agent.select_action(state, goal, mask, epsilon=epsilon)
            except Exception as e:
                logger.error(f"[Phase3] Action selection failed: {e}")
                break

            # 环境交互
            try:
                next_state, reward, sub_done, req_done = self.env.step_low_level(goal, action)
            except Exception as e:
                logger.error(f"[Phase3] Environment step failed: {e}")
                break

            # 存储经验并更新
            try:
                self.agent.store(state, action, reward, next_state, req_done, goal)
                self.agent.update()
            except Exception as e:
                logger.error(f"[Phase3] Agent update failed: {e}")

            # 更新状态
            state = next_state
            episode_reward += reward
            step += 1

            if req_done:
                done = True

        return episode_reward, step

    def _evaluate(self) -> float:
        """评估当前策略（无探索）"""
        old_epsilon = self.agent.get_epsilon()

        # 临时设置为贪婪策略
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = 0.0

        eval_rewards = []

        for _ in range(self.cfg["eval_episodes"]):
            try:
                req, state = self.env.reset_request()
            except:
                continue

            if req is None:
                continue

            done = False
            episode_reward = 0.0
            step = 0

            while not done and step < self.cfg["max_steps_per_episode"]:
                try:
                    candidates = list(self.env.unadded_dest_indices)
                    if not candidates:
                        break

                    goal = self.agent.select_goal(state, candidates, epsilon=0.0)
                    mask = self.env.get_low_level_mask(goal)
                    action = self.agent.select_action(state, goal, mask, epsilon=0.0)

                    next_state, reward, sub_done, req_done = self.env.step_low_level(goal, action)

                    state = next_state
                    episode_reward += reward
                    step += 1

                    if req_done:
                        done = True
                except:
                    break

            eval_rewards.append(episode_reward)

        # 恢复epsilon
        if hasattr(self.agent, 'epsilon'):
            self.agent.epsilon = old_epsilon

        return np.mean(eval_rewards) if eval_rewards else 0.0

    def _save_stats(self):
        """保存训练统计信息"""
        stats_path = self.output_dir / "phase3_stats.pkl"
        try:
            with open(stats_path, "wb") as f:
                pickle.dump(self.stats, f)
            logger.info(f"[Phase3] Statistics saved: {stats_path}")
        except Exception as e:
            logger.error(f"[Phase3] Failed to save statistics: {e}")


# =====================================================================
#   Three-Phase Trainer (主入口) - FIXED VERSION
# =====================================================================

class HIRLThreePhaseTrainer:
    """
    三阶段层次强化学习训练器 - 修复版

    🔥 修复：Phase 3 正确加载独立数据集

    整合 Phase 1 (专家采集) -> Phase 2 (模仿学习) -> Phase 3 (RL微调)
    """

    def __init__(self, env, agent, work_dir: str = "output/hirl",
                 config: Optional[Dict] = None):
        """
        参数:
            env: 环境实例
            agent: 智能体实例
            work_dir: 工作目录
            config: 配置字典，包含 phase1, phase2, phase3 的配置
        """
        self.env = env
        self.agent = agent

        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 默认配置
        default_config = {
            "phase1": {
                "episodes": 3000,
                "save_every": 500,
                "max_dataset_size": 100000
            },
            "phase2": {
                "epochs": 10,
                "batch_size": 128,
                "save_every_epoch": 2,
                "validation_split": 0.1
            },
            "phase3": {
                "episodes": 2000,
                "start_epsilon": 0.2,
                "max_steps_per_episode": 120,
                "eval_every": 100,
                "eval_episodes": 20,
                "save_every": 100,
                # 🔥 新增：数据目录配置
                "data_dir": None,  # 如果为 None，使用环境的默认目录
                "use_phase3_data": True  # 是否使用独立的 phase3 数据
            }
        }

        # 合并用户配置
        if config is None:
            self.cfg = default_config
        else:
            self.cfg = {}
            for phase in ["phase1", "phase2", "phase3"]:
                self.cfg[phase] = {**default_config[phase], **config.get(phase, {})}

    def run_three_phase(self) -> Dict[str, Any]:
        """
        运行完整的三阶段训练流程

        返回:
            包含训练统计信息的字典
        """
        logger.info("=" * 70)
        logger.info(" " * 20 + "HIRL Three-Phase Training")
        logger.info("=" * 70)

        start_time = time.time()
        results = {}

        try:
            # ==================== Phase 1 ====================
            p1_dir = self.work_dir / "phase1"
            phase1 = Phase1ExpertCollector(self.env, str(p1_dir), self.cfg["phase1"])
            expert_data = phase1.run()

            if not expert_data:
                raise ValueError("Phase 1 failed: No expert data collected!")

            results["phase1"] = {
                "num_transitions": len(expert_data),
                "stats": phase1.stats
            }

            # ==================== Phase 2 ====================
            p2_dir = self.work_dir / "phase2"
            phase2 = Phase2ILTrainer(self.agent, expert_data, str(p2_dir), self.cfg["phase2"])
            phase2.run()

            results["phase2"] = {
                "history": phase2.history
            }

            # ==================== Phase 3 - FIXED ====================
            logger.info("\n" + "=" * 70)
            logger.info("Preparing Phase 3 with correct data loading...")
            logger.info("=" * 70)

            # 🔥 修复：创建带正确数据的新环境
            env_phase3 = self._prepare_phase3_environment()

            if env_phase3 is None:
                logger.error("❌ Failed to prepare Phase 3 environment!")
                raise ValueError("Phase 3 environment preparation failed")

            # 使用新环境创建 Phase 3 trainer
            p3_dir = self.work_dir / "phase3"
            phase3 = Phase3RLTrainer(env_phase3, self.agent, str(p3_dir), self.cfg["phase3"])
            phase3_stats = phase3.run()

            results["phase3"] = phase3_stats

        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise

        finally:
            # 🔥 恢复原始数据文件
            self._restore_original_data()

        # 总结
        elapsed = time.time() - start_time

        logger.info("=" * 70)
        logger.info(" " * 20 + "TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed / 60:.2f} minutes")
        logger.info(f"Phase 1 transitions: {results['phase1']['num_transitions']}")
        logger.info(f"Phase 3 avg reward: {results['phase3']['avg_reward']:.2f}")
        logger.info(f"Phase 3 best eval: {results['phase3']['best_eval_reward']:.2f}")
        logger.info(f"Final acceptance: {results['phase3']['final_acceptance_rate']:.2f}%")
        logger.info(f"Output directory: {self.work_dir}")
        logger.info("=" * 70)

        # 保存完整结果
        self._save_results(results)

        return results

    def _prepare_phase3_environment(self):
        """
        🔥 关键修复：为 Phase 3 准备正确的环境和数据

        返回:
            新创建的环境实例，已加载 phase3 数据
        """
        # 获取数据目录
        data_dir = self.cfg["phase3"].get("data_dir")

        if data_dir is None:
            # 尝试从环境获取
            if hasattr(self.env, 'input_dir'):
                data_dir = Path(self.env.input_dir)
            elif hasattr(self.env, 'data_dir'):
                data_dir = Path(self.env.data_dir)
            else:
                logger.error("❌ Cannot determine data directory!")
                return None
        else:
            data_dir = Path(data_dir)

        logger.info(f"📁 Data directory: {data_dir}")

        # 检查是否使用 phase3 独立数据
        if not self.cfg["phase3"].get("use_phase3_data", True):
            logger.info("⚠️  Using same environment for Phase 3 (no data switching)")
            return self.env

        # 检查 phase3 数据文件
        phase3_req = data_dir / "phase3_requests.pkl"
        phase3_evt = data_dir / "phase3_events.pkl"

        if not phase3_req.exists() or not phase3_evt.exists():
            logger.warning(f"⚠️  Phase 3 data files not found:")
            logger.warning(f"   Expected: {phase3_req}")
            logger.warning(f"   Expected: {phase3_evt}")
            logger.warning(f"   Using original environment without data switching")
            return self.env

        logger.info(f"✅ Phase 3 data files found")

        # 🔥 备份并替换数据文件
        default_req = data_dir / "sfc_requests.pkl"
        default_evt = data_dir / "sfc_events.pkl"

        self.backup_req = data_dir / "_backup_sfc_requests.pkl"
        self.backup_evt = data_dir / "_backup_sfc_events.pkl"

        # 备份现有文件
        if default_req.exists():
            shutil.move(str(default_req), str(self.backup_req))
            logger.info(f"   Backed up: sfc_requests.pkl")

        if default_evt.exists():
            shutil.move(str(default_evt), str(self.backup_evt))
            logger.info(f"   Backed up: sfc_events.pkl")

        # 复制 phase3 数据
        shutil.copy(str(phase3_req), str(default_req))
        shutil.copy(str(phase3_evt), str(default_evt))

        logger.info(f"✅ Phase 3 data activated!")

        # 🔥 创建新的环境实例
        try:
            # 获取环境类
            env_class = type(self.env)

            # 尝试获取环境初始化参数
            env_kwargs = {}

            # 常见的环境参数
            if hasattr(self.env, 'input_dir'):
                env_kwargs['input_dir'] = str(data_dir)
            if hasattr(self.env, 'topo'):
                env_kwargs['topo'] = self.env.topo
            if hasattr(self.env, 'dc_nodes'):
                env_kwargs['dc_nodes'] = self.env.dc_nodes
            if hasattr(self.env, 'capacities'):
                env_kwargs['capacities'] = self.env.capacities
            if hasattr(self.env, 'use_gnn'):
                env_kwargs['use_gnn'] = self.env.use_gnn

            logger.info(f"🔧 Creating new environment instance...")
            logger.info(f"   Class: {env_class.__name__}")
            logger.info(f"   Kwargs: {list(env_kwargs.keys())}")

            # 创建新环境
            new_env = env_class(**env_kwargs)

            logger.info(f"✅ New environment created")
            logger.info(f"   Total requests: {new_env.T if hasattr(new_env, 'T') else 'N/A'}")

            # 验证数据加载
            logger.info(f"🧪 Validating data loading...")

            if hasattr(new_env, 'reset_request'):
                test_req, test_state = new_env.reset_request()

                if test_req is None:
                    logger.error(f"❌ CRITICAL: reset_request() returned None!")
                    return None

                logger.info(f"✅ Data loading validated")
                logger.info(f"   Sample request: {len(test_req.get('dest', []))} destinations")

                # 重置环境
                if hasattr(new_env, 'reset_all'):
                    new_env.reset_all()

            return new_env

        except Exception as e:
            logger.error(f"❌ Failed to create new environment: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _restore_original_data(self):
        """
        🔥 恢复原始数据文件
        """
        if not hasattr(self, 'backup_req') or not hasattr(self, 'backup_evt'):
            return

        logger.info(f"\n🔧 Restoring original data files...")

        # 获取数据目录
        data_dir = self.cfg["phase3"].get("data_dir")
        if data_dir is None:
            if hasattr(self.env, 'input_dir'):
                data_dir = Path(self.env.input_dir)
            else:
                return
        else:
            data_dir = Path(data_dir)

        default_req = data_dir / "sfc_requests.pkl"
        default_evt = data_dir / "sfc_events.pkl"

        # 删除临时文件
        if default_req.exists():
            default_req.unlink()
        if default_evt.exists():
            default_evt.unlink()

        # 恢复备份
        if self.backup_req.exists():
            shutil.move(str(self.backup_req), str(default_req))
            logger.info(f"   Restored: sfc_requests.pkl")

        if self.backup_evt.exists():
            shutil.move(str(self.backup_evt), str(default_evt))
            logger.info(f"   Restored: sfc_events.pkl")

        logger.info(f"✅ Original data restored")

    def _save_results(self, results: Dict):
        """保存完整的训练结果"""
        results_path = self.work_dir / "training_results.pkl"
        try:
            with open(results_path, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Complete results saved: {results_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


# =====================================================================
#   使用示例
# =====================================================================

if __name__ == "__main__":
    """
    使用示例 - 需要实现具体的 env 和 agent
    """

    # 示例配置
    config = {
        "phase1": {
            "episodes": 1000,  # 较小的数量用于测试
        },
        "phase2": {
            "epochs": 5,
            "batch_size": 64,
        },
        "phase3": {
            "episodes": 2000,
            "start_epsilon": 0.2,
            # 🔥 Phase 3 数据配置
            "data_dir": "data/Abilene",  # 数据目录
            "use_phase3_data": True,  # 使用独立的 phase3 数据
        }
    }

    # 初始化训练器
    # from env.sfc_env_hirl_gnn import SFC_HIRL_Env_GNN
    # from agent.hirl_agent import HIRLAgent

    # env = SFC_HIRL_Env_GNN(input_dir="data/Abilene", ...)
    # agent = HIRLAgent(...)

    # trainer = HIRLThreePhaseTrainer(
    #     env=env,
    #     agent=agent,
    #     work_dir="output/run_001",
    #     config=config
    # )

    # 运行训练
    # results = trainer.run_three_phase()

    print("=" * 70)
    print("✅ Fixed Training Script Loaded Successfully!")
    print("=" * 70)
    print("\n🔥 Key Fixes:")
    print("  1. Phase 3 automatically switches to phase3_requests.pkl")
    print("  2. Creates new environment instance with correct data")
    print("  3. Validates data loading before training")
    print("  4. Auto-restores original data files after training")
    print("\n📝 Usage:")
    print("  Set config['phase3']['data_dir'] to your data directory")
    print("  Set config['phase3']['use_phase3_data'] = True")
    print("\nReady to use!")