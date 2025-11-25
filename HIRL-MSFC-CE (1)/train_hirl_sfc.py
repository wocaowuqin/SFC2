#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_hirl_sfc_refactored.py
重构后的训练主文件 — 兼容性更好、异常更稳健。
功能亮点：
 - 低层 update() 兼容封装（如果 Agent_SFC 没有 update，则尝试 train_from_memory / train）
 - 预训练 & 正式训练均有专家失败保护、超时回滚、日志清晰
 - DAgger 风格的专家依赖衰减
 - 更稳健的绘图与保存（避免 IPython/matplotlib 在无 GUI 环境的问题）
 - 统计阻塞率/失败数（若 env 支持则使用）
"""
from __future__ import annotations
import os
import sys
import logging
import traceback
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# 非交互式后端，避免 IPython 相关错误
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# TensorFlow / Keras 工具
from tensorflow.keras.utils import to_categorical

# 项目模块（保持原样）
import hyperparameters as H
from hirl_sfc_env import SFC_HIRL_Env
from hirl_sfc_models import MetaControllerNN, Hdqn_SFC
from hirl_sfc_agent import Agent_SFC

# Experience 命名元组（与原版兼容）
ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])

# 日志
logger = logging.getLogger(__name__)


# ------------------------
# 兼容性 & 工具函数
# ------------------------
def safe_update_agent(agent, t: int = None):
    """
    尝试兼容地调用 agent 的更新函数：
    优先尝试 update(t)，再尝试 train_from_memory / train / train_step 等常见名字。
    返回 (loss, avgQ, avgTD) 或 None。
    """
    try_names = ['update', 'train_from_memory', 'train', 'train_step']
    for name in try_names:
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                # 有些实现期望参数 t，有些不
                import inspect
                sig = inspect.signature(fn)
                if len(sig.parameters) == 0:
                    return fn()
                else:
                    return fn(t)
            except Exception as e:
                logger.exception("调用 %s 失败: %s", name, e)
                # 尝试下一种
                continue
    logger.warning("未找到兼容的训练调用 (update/train_from_memory/train)，跳过本次训练调用")
    return None


def safe_store_and_maybe_train(agent, exp: ActorExperience, step_count: int):
    """
    存储经验并按 agent.trainFreq 触发训练（兼容 agent 字段名差异）。
    """
    try:
        if hasattr(agent, 'store'):
            agent.store(exp)
        elif hasattr(agent, 'memory_push'):
            agent.memory_push(exp)
        else:
            logger.debug("agent 无显式 store 方法，跳过存储")
    except Exception as e:
        logger.exception("存储经验失败: %s", e)

    train_freq = getattr(agent, 'trainFreq', getattr(agent, 'train_freq', getattr(agent, 'trainFreq', None)))
    if train_freq is None:
        # 如果 agent 没有 trainFreq，仍可以每 step 调用一次 safe_update
        safe_update_agent(agent, step_count)
    else:
        if train_freq > 0 and (step_count % train_freq == 0):
            safe_update_agent(agent, step_count)


def save_training_plots_from_df(df: pd.DataFrame, out_dir: Path):
    """
    使用面向对象 matplotlib API 生成关键图表并保存
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        def smoothed(values, window=100):
            if len(values) == 0:
                return values
            window = min(window, max(1, len(values)))
            return pd.Series(values).rolling(window, min_periods=1).mean().values

        def plot_column(col, title, ylabel, smooth=False):
            if col not in df.columns:
                logger.debug("CSV 未包含列 %s，跳过", col)
                return
            fig = Figure(figsize=(9, 4))
            ax = fig.add_subplot(1, 1, 1)
            data = df[col].values
            ax.plot(data, alpha=0.4, label='raw')
            if smooth:
                ax.plot(smoothed(data), label='smooth', linewidth=2)
                ax.legend()
            ax.set_title(title)
            ax.set_xlabel("record idx")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fname = out_dir / f"{col}.png"
            canvas = FigureCanvas(fig)
            canvas.print_figure(str(fname), dpi=150)
            logger.info("Saved plot %s", fname)

        # 关键图
        plot_column('reward', 'Episode Reward', 'Reward', smooth=True)
        plot_column('acceptance_rate', 'Request Acceptance Rate', '%', smooth=False)
        plot_column('blocking_rate', 'Blocking Rate', '%', smooth=False)
        plot_column('avg_cpu_util', 'Avg CPU Utilization', '%', smooth=True)
        plot_column('avg_mem_util', 'Avg MEM Utilization', '%', smooth=True)
        plot_column('avg_bw_util', 'Avg BW Utilization', '%', smooth=True)
    except Exception:
        logger.exception("绘图失败")


def save_tracking_data(tracking_data: dict, out_dir: Path):
    """保存 tracking_data 并画图"""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(tracking_data)
        csv_path = out_dir / "training_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved tracking CSV to %s", csv_path)
        save_training_plots_from_df(df, out_dir)
    except Exception:
        logger.exception("保存训练数据失败")


# ------------------------
# 主训练函数
# ------------------------
def main():
    # 基本检查
    if not H.INPUT_DIR.exists():
        print(f"输入数据目录不存在: {H.INPUT_DIR}")
        return

    # 日志配置（文件 + 控制台）
    H.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            logging.FileHandler(H.OUTPUT_DIR / "training.log", encoding='utf-8'),
                            logging.StreamHandler(sys.stdout)
                        ])
    global logger
    logger = logging.getLogger(__name__)

    logger.info("启动 HIRL-SFC 训练（重构版）")

    # 初始化 tracking
    tracking_data = {
        'episode': [], 'reward': [],
        'acceptance_rate': [], 'blocking_rate': [],
        'avg_cpu_util': [], 'avg_mem_util': [], 'avg_bw_util': []
    }

    # =========================
    # 初始化 env / nets / agents
    # =========================
    env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, H.CAPACITIES)
    STATE_SHAPE = env.observation_space.shape
    NB_GOALS = env.NB_HIGH_LEVEL_GOALS
    NB_ACTIONS = env.NB_LOW_LEVEL_ACTIONS

    # 高层元控制器 (same interface)
    metacontroller = MetaControllerNN(state_shape=STATE_SHAPE, n_goals=NB_GOALS, lr=H.LR)
    hdqn_net = Hdqn_SFC(state_shape=STATE_SHAPE, n_goals=NB_GOALS, n_actions=NB_ACTIONS, lr=H.LR)

    low_level_agent = Agent_SFC(
        net=hdqn_net,
        n_actions=NB_ACTIONS,
        mem_cap=H.EXP_MEMORY,
        exploration_steps=H.EXPLORATION_STEPS,
        train_freq=H.TRAIN_FREQ,
        hard_update=H.HARD_UPDATE_FREQUENCY,
        n_samples=H.BATCH_SIZE,
        gamma=H.GAMMA
    )

    # 兼容：若 agent 有 compile 方法调用一下（例如在 Keras 中构建模型）
    if hasattr(low_level_agent, "compile") and callable(low_level_agent.compile):
        try:
            low_level_agent.compile()
        except Exception:
            logger.exception("low_level_agent.compile() 出错，继续运行")

    # 记录基本信息
    logger.info("STATE_SHAPE=%s NB_GOALS=%d NB_ACTIONS=%d", STATE_SHAPE, NB_GOALS, NB_ACTIONS)

    # ================
    # 阶段1：模仿学习预训练（safety hardened）
    # ================
    logger.info("阶段1：模仿学习预训练，步数=%d", H.PRE_TRAIN_STEPS)
    t = 0
    failed_requests = 0
    current_request, high_level_state = env.reset_request()
    # safety: 如果 reset_request 一直返回 None，则提前退出
    if current_request is None:
        logger.warning("环境在开始时没有请求，退出预训练")
    while t < H.PRE_TRAIN_STEPS and current_request is not None:
        try:
            # 获取专家轨迹（专家可能抛出）
            try:
                _, expert_traj = env.expert.solve_request_for_expert(current_request, env._get_network_state_dict())
            except Exception as e:
                logger.exception("专家求解异常，跳过该请求")
                failed_requests += 1
                # 记录失败并推进计数，防止死循环
                t += 1
                current_request, high_level_state = env.reset_request()
                continue

            if not expert_traj:
                failed_requests += 1
                logger.warning("[预训练] 专家未给出轨迹，跳过请求 (累计失败 %d)", failed_requests)
                t += 1  # 保证计数推进，防止卡住
                current_request, high_level_state = env.reset_request()
                continue

            # 遍历专家轨迹并让低层学习
            for (exp_goal, exp_action_tuple, exp_cost) in expert_traj:
                if exp_goal not in env.unadded_dest_indices:
                    # 目标已完成或失效，跳过
                    continue

                # 从动作 tuple 恢复成单 int（与原实现一致）
                exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

                # 在 env 上执行专家的低层动作（实际会改变 env 状态）
                next_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

                # 计算监督 reward（让 agent 学专家完成子任务）
                try:
                    reward = low_level_agent.criticize(sub_task_completed=True, cost=cost, request_failed=False)
                except Exception:
                    # 如果 agent 没有 criticize，使用简单替代
                    reward = 1.0 if sub_task_done else -1.0

                goal_one_hot = to_categorical(exp_goal, num_classes=NB_GOALS)
                experience = ActorExperience(high_level_state, goal_one_hot, exp_action, reward, next_state, sub_task_done)

                # 存储并根据 trainFreq 触发训练（内部兼容）
                safe_store_and_maybe_train(low_level_agent, experience, t)

                # metacontroller 收集专家高层标签（用于 DAgger）
                try:
                    metacontroller.collect(high_level_state, exp_goal)
                    if t % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                        metacontroller.train()
                except Exception:
                    logger.debug("metacontroller collect/train 出错，忽略")

                high_level_state = next_state
                t += 1
                if t >= H.PRE_TRAIN_STEPS:
                    break
                if request_done:
                    break

            # 请求结束后获取下一个请求
            current_request, high_level_state = env.reset_request()

        except Exception as e:
            logger.exception("预训练循环出错，跳过当前请求并继续")
            t += 1
            current_request, high_level_state = env.reset_request()

    logger.info("预训练结束：步数=%d, 失败请求=%d", t, failed_requests)

    # ================
    # 阶段2：混合 IL / RL 训练
    # ================
    logger.info("阶段2：混合 IL/RL 训练开始")
    step_count = t
    episode_count = 0
    low_level_agent.randomPlay = False if hasattr(low_level_agent, 'randomPlay') else False

    # DAgger 衰减参数（高层逐渐减小对专家的依赖）
    dagger_total = max(1, H.STEPS_LIMIT)
    dagger_initial = 1.0
    dagger_final = 0.05

    while episode_count < H.EPISODE_LIMIT and step_count < H.STEPS_LIMIT and env.t < env.T:
        try:
            current_request, high_level_state = env.reset_request()
            if current_request is None:
                logger.info("没有更多请求，训练结束")
                break

            episode_reward = 0.0
            episode_steps = 0
            request_done = False

            while not request_done:
                # 高层决策（元控制器）
                high_state_v = np.reshape(high_level_state, (1, -1))
                try:
                    goal_probs = metacontroller.predict(high_state_v)
                    agent_chosen_goal = metacontroller.sample(goal_probs)
                except Exception:
                    # 如果元控制器崩溃，fallback 为随机或第一个未完成目的
                    valid_goals = list(env.unadded_dest_indices) if env.unadded_dest_indices else [0]
                    agent_chosen_goal = valid_goals[0]

                # 计算专家真实目标并做 DAgger 决策（decide whether to use expert's label）
                true_goal = env.get_expert_high_level_goal(high_state_v)
                # β 衰减（线性）
                beta = dagger_final + (dagger_initial - dagger_final) * max(0, 1 - (step_count / dagger_total))
                use_expert = (np.random.rand() < beta)
                chosen_goal = true_goal if use_expert else agent_chosen_goal

                # 如果 chosen_goal 不再有效，则 fallback 到专家（如果专家也无效，则结束请求）
                if chosen_goal not in env.unadded_dest_indices:
                    logger.warning("Chosen goal %s 不再有效，尝试专家目标", chosen_goal)
                    chosen_goal = true_goal
                    if chosen_goal not in env.unadded_dest_indices:
                        logger.warning("专家目标也无效，放弃请求")
                        break

                # 低层执行直到子任务完成或超时
                sub_task_done = False
                low_level_state = high_level_state
                low_level_timeout = 0
                # 记录单子任务最大尝试次数（防无限循环）
                max_low_level_attempts = env.K_path + 3

                while not sub_task_done:
                    low_level_timeout += 1
                    if low_level_timeout > max_low_level_attempts:
                        logger.warning("低层超时: 无法为 goal %s 找到可行路径", chosen_goal)
                        # 给一次失败的内部惩罚并放弃该请求
                        try:
                            fail_reward = low_level_agent.criticize(False, cost=9999, request_failed=True)
                        except Exception:
                            fail_reward = -10.0
                        # 存储失败经验（使 agent 学会避免此类状态）
                        exp = ActorExperience(high_level_state, to_categorical(chosen_goal, num_classes=NB_GOALS),
                                              0, fail_reward, low_level_state, True)
                        safe_store_and_maybe_train(low_level_agent, exp, step_count)
                        request_done = True
                        sub_task_done = True
                        break

                    low_v = np.reshape(low_level_state, (1, -1))
                    valid_actions = env.get_valid_low_level_actions()
                    if not valid_actions:
                        action = 0
                    else:
                        try:
                            action = low_level_agent.selectMove(low_v, np.reshape(to_categorical(chosen_goal, NB_GOALS), (1, -1)), valid_actions)
                        except Exception:
                            # fallback：随机或第一个
                            action = valid_actions[0]

                    # 校验 action（debug）
                    if action not in valid_actions:
                        logger.debug("动作 %s 非法，使用第一个有效动作", action)
                        action = valid_actions[0]

                    next_state, cost, sub_task_done, request_done = env.step_low_level(chosen_goal, action)
                    try:
                        reward = low_level_agent.criticize(sub_task_done, cost, request_failed=(not sub_task_done))
                    except Exception:
                        reward = 1.0 if sub_task_done else -1.0

                    exp = ActorExperience(high_level_state, to_categorical(chosen_goal, NB_GOALS).flatten(), action, reward, next_state, sub_task_done)
                    safe_store_and_maybe_train(low_level_agent, exp, step_count)

                    # 元控制器 DAgger 收集专家标签（始终收集专家答案以便后续训练）
                    try:
                        metacontroller.collect(high_level_state, true_goal)
                        if step_count % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                            metacontroller.train()
                    except Exception:
                        logger.debug("metacontroller train/collect 出错，忽略")

                    # Epsilon 衰减
                    if hasattr(low_level_agent, 'annealControllerEpsilon'):
                        try:
                            low_level_agent.annealControllerEpsilon(step_count)
                        except Exception:
                            pass

                    low_level_state = next_state
                    step_count += 1
                    episode_steps += 1
                    episode_reward += reward

                    # 有时我们也需要周期性保存/打印低层 training metrics
                    if step_count % 1000 == 0:
                        _ = safe_update_agent(low_level_agent, step_count)

                    if request_done:
                        break

                # 更新 high_level_state
                high_level_state = low_level_state

            # 回合结束 -> 保存统计
            avg_cpu_util = (1.0 - np.mean(env.C) / env.C_cap) * 100.0 if getattr(env, 'C_cap', None) else 0.0
            avg_mem_util = (1.0 - np.mean(env.M) / env.M_cap) * 100.0 if getattr(env, 'M_cap', None) else 0.0
            avg_bw_util = (1.0 - np.mean(env.B) / env.B_cap) * 100.0 if getattr(env, 'B_cap', None) else 0.0

            tracking_data['episode'].append(episode_count)
            tracking_data['reward'].append(episode_reward)

            total_reqs = max(1, getattr(env, 'total_requests_seen', 1))
            req_accept_rate = (getattr(env, 'total_requests_accepted', 0) / total_reqs) * 100.0

            # 尝试使用 env 里可能存在的 failed 统计（优先使用）
            blocked = None
            if hasattr(env, 'total_requests_failed'):
                blocked = getattr(env, 'total_requests_failed')
            else:
                blocked = total_reqs - getattr(env, 'total_requests_accepted', 0)

            blocking_rate = (blocked / total_reqs) * 100.0

            dest_accept_rate = (getattr(env, 'total_dest_accepted', 0) / max(1, getattr(env, 'total_dest_seen', 1))) * 100.0

            tracking_data['acceptance_rate'].append(req_accept_rate)
            tracking_data['blocking_rate'].append(blocking_rate)
            tracking_data['avg_cpu_util'].append(avg_cpu_util)
            tracking_data['avg_mem_util'].append(avg_mem_util)
            tracking_data['avg_bw_util'].append(avg_bw_util)

            logger.info("--- Episode %d (T=%d) ---", episode_count, env.t)
            logger.info("Steps total: %d  | Epsilon: %.4f", step_count, getattr(low_level_agent, 'controllerEpsilon', 0.0))
            logger.info("Episode reward: %.3f | Episode steps: %d", episode_reward, episode_steps)
            logger.info("Acceptance: %.2f%% (%d/%d) | Blocking: %.2f%% (%d/%d)" %
                        (req_accept_rate, getattr(env, 'total_requests_accepted', 0), total_reqs, blocking_rate, blocked, total_reqs))
            logger.info("Dest acceptance: %.2f%%", dest_accept_rate)
            logger.info("Resource utilization CPU: %.2f%% MEM: %.2f%% BW: %.2f%%" %
                        (avg_cpu_util, avg_mem_util, avg_bw_util))

            # 周期性保存模型
            episode_count += 1
            if episode_count % 50 == 0:
                try:
                    model_path = H.OUTPUT_DIR / f"sfc_hirl_model_ep{episode_count}"
                    hdqn_net.saveWeight(str(model_path))
                    logger.info("Saved model at %s", model_path)
                except Exception:
                    logger.exception("保存模型失败")

        except KeyboardInterrupt:
            logger.warning("用户中断训练")
            break
        except Exception:
            logger.exception("回合 %d 出错，跳过并继续", episode_count)
            episode_count += 1
            continue

    # 训练结束 -> 保存并绘图
    try:
        logger.info("训练结束，保存追踪数据与最终模型")
        save_tracking_data(tracking_data, H.OUTPUT_DIR)
        if hdqn_net is not None:
            hdqn_net.saveWeight(str(H.OUTPUT_DIR / "sfc_hirl_model_final"))
            logger.info("Saved final model")
    except Exception:
        logger.exception("结束时保存失败")

    logger.info("训练脚本退出")


if __name__ == "__main__":
    main()
