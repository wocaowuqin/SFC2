#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_hirl_sfc_final.py
Final Integrated Training Script for HIRL-SFC
Features:
- Hierarchical RL (Meta-Controller + Low-Level Agent)
- DAgger (Imitation Learning from Expert)
- Robust Backup Policy Integration
- Comprehensive Logging (Resource Usage, Blocking Rate, Backup Stats)
"""
from __future__ import annotations
import os
import sys
import logging
import traceback
import csv
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas as pd

# [CRITICAL FIX] Force non-interactive backend for Matplotlib
os.environ['MPLBACKEND'] = 'Agg'
try:
    import sys

    sys.modules['IPython'] = None
except ImportError:
    pass

import matplotlib

matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from tensorflow.keras.utils import to_categorical

# Project modules
import hyperparameters as H
from hirl_sfc_env import SFC_HIRL_Env
from hirl_sfc_models import MetaControllerNN, Hdqn_SFC
from hirl_sfc_agent import Agent_SFC

# Experience Tuple
ActorExperience = namedtuple("ActorExperience", ["state", "goal", "action", "reward", "next_state", "done"])

# Logging Configuration
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler(H.OUTPUT_DIR / "training.log", mode='w',
                                            encoding='utf-8') if H.OUTPUT_DIR.exists() else logging.StreamHandler(
                            sys.stdout)
                    ])
logger = logging.getLogger(__name__)


def safe_update_agent(agent, t: int = None):
    """
    Safely call agent update/train methods with compatibility checks.
    """
    try_names = ['update', 'train_from_memory', 'train', 'train_step']
    for name in try_names:
        fn = getattr(agent, name, None)
        if callable(fn):
            try:
                import inspect
                sig = inspect.signature(fn)
                if len(sig.parameters) == 0:
                    return fn()
                else:
                    return fn(t)
            except Exception:
                continue
    return None


def safe_store_and_maybe_train(agent, exp: ActorExperience, step_count: int):
    """
    Store experience and trigger training based on frequency.
    """
    try:
        if hasattr(agent, 'store'):
            agent.store(exp)
        elif hasattr(agent, 'memory_push'):
            agent.memory_push(exp)
    except Exception:
        pass

    train_freq = getattr(agent, 'trainFreq', getattr(agent, 'train_freq', getattr(agent, 'trainFreq', None)))

    if train_freq is None:
        safe_update_agent(agent, step_count)
    else:
        if train_freq > 0 and (step_count % train_freq == 0):
            safe_update_agent(agent, step_count)


def save_training_plots_from_df(df: pd.DataFrame, out_dir: Path):
    """
    Generate and save training curves.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        def smoothed(values, window=50):
            if len(values) == 0: return values
            window = min(window, max(1, len(values)))
            return pd.Series(values).rolling(window, min_periods=1).mean().values

        def plot_column(col, title, ylabel, smooth=False):
            if col not in df.columns: return

            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(1, 1, 1)

            data = df[col].values
            ax.plot(data, alpha=0.3, color='blue', label='Raw')
            if smooth:
                ax.plot(smoothed(data), color='red', linewidth=2, label='Smoothed')

            ax.set_title(title)
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

            fname = out_dir / f"{col}.png"
            FigureCanvas(fig).print_figure(str(fname))
            logger.info("Saved plot %s", fname)

        # Plot Metrics
        plot_column('reward', 'Episode Reward', 'Reward', smooth=True)
        plot_column('acceptance_rate', 'Request Acceptance Rate', '%', smooth=True)
        plot_column('blocking_rate', 'Blocking Rate', '%', smooth=True)
        plot_column('avg_cpu_util', 'Avg CPU Utilization', '%', smooth=True)
        plot_column('avg_bw_util', 'Avg BW Utilization', '%', smooth=True)

        # Backup Policy Metrics
        plot_column('backup_activation', 'Backup Policy Activation Rate (Expert Fail Rate)', '%', smooth=True)
        plot_column('backup_success', 'Backup Policy Success Rate', '%', smooth=True)

    except Exception:
        logger.exception("Plotting failed")


def save_tracking_data(tracking_data: dict, out_dir: Path):
    """
    Save metrics to CSV and plot.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(tracking_data)
        csv_path = out_dir / "training_metrics.csv"
        df.to_csv(csv_path, index=False)
        save_training_plots_from_df(df, out_dir)
    except Exception:
        logger.exception("Failed to save tracking data")


def main():
    """
    Main Training Loop
    """
    # 1. Validation and Setup
    if not H.INPUT_DIR.exists():
        print(f"Input directory not found: {H.INPUT_DIR}")
        return

    H.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize Resource Log CSV
    resource_log_path = H.OUTPUT_DIR / "resource_metrics.csv"
    with open(resource_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header with Status
        writer.writerow(["Episode", "RequestID", "Status", "ConsumedBW", "ConsumedCPU", "ConsumedMem", "Links", "VNFs"])

    logger.info("Starting HIRL-SFC Training (Final Integrated Version)")

    # Tracking Dictionary
    tracking_data = {
        'episode': [], 'reward': [],
        'acceptance_rate': [], 'blocking_rate': [],
        'avg_cpu_util': [], 'avg_mem_util': [], 'avg_bw_util': [],
        'backup_activation': [], 'backup_success': []
    }

    # 2. Initialize Environment & Agents
    env = SFC_HIRL_Env(H.INPUT_DIR, H.TOPOLOGY_MATRIX, H.DC_NODES, H.CAPACITIES)
    STATE_SHAPE = env.observation_space.shape
    NB_GOALS = env.NB_HIGH_LEVEL_GOALS
    NB_ACTIONS = env.NB_LOW_LEVEL_ACTIONS

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

    logger.info("Initialization Complete: STATE=%s, GOALS=%d, ACTIONS=%d", STATE_SHAPE, NB_GOALS, NB_ACTIONS)

    # ==========================================
    # Phase 1: Imitation Learning (Pre-training)
    # ==========================================
    logger.info("Phase 1: Imitation Learning Pre-training (%d steps)", H.PRE_TRAIN_STEPS)
    t = 0
    current_request, high_level_state = env.reset_request()

    while t < H.PRE_TRAIN_STEPS and current_request is not None:
        try:
            # Get Expert Trajectory
            try:
                _, expert_traj = env.expert.solve_request_for_expert(current_request, env._get_network_state_dict())
            except Exception:
                expert_traj = None

            if not expert_traj:
                t += 1
                current_request, high_level_state = env.reset_request()
                continue

            for (exp_goal, exp_action_tuple, exp_cost) in expert_traj:
                if exp_goal not in env.unadded_dest_indices: continue

                exp_action = exp_action_tuple[0] * env.K_path + exp_action_tuple[1]

                # Execute Step
                next_state, cost, sub_task_done, request_done = env.step_low_level(exp_goal, exp_action)

                reward = 1.0 if sub_task_done else -1.0
                goal_one_hot = to_categorical(exp_goal, num_classes=NB_GOALS)

                # Store Experience
                experience = ActorExperience(high_level_state, goal_one_hot, exp_action, reward, next_state,
                                             sub_task_done)
                safe_store_and_maybe_train(low_level_agent, experience, t)

                # Train Meta-Controller
                try:
                    metacontroller.collect(high_level_state, exp_goal)
                    if t % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                        metacontroller.train()
                except Exception:
                    pass

                high_level_state = next_state
                t += 1
                if t >= H.PRE_TRAIN_STEPS or request_done: break

            current_request, high_level_state = env.reset_request()

        except Exception:
            t += 1
            current_request, high_level_state = env.reset_request()

    logger.info("Pre-training Finished.")

    # ==========================================
    # Phase 2: Mixed IL/RL Training
    # ==========================================
    logger.info("Phase 2: Mixed IL/RL Training Start")
    step_count = t
    episode_count = 0

    dagger_total = max(1, H.STEPS_LIMIT)
    dagger_initial = 1.0
    dagger_final = 0.05

    while episode_count < H.EPISODE_LIMIT and step_count < H.STEPS_LIMIT:
        try:
            # Reset Env
            current_request, high_level_state = env.reset_request()
            if current_request is None: break

            episode_reward = 0.0
            episode_steps = 0
            request_done = False
            req_id = current_request.get('id', 'unknown')

            # Timeout mechanism to detect blocking
            attempts_per_req = 0
            max_attempts = 15

            while not request_done and attempts_per_req < max_attempts:
                attempts_per_req += 1
                high_state_v = np.reshape(high_level_state, (1, -1))

                # --- 1. Meta-Controller (Agent) ---
                try:
                    goal_probs = metacontroller.predict(high_state_v)
                    agent_chosen_goal = metacontroller.sample(goal_probs)
                except Exception:
                    agent_chosen_goal = list(env.unadded_dest_indices)[0] if env.unadded_dest_indices else 0

                # --- 2. Expert Opinion ---
                true_goal = env.get_expert_high_level_goal(high_state_v)

                # --- 3. DAgger Decision ---
                beta = dagger_final + (dagger_initial - dagger_final) * max(0, 1 - (step_count / dagger_total))
                use_expert = (np.random.rand() < beta)
                chosen_goal = true_goal if use_expert else agent_chosen_goal

                if chosen_goal not in env.unadded_dest_indices:
                    chosen_goal = true_goal
                    if chosen_goal not in env.unadded_dest_indices: break

                # --- 4. Low Level Execution ---
                sub_task_done = False
                low_level_state = high_level_state

                # Sub-task retry loop
                sub_attempts = 0
                while not sub_task_done and sub_attempts < 10:
                    sub_attempts += 1
                    low_v = np.reshape(low_level_state, (1, -1))
                    valid_actions = env.get_valid_low_level_actions()

                    # Select Action (Epsilon-Greedy with Mask)
                    try:
                        action = low_level_agent.selectMove(low_v,
                                                            np.reshape(to_categorical(chosen_goal, NB_GOALS), (1, -1)),
                                                            valid_actions)
                    except Exception:
                        action = valid_actions[0]

                    # Step Environment
                    # Note: env.step logic now includes Backup Policy invocation automatically
                    next_state, reward, sub_task_done, request_done = env.step_low_level(chosen_goal, action)

                    # Store & Train
                    exp = ActorExperience(high_level_state, to_categorical(chosen_goal, NB_GOALS).flatten(), action,
                                          reward, next_state, sub_task_done)
                    safe_store_and_maybe_train(low_level_agent, exp, step_count)

                    # DAgger Aggregation
                    try:
                        metacontroller.collect(high_level_state, true_goal)
                        if step_count % H.META_TRAIN_FREQ == 0 and metacontroller.check_training_clock():
                            metacontroller.train()
                    except Exception:
                        pass

                    low_level_state = next_state
                    step_count += 1
                    episode_steps += 1
                    episode_reward += reward

                    if request_done: break

                high_level_state = low_level_state

            # --- Episode End: Logging & Metrics ---

            # 1. Resource Logging
            # Note: We need to capture resource info from environment if possible,
            # or calculate it here if request succeeded.
            # Assuming hirl_sfc_env.py does NOT return resource_info directly in step_low_level,
            # we check request_done status.

            if request_done:
                # Calculate resources for logging
                try:
                    tree_links = np.sum(env.current_tree['tree'] > 0)
                    req_bw = float(current_request.get('bw_origin', 0.0))
                    total_bw = tree_links * req_bw
                    total_cpu = sum(current_request.get('cpu_origin', []))
                    total_mem = sum(current_request.get('memory_origin', []))
                    vnf_cnt = len(current_request.get('vnf', []))

                    with open(resource_log_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            episode_count, req_id, "Accepted",
                            f"{total_bw:.2f}", f"{total_cpu:.2f}", f"{total_mem:.2f}",
                            int(tree_links), vnf_cnt
                        ])
                except:
                    pass
            else:
                # Blocked
                with open(resource_log_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        episode_count, req_id, "Blocked",
                        0, 0, 0, 0, len(current_request.get('vnf', []))
                    ])
                logger.warning(f"Req {req_id} Blocked (Max attempts reached)")

            # 2. General Metrics
            backup_stats = env.get_backup_metrics() if hasattr(env, 'get_backup_metrics') else {'activation_rate': 0,
                                                                                                'success_rate': 0}

            tracking_data['episode'].append(episode_count)
            tracking_data['reward'].append(episode_reward)

            total_req = max(1, getattr(env, 'total_requests_seen', 1))
            acc_rate = (getattr(env, 'total_requests_accepted', 0) / total_req) * 100.0
            blk_rate = 100.0 - acc_rate

            tracking_data['acceptance_rate'].append(acc_rate)
            tracking_data['blocking_rate'].append(blk_rate)

            avg_cpu = (1.0 - np.mean(env.C) / env.C_cap) * 100.0
            avg_bw = (1.0 - np.mean(env.B) / env.B_cap) * 100.0
            tracking_data['avg_cpu_util'].append(avg_cpu)
            tracking_data['avg_mem_util'].append((1.0 - np.mean(env.M) / env.M_cap) * 100.0)
            tracking_data['avg_bw_util'].append(avg_bw)

            tracking_data['backup_activation'].append(backup_stats.get('activation_rate', 0))
            tracking_data['backup_success'].append(backup_stats.get('success_rate', 0))

            logger.info(
                f"Ep {episode_count}: R={episode_reward:.2f}, Acc={acc_rate:.1f}%, Blk={blk_rate:.1f}%, "
                f"BW={avg_bw:.1f}%, BkAct={backup_stats.get('activation_rate', 0):.1f}%"
            )

            episode_count += 1
            if episode_count % 50 == 0:
                hdqn_net.saveWeight(str(H.OUTPUT_DIR / f"model_ep{episode_count}"))

        except KeyboardInterrupt:
            logger.warning("User interrupted training")
            break
        except Exception:
            logger.exception("Episode failed")
            episode_count += 1

    logger.info("Training finished.")
    save_tracking_data(tracking_data, H.OUTPUT_DIR)
    if hdqn_net:
        hdqn_net.saveWeight(str(H.OUTPUT_DIR / "model_final"))


if __name__ == "__main__":
    main()