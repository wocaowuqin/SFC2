#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_sfc_models_v3_enhanced.py
# @Author  : Gemini & User (V3 Enhanced)
# @Date    : 2025-11-26
"""
Network / Hierarchical RL model helpers for SFC experiments.

V3 Enhancements:
- Typing: Comprehensive type hints throughout.
- Robustness: Strict input validation and error handling.
- Features: Deterministic sampling, evaluation metrics, target net sync.
- Diagnostics: Model summary, GPU check, training stats.
- Performance: Optimized model calls and tensor ops.
"""

from __future__ import annotations

import random
import threading
import logging
from typing import Tuple, Optional, List, Union, Dict, Any
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical, register_keras_serializable

# Setup Logging
logger = logging.getLogger("HRL_Models")
# Ensure handler exists if not configured elsewhere
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@register_keras_serializable(package="Custom", name="huber_loss")
def huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, clip_value: float = 1.0) -> tf.Tensor:
    """
    Huber loss (smooth L1). Ensures clip_value > 0.
    Registered as serializable for model checkpointing.
    """
    clip_value = float(clip_value)
    if not np.isfinite(clip_value) or clip_value <= 0.0:
        clip_value = 1.0

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    x = y_true - y_pred
    abs_x = tf.abs(x)

    squared_loss = 0.5 * tf.square(x)
    linear_loss = clip_value * (abs_x - 0.5 * clip_value)

    return tf.where(abs_x <= clip_value, squared_loss, linear_loss)


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and info."""
    gpu_info = {
        'available': False,
        'devices': []
    }
    try:
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info['available'] = len(gpus) > 0
        for gpu in gpus:
            gpu_info['devices'].append(str(gpu))
    except Exception as e:
        logger.warning(f"Could not query GPU info: {e}")

    return gpu_info


class MetaControllerNN:
    """
    High-level meta controller (policy network) for hierarchical RL.

    This network learns to select high-level goals based on the current state.
    Uses DAgger-style imitation learning with thread-safe experience collection.
    """

    def __init__(self, state_shape: Tuple[int, ...], n_goals: int,
                 lr: float = 2.5e-4, replay_capacity: int = 1000,
                 seed: Optional[int] = None):
        """Initialize with comprehensive validation."""
        # Input Validation
        if not isinstance(state_shape, tuple) or len(state_shape) == 0:
            raise ValueError("state_shape must be a non-empty tuple")
        if n_goals <= 0:
            raise ValueError("n_goals must be positive")
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        if replay_capacity <= 0:
            raise ValueError("replay_capacity must be positive")

        self.state_shape = tuple(state_shape)
        self.n_goals = int(n_goals)
        self.replay_capacity = int(replay_capacity)
        self.lr = float(lr)

        # Replay Buffer
        self.replay_hist: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * self.replay_capacity
        self.ind = 0
        self.count = 0

        # Concurrency
        self._collect_lock = threading.Lock()

        # Seed
        if seed is not None:
            self.set_seed(seed)

        # Model
        self.meta_controller = self._build_model()
        self._compile_model()

        logger.info(f"MetaControllerNN initialized: state_shape={state_shape}, "
                    f"n_goals={n_goals}, capacity={replay_capacity}")

    def set_seed(self, seed: int):
        self._seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _build_model(self) -> Sequential:
        model = Sequential(name="MetaController")
        model.add(Input(shape=self.state_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.n_goals, activation='softmax'))
        return model

    def _compile_model(self) -> None:
        """Compile the model with configured optimizer."""
        try:
            # Using legacy RMSprop for RL stability often preferred, but standard works too
            rmsprop = optimizers.RMSprop(
                learning_rate=self.lr,
                rho=0.95,
                epsilon=1e-08,
                clipnorm=1.0  # Gradient clipping
            )
            self.meta_controller.compile(
                loss='categorical_crossentropy',
                optimizer=rmsprop,
                metrics=['accuracy']
            )
        except Exception as e:
            logger.error(f"Failed to compile model: {e}")
            raise

    def check_training_clock(self) -> bool:
        return (self.count >= 100)

    def collect(self, state: np.ndarray, expert_goal: int) -> None:
        """
        Thread-safe collection of expert demonstrations.
        """
        if not (0 <= expert_goal < self.n_goals):
            raise ValueError(f"expert_goal {expert_goal} out of range [0, {self.n_goals - 1}]")

        with self._collect_lock:
            try:
                s = np.asarray(state, dtype=np.float32)
                # Basic shape check (handling potential batch dim confusion)
                if s.shape != self.state_shape:
                    # Allow if it's just missing batch dim or has 1 batch dim
                    if s.ndim == len(self.state_shape) + 1 and s.shape[1:] == self.state_shape:
                        s = s[0]  # Flatten batch dim if 1
                    elif s.shape != self.state_shape:
                        raise ValueError(f"State shape {s.shape} != expected {self.state_shape}")

                one_hot = to_categorical(expert_goal, num_classes=self.n_goals).astype(np.float32)
                self.replay_hist[self.ind] = (s, one_hot)
                self.ind = (self.ind + 1) % self.replay_capacity
                self.count += 1

            except Exception as e:
                logger.error(f"Error in collect: {e}")
                raise

    def train(self, batch_size: int = 32, epochs: int = 1, validation_split: float = 0.0) -> Dict[str, float]:
        """
        Train the meta-controller with optional validation.
        """
        # Check for valid data presence
        # Optimization: Check buffer status directly
        valid_indices = [i for i, x in enumerate(self.replay_hist) if x is not None]
        num_valid = len(valid_indices)

        if num_valid < batch_size:
            # Silent return is common in RL loops to avoid log spam, but debug log is good
            # logger.debug(f"Insufficient data for training: {num_valid} < {batch_size}")
            return {'loss': float('nan'), 'accuracy': float('nan')}

        try:
            # Random sampling
            indices = np.random.choice(valid_indices, size=min(batch_size, num_valid), replace=False)
            data = [self.replay_hist[i] for i in indices]

            train_x = np.stack([d[0] for d in data], axis=0).astype(np.float32)
            train_y = np.stack([d[1] for d in data], axis=0).astype(np.float32)

            history = self.meta_controller.fit(
                train_x, train_y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=0,
                validation_split=validation_split
            )

            self.count = 0

            metrics = {
                'loss': float(history.history['loss'][-1]),
                'accuracy': float(history.history['accuracy'][-1])
            }
            if 'val_loss' in history.history:
                metrics.update({
                    'val_loss': float(history.history['val_loss'][-1]),
                    'val_accuracy': float(history.history['val_accuracy'][-1])
                })

            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'loss': float('nan'), 'accuracy': float('nan')}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Fast prediction."""
        x_arr = np.asarray(x, dtype=np.float32)
        is_single = (x_arr.ndim == len(self.state_shape))

        if is_single:
            x_arr = np.expand_dims(x_arr, axis=0)

        # Using __call__ is faster for small batches than .predict()
        preds = self.meta_controller(x_arr, training=False).numpy()

        if is_single:
            return preds[0]
        return preds

    def sample(self, prob_vec: np.ndarray, temperature: float = 0.1,
               deterministic: bool = False) -> int:
        """
        Sample goal from probability distribution with temperature scaling.
        """
        pv = np.asarray(prob_vec, dtype=np.float64).ravel()

        # Normalize if needed
        if not np.allclose(np.sum(pv), 1.0, atol=1e-5):
            # logger.warning("Probability vector does not sum to 1, normalizing")
            pv = pv / (np.sum(pv) + 1e-12)

        if deterministic:
            return int(np.argmax(pv))

        temp = max(float(temperature), 1e-6)
        # Clip to avoid log(0)
        logits = np.log(np.clip(pv, 1e-12, 1.0)) / temp
        logits = logits - np.max(logits)  # Stability
        exp_logits = np.exp(logits)
        softmax_probs = exp_logits / (np.sum(exp_logits) + 1e-16)

        try:
            return int(np.random.choice(len(softmax_probs), p=softmax_probs))
        except ValueError:
            # Fallback if probs sum is weirdly off due to float precision
            return int(np.argmax(pv))

    def evaluate(self, states: np.ndarray, expert_goals: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        try:
            states = np.asarray(states, dtype=np.float32)
            expert_goals = np.asarray(expert_goals, dtype=int)

            predictions = self.predict(states)
            predicted_goals = np.argmax(predictions, axis=1)

            accuracy = np.mean(predicted_goals == expert_goals)
            # Basic CE
            row_indices = np.arange(len(expert_goals))
            confidences = predictions[row_indices, expert_goals]
            cross_entropy = -np.mean(np.log(np.clip(confidences, 1e-12, 1.0)))

            metrics = {
                'accuracy': float(accuracy),
                'cross_entropy': float(cross_entropy),
                'n_samples': len(states)
            }

            logger.info(f"Evaluation: acc={accuracy:.3f}, ce={cross_entropy:.3f}")
            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {'accuracy': float('nan'), 'cross_entropy': float('nan'), 'n_samples': 0}

    def get_training_stats(self) -> Dict[str, Any]:
        """Get buffer stats."""
        valid_data = [d for d in self.replay_hist if d is not None]
        num_valid = len(valid_data)

        if num_valid == 0:
            return {'total_samples': 0, 'buffer_utilization': 0.0}

        goals = [np.argmax(d[1]) for d in valid_data]
        unique, counts = np.unique(goals, return_counts=True)
        dist = {int(g): int(c) for g, c in zip(unique, counts)}

        return {
            'total_samples': num_valid,
            'buffer_utilization': num_valid / self.replay_capacity,
            'goal_distribution': dist,
            'training_ready': self.check_training_clock()
        }

    def save_weights(self, prefix: str):
        p = Path(f"{prefix}_meta.weights.h5")
        self.meta_controller.save_weights(str(p))

    def load_weights(self, prefix: str):
        p = Path(f"{prefix}_meta.weights.h5")
        if p.exists():
            self.meta_controller.load_weights(str(p))
            logger.info(f"MetaController weights loaded from {p}")
        else:
            logger.warning(f"Weights file not found: {p}")


class Hdqn_SFC:
    """
    Low-level Q-network for hierarchical deep Q-learning.
    Conditioned on (State, Goal) -> Action Q-values.
    """

    def __init__(self, state_shape: Tuple[int, ...], n_goals: int, n_actions: int,
                 lr: float = 2.5e-4, seed: Optional[int] = None):
        if n_actions <= 0:
            raise ValueError("n_actions must be positive")

        self.state_shape = tuple(state_shape)
        self.n_goals = int(n_goals)
        self.n_actions = int(n_actions)
        self.lr = float(lr)

        if seed is not None:
            self.set_seed(seed)

        self.controllerNet = self._build_model("ControllerNet")
        self.targetControllerNet = self._build_model("TargetNet")
        self.sync_target_network()

        logger.info(f"Hdqn_SFC initialized: state={state_shape}, goals={n_goals}, actions={n_actions}")

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _build_model(self, name: str) -> Model:
        state_input = Input(shape=self.state_shape, name='state_input')
        goal_input = Input(shape=(self.n_goals,), name='goal_input')

        merged = concatenate([state_input, goal_input])
        x = Dense(256, activation='relu')(merged)
        x = Dense(256, activation='relu')(x)
        out = Dense(self.n_actions, activation='linear', name='q_values')(x)

        model = Model(inputs=[state_input, goal_input], outputs=out, name=name)

        # Using MSE loss for standard DQN
        rmsProp = optimizers.RMSprop(learning_rate=self.lr, rho=0.95, epsilon=1e-08)
        model.compile(loss='mse', optimizer=rmsProp)
        return model

    def sync_target_network(self) -> None:
        """Synchronize target network weights."""
        self.targetControllerNet.set_weights(self.controllerNet.get_weights())
        # logger.debug("Target network synchronized")

    def predict_q(self, state: np.ndarray, goal_onehot: np.ndarray,
                  use_target: bool = False) -> np.ndarray:
        """
        Predict Q-values for state-goal pairs.
        """
        model = self.targetControllerNet if use_target else self.controllerNet

        s = np.asarray(state, dtype=np.float32)
        g = np.asarray(goal_onehot, dtype=np.float32)

        is_single = (s.ndim == len(self.state_shape))
        if is_single:
            s = np.expand_dims(s, axis=0)
            g = np.expand_dims(g, axis=0)

        # Shape validation skipped for performance in tight loops, relies on Keras errors

        # Fast inference
        qvals = model([s, g], training=False).numpy()

        if is_single:
            return qvals[0]
        return qvals

    def train_on_batch(self, states: np.ndarray, goals: np.ndarray,
                       targets: np.ndarray) -> float:
        """
        Perform one training step.
        """
        try:
            # train_on_batch is standard Keras API
            loss = self.controllerNet.train_on_batch([states, goals], targets)
            return float(loss)
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return float('nan')

    def saveWeight(self, file_prefix: str) -> bool:
        try:
            p = Path(f"{file_prefix}_controller.weights.h5")
            self.controllerNet.save_weights(str(p))
            logger.info(f"Weights saved to {p}")
            return True
        except Exception as e:
            logger.error(f"Failed to save weights: {e}")
            return False

    def loadWeight(self, file_prefix: str) -> bool:
        try:
            p = Path(f"{file_prefix}_controller.weights.h5")
            if not p.exists():
                logger.error(f"Weight file not found: {p}")
                return False

            self.controllerNet.load_weights(str(p))
            self.sync_target_network()

            # Verify sync
            cw = self.controllerNet.get_weights()
            tw = self.targetControllerNet.get_weights()
            if len(cw) > 0:
                if not np.allclose(cw[0], tw[0], atol=1e-5):
                    logger.warning("Target sync verification failed")

            logger.info(f"Weights loaded from {p}")
            return True
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
            return False

    def get_model_summary(self) -> str:
        string_list = []
        self.controllerNet.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)