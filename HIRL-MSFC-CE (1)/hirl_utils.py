#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @File    : hirl_utils_v11_robust.py
# @Author  : Gemini & User
# @Date    : 2025-06-26
"""
Prioritized Experience Replay - V11 Robust (Anti-Crash)
Fixes:
1. [Crash Fix] update_priorities(): Now handles non-scalar priorities (e.g., arrays, lists)
   by taking np.mean() automatically. This fixes "only length-1 arrays..." TypeError.
2. [Plot Fix] Sets matplotlib backend to 'Agg' to avoid IPython conflicts.
3. [Compat] Retains all previous V10 fixes (size alias, legacy init, safe save).
"""

from __future__ import annotations

import os
import tempfile
import gzip
import pickle
import threading
import time
import logging
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Final

# [Plot Fix] Try to fix IPython/Matplotlib conflict early
try:
    import matplotlib

    # Force Agg backend if not already interacting with a window system
    # This prevents the "IPython has no attribute version_info" crash
    matplotlib.use('Agg')
except Exception:
    pass

import numpy as np

# logger
logger = logging.getLogger("PER_Buffer")
logger.addHandler(logging.NullHandler())


class ReadWriteLock:
    """Writer-preference ReadWrite lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self._read_ready = threading.Condition(self._lock)
        self._readers = 0
        self._writers_waiting = 0
        self._writer_active = False

    def acquire_read(self):
        with self._lock:
            while self._writer_active or self._writers_waiting > 0:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        with self._lock:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        with self._lock:
            self._writers_waiting += 1
            while self._readers > 0 or self._writer_active:
                self._read_ready.wait()
            self._writers_waiting -= 1
            self._writer_active = True

    def release_write(self):
        with self._lock:
            self._writer_active = False
            self._read_ready.notify_all()

    def read_lock(self):
        return _ReadLockContext(self)

    def write_lock(self):
        return _WriteLockContext(self)


class _ReadLockContext:
    def __init__(self, rw_lock):
        self.rw_lock = rw_lock

    def __enter__(self):
        self.rw_lock.acquire_read()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_read()
        return False


class _WriteLockContext:
    def __init__(self, rw_lock):
        self.rw_lock = rw_lock

    def __enter__(self):
        self.rw_lock.acquire_write()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rw_lock.release_write()
        return False


@dataclass
class PERConfig:
    buffer_size: int = 100000
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 100000
    epsilon: float = 1e-6
    min_priority: float = 1e-6
    seed: Optional[int] = None

    # storage
    use_memmap: bool = False
    memmap_dir: Optional[Union[str, Path]] = None

    # Performance flag
    zero_on_clear: bool = False

    # shapes (optional)
    obs_shape: Optional[Tuple[int, ...]] = None
    goal_shape: Optional[Tuple[int, ...]] = None
    action_shape: Optional[Tuple[int, ...]] = None
    action_dtype: Any = np.int32
    reward_dtype: Any = np.float32

    use_compression: bool = False

    def __post_init__(self):
        if self.buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if self.use_memmap:
            if self.memmap_dir is None:
                raise ValueError("memmap_dir is required when use_memmap=True")
            self.memmap_dir = Path(self.memmap_dir)


class SegmentTree:
    """Segment Tree with float64 storage."""
    _MIN_TOTAL: Final[float] = 1e-64

    def __init__(self, capacity: int, operation: Callable[[float, float], float], neutral_element: float):
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError("capacity must be positive and power of 2")
        self._capacity = capacity
        self._value: np.ndarray = np.full(2 * capacity, float(neutral_element), dtype=np.float64)
        self._operation = operation
        self.neutral_element = float(neutral_element)

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._capacity

    def __repr__(self) -> str:
        try:
            total = self.reduce()
        except Exception:
            total = float('nan')
        return (f"SegmentTree(capacity={self._capacity}, total={total:.6f})")

    def reduce(self, start: int = 0, end: Optional[int] = None) -> float:
        if end is None: end = self._capacity
        if start < 0: start += self._capacity
        if end < 0: end += self._capacity
        if start < 0 or end > self._capacity or start >= end:
            return self.neutral_element
        start += self._capacity
        end += self._capacity
        res = self.neutral_element
        while start < end:
            if start & 1:
                res = self._operation(res, float(self._value[start]))
                start += 1
            if end & 1:
                end -= 1
                res = self._operation(res, float(self._value[end]))
            start //= 2
            end //= 2
        return float(res)

    def __setitem__(self, idx: int, val: float):
        if not (0 <= idx < self._capacity):
            raise IndexError(f"SegmentTree index out of range: {idx}")
        idx_leaf = idx + self._capacity
        self._value[idx_leaf] = float(val)
        idx_parent = idx_leaf // 2
        while idx_parent >= 1:
            self._value[idx_parent] = self._operation(self._value[2 * idx_parent], self._value[2 * idx_parent + 1])
            idx_parent //= 2

    def __getitem__(self, idx: int) -> float:
        if not (0 <= idx < self._capacity):
            raise IndexError(f"SegmentTree index out of range: {idx}")
        return float(self._value[idx + self._capacity])

    def update_batch(self, idxes: List[int], values: List[float]):
        if len(idxes) != len(values):
            raise ValueError("idxes and values must be same length")
        for idx, val in zip(idxes, values):
            if not (0 <= idx < self._capacity):
                continue
            self._value[idx + self._capacity] = float(val)

        parents = {(idx + self._capacity) // 2 for idx in idxes if 0 <= idx < self._capacity}
        if not parents: return

        while parents:
            next_parents = set()
            for p in sorted(parents):
                if p < 1: continue
                self._value[p] = self._operation(self._value[2 * p], self._value[2 * p + 1])
                if p > 1: next_parents.add(p // 2)
            parents = next_parents

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        idx = 1
        while idx < self._capacity:
            left = float(self._value[2 * idx])
            if prefixsum < left:
                idx = 2 * idx
            else:
                prefixsum -= left
                idx = 2 * idx + 1
        return min(idx - self._capacity, self._capacity - 1)


class LinearSchedule:
    def __init__(self, schedule_timesteps: int, final_p: float, initial_p: float = 1.0):
        self.schedule_timesteps = int(max(1, schedule_timesteps))
        self.final_p = float(final_p)
        self.initial_p = float(initial_p)

    def value(self, t: int) -> float:
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return float(self.initial_p + fraction * (self.final_p - self.initial_p))


class PrioritizedReplayBuffer:
    VERSION = "11.0-robust"

    def __init__(self,
                 capacity_or_config: Union[int, PERConfig],
                 alpha: float = 0.6,
                 beta_start: float = 0.4,
                 beta_frames: int = 100000,
                 epsilon: float = 1e-6,
                 **kwargs):
        """
        Supports both new style (PERConfig) and legacy style (capacity, alpha, ...)
        """
        # [Legacy Support] Detect if first arg is int (capacity) or PERConfig
        if isinstance(capacity_or_config, PERConfig):
            self.config = capacity_or_config
        else:
            # Construct config from legacy arguments
            self.config = PERConfig(
                buffer_size=int(capacity_or_config),
                alpha=alpha,
                beta_start=beta_start,
                beta_frames=beta_frames,
                epsilon=epsilon,
                **kwargs
            )

        self._maxsize = int(self.config.buffer_size)
        self._next_idx = 0
        self._size = 0
        self._alpha = float(self.config.alpha)

        self._rw_lock = ReadWriteLock()
        self._meta_lock = threading.Lock()
        self._rng = np.random.default_rng(self.config.seed)

        it_capacity = 1
        while it_capacity < self._maxsize:
            it_capacity *= 2
        self._it_sum = SegmentTree(it_capacity, lambda x, y: x + y, 0.0)
        self._it_min = SegmentTree(it_capacity, min, float('inf'))
        self._max_priority = 1.0

        self.beta_schedule = LinearSchedule(self.config.beta_frames, final_p=1.0, initial_p=self.config.beta_start)

        self.metrics = {'add_count': 0, 'sample_count': 0, 'update_count': 0, 'total_samples': 0, 'priority_updates': 0}
        self._sample_elapsed_ms = 0.0

        self._storage_initialized = False
        self.states = None
        self.goals = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.dones = None

        if self.config.obs_shape is not None:
            self._init_storage(
                obs_shape=self.config.obs_shape,
                goal_shape=self.config.goal_shape if self.config.goal_shape else self.config.obs_shape,
                action_shape=self.config.action_shape if self.config.action_shape else (1,),
                action_dtype=self.config.action_dtype,
                reward_dtype=self.config.reward_dtype
            )

    def __repr__(self) -> str:
        return (f"PrioritizedReplayBuffer(size={self._size}/{self._maxsize}, "
                f"alpha={self._alpha:.2f}, memmap={self.config.use_memmap})")

    def __len__(self) -> int:
        return self._size

    def size(self) -> int:
        """Compatibility alias for agents calling .size()"""
        return self._size

    def _init_storage(self, obs_shape, goal_shape, action_shape, action_dtype, reward_dtype):
        c = self.config
        obs_shape = tuple(obs_shape)
        goal_shape = tuple(goal_shape)
        action_shape = tuple(action_shape)

        if c.use_memmap:
            memdir = str(c.memmap_dir)
            os.makedirs(memdir, exist_ok=True)
            mode = 'r+' if os.path.exists(os.path.join(memdir, 'states.dat')) else 'w+'

            def create(name, shape, dtype):
                path = os.path.join(memdir, f'{name}.dat')
                return np.memmap(path, dtype=dtype, mode=mode, shape=(self._maxsize, *shape))
        else:
            def create(name, shape, dtype):
                return np.zeros((self._maxsize, *shape), dtype=dtype)

        try:
            self.states = create('states', obs_shape, np.float32)
            self.goals = create('goals', goal_shape, np.float32)
            self.actions = create('actions', action_shape, action_dtype)
            self.rewards = create('rewards', (1,), reward_dtype)
            self.next_states = create('next_states', obs_shape, np.float32)
            self.dones = create('dones', (1,), np.float32)
            self._storage_initialized = True

            self.config.obs_shape = obs_shape
            self.config.goal_shape = goal_shape
            self.config.action_shape = action_shape
        except Exception:
            logger.error("Failed to initialize storage", exc_info=True)
            raise

    def _lazy_init(self, state, goal, action, reward):
        logger.info("Lazy init storage from first sample")
        s_arr = np.asarray(state)
        g_arr = np.asarray(goal)
        obs_shape = s_arr.shape
        goal_shape = g_arr.shape

        a_arr = np.asarray(action)
        action_shape = a_arr.shape if a_arr.ndim > 0 else (1,)

        # Smart Dtype Inference
        if a_arr.size > 0:
            if np.issubdtype(a_arr.dtype, np.integer):
                action_dtype = np.int32
            elif np.issubdtype(a_arr.dtype, np.floating):
                action_dtype = np.float32
            else:
                action_dtype = a_arr.dtype
        else:
            action_dtype = np.int32

        reward_arr = np.asarray(reward)
        reward_dtype = reward_arr.dtype if reward_arr.size > 0 else np.float32

        self._init_storage(obs_shape, goal_shape, action_shape, action_dtype, reward_dtype)

    def add(self, state, goal, action, reward, next_state, done):
        if not self._storage_initialized:
            with self._rw_lock.write_lock():
                if not self._storage_initialized:
                    try:
                        self._lazy_init(state, goal, action, reward)
                    except Exception as e:
                        raise RuntimeError(f"Init failed: {e}") from e

        try:
            with self._rw_lock.write_lock():
                idx = self._next_idx

                try:
                    self.states[idx] = np.asarray(state, dtype=np.float32)
                except Exception as e:
                    raise ValueError(f"State assign error: {e}")

                try:
                    self.goals[idx] = np.asarray(goal, dtype=np.float32)
                except Exception as e:
                    raise ValueError(f"Goal assign error: {e}")

                try:
                    if np.isscalar(action):
                        self.actions[idx] = action
                    else:
                        self.actions[idx] = np.asarray(action)
                except Exception as e:
                    raise ValueError(f"Action assign error: {e}")

                try:
                    self.rewards[idx, 0] = float(reward)
                except Exception as e:
                    raise ValueError(f"Reward assign error: {e}")

                try:
                    self.next_states[idx] = np.asarray(next_state, dtype=np.float32)
                except Exception as e:
                    raise ValueError(f"NextState assign error: {e}")

                try:
                    self.dones[idx, 0] = 1.0 if bool(done) else 0.0
                except Exception as e:
                    raise ValueError(f"Done assign error: {e}")

                if self._alpha == 0:
                    priority = max(self._max_priority, self.config.min_priority)
                else:
                    priority = max(self._max_priority, self.config.min_priority) ** self._alpha

                self._it_sum[idx] = priority
                self._it_min[idx] = priority

                self._next_idx = (idx + 1) % self._maxsize
                self._size = min(self._size + 1, self._maxsize)

                with self._meta_lock:
                    self.metrics['add_count'] += 1
        except Exception:
            logger.exception("Exception in add()")
            raise

    def _sample_proportional(self, batch_size: int, size: int) -> List[int]:
        total = max(self._it_sum.reduce(0, size), self._it_sum._MIN_TOTAL)
        idxes = []
        for _ in range(batch_size):
            mass = self._rng.random() * total
            idx = self._it_sum.find_prefixsum_idx(mass)
            if idx >= size: idx = size - 1
            idxes.append(int(idx))
        return idxes

    def _sample_proportional_no_replace(self, batch_size: int, size: int) -> List[int]:
        total = max(self._it_sum.reduce(0, size), self._it_sum._MIN_TOTAL)
        segment = total / batch_size
        idxes = []
        for i in range(batch_size):
            a = segment * i
            b = a + segment
            mass = a + self._rng.random() * (b - a)
            idx = self._it_sum.find_prefixsum_idx(mass)
            if idx >= size: idx = size - 1
            idxes.append(int(idx))

        unique_idxes = []
        seen = set()
        for idv in idxes:
            if idv not in seen:
                unique_idxes.append(idv)
                seen.add(idv)
        if len(unique_idxes) < batch_size:
            fill = list(set(range(size)) - seen)
            if len(fill) < (batch_size - len(unique_idxes)):
                extra = self._rng.choice(size, size=(batch_size - len(unique_idxes)), replace=True).tolist()
            else:
                extra = self._rng.choice(fill, size=(batch_size - len(unique_idxes)), replace=False).tolist()
            unique_idxes.extend(extra)
        return unique_idxes[:batch_size]

    def sample(self, batch_size: int, beta: Optional[float] = None, replace: bool = True) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        if batch_size <= 0: raise ValueError("batch_size must be positive")

        # Phase 1: Calc & Snapshot (Read Lock)
        with self._rw_lock.read_lock():
            if self._size == 0: raise RuntimeError("Buffer empty")
            current_size = self._size
            actual_batch_size = min(batch_size, current_size)

            idxes = (self._sample_proportional(actual_batch_size, current_size)
                     if replace
                     else self._sample_proportional_no_replace(actual_batch_size, current_size))

            total = self._it_sum.reduce(0, current_size)
            total_safe = max(total, self._it_sum._MIN_TOTAL)
            p_min_raw = self._it_min.reduce(0, current_size)
            p_min = (float(p_min_raw) / total_safe) if np.isfinite(p_min_raw) else (
                        (self.config.min_priority ** self._alpha) / total_safe)

            with self._meta_lock:
                if beta is None: beta = float(self.beta_schedule.value(self.metrics['sample_count']))

            max_weight = (p_min * current_size) ** (-beta) if p_min > 0 else 1.0

            weights = np.zeros(len(idxes), dtype=np.float32)
            for i, idx in enumerate(idxes):
                p_sample = float(self._it_sum[idx]) / total_safe
                if p_sample > 0:
                    weight = (p_sample * current_size) ** (-beta)
                    weights[i] = weight / max_weight if max_weight > 0 else 0.0

            # Refs
            s_ref, g_ref, a_ref = self.states, self.goals, self.actions
            r_ref, ns_ref, d_ref = self.rewards, self.next_states, self.dones

        # Phase 2: Copy (Lock Free)
        batch = {
            "states": np.array(s_ref[idxes], copy=True),
            "goals": np.array(g_ref[idxes], copy=True),
            "actions": np.array(a_ref[idxes], copy=True),
            "rewards": np.array(r_ref[idxes], copy=True).reshape(len(idxes)),
            "next_states": np.array(ns_ref[idxes], copy=True),
            "dones": np.array(d_ref[idxes], copy=True).reshape(len(idxes)),
            "weights": weights,
            "idxes": idxes
        }

        with self._meta_lock:
            self.metrics['sample_count'] += 1
            self.metrics['total_samples'] += len(idxes)

        self._sample_elapsed_ms = (time.perf_counter() - t0) * 1000
        return batch

    def update_priorities(self, idxes: List[int], priorities: List[float]):
        if len(idxes) != len(priorities): raise ValueError("len mismatch")

        with self._rw_lock.write_lock():
            uniq_updates = OrderedDict()
            max_p = 0.0
            for idx, p in zip(idxes, priorities):
                if 0 <= idx < self._size:
                    # [Crash Fix] Handle array/list inputs safely
                    if np.ndim(p) > 0:
                        p_val = float(np.mean(p))
                    else:
                        p_val = float(p)

                    p_val = max(p_val, float(self.config.min_priority))
                    uniq_updates[idx] = p_val
                    if p_val > max_p: max_p = p_val

            if uniq_updates:
                self._max_priority = max(self._max_priority, max_p)
                valid_idxes = list(uniq_updates.keys())
                valid_vals = [1.0 for _ in uniq_updates] if self._alpha == 0 else [p ** self._alpha for p in
                                                                                   uniq_updates.values()]

                self._it_sum.update_batch(valid_idxes, valid_vals)
                for i, v in zip(valid_idxes, valid_vals):
                    self._it_min[i] = v

                with self._meta_lock:
                    self.metrics['update_count'] += 1
                    self.metrics['priority_updates'] += len(valid_idxes)

    def clear(self):
        with self._rw_lock.write_lock():
            self._next_idx = 0
            self._size = 0
            self._max_priority = 1.0
            self._it_sum = SegmentTree(self._it_sum._capacity, lambda x, y: x + y, 0.0)
            self._it_min = SegmentTree(self._it_min._capacity, min, float('inf'))
            with self._meta_lock:
                self.metrics = {k: 0 for k in self.metrics}

            if self._storage_initialized and self.config.zero_on_clear:
                try:
                    if self.states is not None:
                        self.states[:] = 0
                        self.goals[:] = 0
                        self.actions[:] = 0
                        self.rewards[:] = 0
                        self.next_states[:] = 0
                        self.dones[:] = 0
                        if self.config.use_memmap:
                            for arr in (self.states, self.goals, self.actions, self.rewards, self.next_states,
                                        self.dones):
                                if hasattr(arr, 'flush'): arr.flush()
                except Exception:
                    logger.warning("Zero-out failed", exc_info=True)
            logger.info("Buffer cleared")

    def save(self, filepath: Union[str, Path]) -> bool:
        with self._rw_lock.read_lock():
            try:
                fp = Path(filepath)
                fp.parent.mkdir(parents=True, exist_ok=True)

                def _clean_val(v):
                    if isinstance(v, Path): return str(v)
                    if isinstance(v, (np.ndarray, np.generic)): return v.tolist()
                    return v

                safe_config = {k: _clean_val(v) for k, v in self.config.__dict__.items()}

                state = {
                    'version': self.VERSION,
                    'next_idx': self._next_idx,
                    'size': self._size,
                    'max_priority': self._max_priority,
                    'metrics': self.metrics,
                    'config': safe_config,
                    'rng_state': self._rng.bit_generator.state,
                    'tree_sum': self._it_sum._value,
                    'tree_min': self._it_min._value,
                }

                if self._storage_initialized and not self.config.use_memmap:
                    state['data'] = {
                        'states': self.states, 'goals': self.goals, 'actions': self.actions,
                        'rewards': self.rewards, 'next_states': self.next_states, 'dones': self.dones,
                    }

                if self.config.use_memmap and self._storage_initialized:
                    for arr in (self.states, self.goals, self.actions, self.rewards, self.next_states, self.dones):
                        try:
                            if hasattr(arr, "flush"): arr.flush()
                        except Exception:
                            pass

                suffix = '.pkl.gz' if self.config.use_compression else '.pkl'
                if not str(fp).endswith(suffix): fp = fp.with_suffix(suffix)

                tmp = None
                try:
                    tmpf = tempfile.NamedTemporaryFile(delete=False, dir=str(fp.parent), suffix=fp.suffix)
                    tmpf.close()
                    tmp = Path(tmpf.name)

                    if self.config.use_compression:
                        with gzip.open(tmp, 'wb') as gz:
                            pickle.dump(state, gz)
                    else:
                        with open(tmp, 'wb') as f:
                            pickle.dump(state, f)

                    if os.name == 'nt' and fp.exists():
                        try:
                            os.remove(fp)
                        except OSError:
                            pass

                    os.replace(tmp, fp)
                    tmp = None
                    logger.info(f"Saved to {fp}")
                    return True
                finally:
                    if tmp is not None and tmp.exists():
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass
            except Exception:
                logger.exception("Save failed")
                return False

    def load(self, filepath: Union[str, Path]) -> bool:
        fp = Path(filepath)
        candidates = [fp, fp.with_suffix('.pkl.gz'), fp.with_suffix('.pkl')]
        found = next((c for c in candidates if c.exists()), None)
        if not found:
            logger.error("Load failed: not found")
            return False

        with self._rw_lock.write_lock():
            try:
                open_func = gzip.open if str(found).endswith('.gz') else open
                with open_func(found, 'rb') as f:
                    state = pickle.load(f)

                self._next_idx = state.get('next_idx', 0)
                self._size = state.get('size', 0)
                self._max_priority = state.get('max_priority', 1.0)
                self.metrics = state.get('metrics', self.metrics)

                if state.get('rng_state') is not None:
                    try:
                        self._rng.bit_generator.state = state['rng_state']
                    except Exception:
                        pass

                ts = np.asarray(state.get('tree_sum'))
                if ts is not None:
                    if ts.shape == self._it_sum._value.shape:
                        self._it_sum._value[:] = ts
                    else:
                        if ts.size >= self._it_sum._capacity:
                            cap = ts.size // 2
                            leaves = ts[cap: cap + min(self._size, cap)]
                            self._it_sum.update_batch(list(range(len(leaves))), leaves.tolist())

                tm = np.asarray(state.get('tree_min'))
                if tm is not None:
                    if tm.shape == self._it_min._value.shape:
                        self._it_min._value[:] = tm
                    else:
                        if tm.size >= self._it_min._capacity:
                            cap = tm.size // 2
                            leaves = tm[cap: cap + min(self._size, cap)]
                            self._it_min.update_batch(list(range(len(leaves))), leaves.tolist())

                saved_cfg = state.get('config', {})
                if saved_cfg:
                    for key in ['obs_shape', 'goal_shape', 'action_shape']:
                        if saved_cfg.get(key): setattr(self.config, key, tuple(saved_cfg[key]))

                if self.config.use_memmap:
                    self._init_storage(self.config.obs_shape, self.config.goal_shape, self.config.action_shape,
                                       self.config.action_dtype, self.config.reward_dtype)
                else:
                    d = state.get('data', None)
                    if d is not None:
                        self.states = np.array(d['states'], copy=True)
                        self.goals = np.array(d['goals'], copy=True)
                        self.actions = np.array(d['actions'], copy=True)
                        self.rewards = np.array(d['rewards'], copy=True)
                        self.next_states = np.array(d['next_states'], copy=True)
                        self.dones = np.array(d['dones'], copy=True)
                        self._storage_initialized = True

                logger.info(f"Loaded from {found}")
                return True
            except Exception:
                logger.exception("Load failed")
                return False

    def validate(self) -> Dict[str, Any]:
        issues = []
        with self._rw_lock.read_lock():
            if self._size > self._maxsize: issues.append("Size > Capacity")

            try:
                if not np.isfinite(self._it_sum.reduce(0, self._size)): issues.append("Invalid total priority")
            except Exception as e:
                issues.append(f"Tree sum check: {e}")

            if self._storage_initialized:
                for name, arr in [('states', self.states), ('rewards', self.rewards)]:
                    if arr is None: issues.append(f"{name} is None")

                if self.config.use_memmap:
                    for name in ['states', 'goals', 'actions', 'rewards', 'next_states', 'dones']:
                        if not (self.config.memmap_dir / f"{name}.dat").exists():
                            issues.append(f"Missing memmap: {name}")

        return {'valid': len(issues) == 0, 'issues': issues, 'size': self._size}

    def get_sampling_bias_stats(self) -> Dict[str, Any]:
        if self._size == 0: return {'available': False}
        with self._rw_lock.read_lock():
            priorities = [float(self._it_sum[i]) for i in range(self._size)]
            return {'mean': float(np.mean(priorities)), 'max': float(np.max(priorities))}