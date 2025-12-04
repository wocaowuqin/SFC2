# tests/conftest.py
import sys
from pathlib import Path

# 关键：把项目根目录加入 sys.path（这样所有测试都能 import sfc_backup_system）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from typing import Dict, Any


class DummyExpert:
    """模拟 MSFCE_Solver，必须实现 _get_path_info"""
    def __init__(self):
        # 关键路径必须完整覆盖 compose_via_relay 测试
        self.path_db = {
            (1, 10): [([1, 5, 10], 2, [11, 52]), ([1, 3, 8, 10], 3, [5, 38, 51])],
            (1, 20): [([1, 7, 15, 20], 3, [17, 73, 99])],
            (10, 20): [([10, 15, 20], 2, [52, 99])],
            (5, 20): [([5, 15, 20], 2, [45, 99])],
            (1, 5): [([1, 5], 1, [11])],                 # 必须有
            (5, 10): [([5, 10], 1, [52])],
        }
        self.k_path = 5
        self.current_request = {"source": 1}

    def _get_path_info(self, src: int, dst: int, k: int):
        entry = self.path_db.get((src, dst), [])
        if k <= len(entry):
            return entry[k-1]
        # 重要：返回三个 None，而不是 raise
        return None, None, None


@pytest.fixture
def dummy_expert():
    return DummyExpert()


@pytest.fixture
def sample_network_state() -> Dict[str, Any]:
    return {
        "cpu": {i: 1500 + i*10 for i in range(28)},      # 0-based
        "mem": {i: 800 + i*20 for i in range(28)},
        # 改成 list，避免 “truth value of array” 错误
        "bw": [100.0] * 1000,
        "hvt": np.zeros((28, 10), dtype=int),
        "tree_nodes": [1, 5, 7, 10, 15],
    }


@pytest.fixture
def sample_request():
    return {
        "source": 1,
        "dest": [10, 20],
        "vnf": [0, 1, 2],
        "cpu_origin": [200, 300, 250],
        "memory_origin": [400, 500, 450],
        "bw_origin": 50.0,
    }