# tests/test_backup_policy.py
import numpy as np
from sfc_backup_system.backup_policy import BackupPolicy


# tests/test_backup_policy.py

def test_backup_policy_never_fail(dummy_expert, sample_network_state, sample_request):
    policy = BackupPolicy(
        expert=dummy_expert,
        n=28,
        L=1000,
        K_vnf=10,
        dc_nodes=[3, 6, 9, 12, 15, 18, 21, 24, 27]
    )

    policy.set_current_tree({"nodes": [1, 5, 7, 10]})
    policy.set_current_request(sample_request)

    plan = policy.get_backup_plan(
        goal_dest_idx=0,  # dest[0] = 10
        network_state=sample_network_state
    )

    # 正确的断言：必须可行 + 终点正确 + 路径合法
    assert plan["feasible"] is True
    assert len(plan["nodes"]) >= 2
    assert plan["nodes"][-1] == 10  # 终点必须是目标节点
    assert plan["nodes"][0] in policy.current_tree["nodes"]  # 起点必须在当前树上（就近接入）
    assert isinstance(plan["tree"], np.ndarray)
    assert isinstance(plan["hvt"], np.ndarray)
    assert plan["score"] > -1000  # 评分不能太离谱

def test_backup_policy_dest_idx1(dummy_expert, sample_network_state, sample_request):
    policy = BackupPolicy(dummy_expert, n=28, L=1000, K_vnf=10)
    policy.set_current_tree({"nodes": [1, 5, 10]})
    policy.set_current_request(sample_request)

    plan = policy.get_backup_plan(goal_dest_idx=1, network_state=sample_network_state)  # dest[1]=20
    assert plan["feasible"] is True
    assert plan["nodes"][-1] == 20