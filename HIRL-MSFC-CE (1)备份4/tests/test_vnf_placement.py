# tests/test_vnf_placement.py
from sfc_backup_system.vnf_placement import VNFPlacement


def test_simple_round_robin():
    placement = VNFPlacement.simple_round_robin(
        vnf_sequence=[0, 1, 2],
        path_nodes=[1, 3, 6, 9, 12],
        is_dc_fn=lambda n: n % 3 == 0
    )
    # DC 节点: 3,6,9,12 → 轮询
    assert placement == {0: 3, 1: 6, 2: 9}


def test_resource_aware_success(sample_network_state):
    def is_dc(n): return n % 3 == 0

    placement = VNFPlacement.resource_aware(
        vnf_sequence=[0, 1],
        path_nodes=[3, 6, 9],
        network_state=sample_network_state,
        is_dc_fn=is_dc,
    )
    assert placement is not None
    assert len(placement) == 2