# tests/test_path_finder.py
from sfc_backup_system.path_finder import PathFinder


def test_get_k_path(dummy_expert):
    finder = PathFinder(dummy_expert, n=28, max_k=5)

    nodes, dist, links = finder.get_k_path(1, 10, k=1)
    assert nodes == [1, 5, 10]
    assert len(links) == 2

    nodes2, _, _ = finder.get_k_path(1, 10, k=2)
    assert nodes2 == [1, 3, 8, 10]

    none_nodes, _, _ = finder.get_k_path(1, 10, k=3)
    assert none_nodes is None


def test_find_any_path(dummy_expert):
    finder = PathFinder(dummy_expert, n=28)
    nodes, links = finder.find_any_path(1, 10)
    assert nodes == [1, 5, 10]




def test_compose_via_relay(dummy_expert):
    finder = PathFinder(dummy_expert, n=28)
    nodes, links = finder.compose_via_relay(src=1, relay=5, dst=20)

    # 现在一定能拼接成功
    assert nodes is not None
    assert links is not None
    assert nodes[0] == 1
    assert nodes[1] == 5  # 中转节点在路径中
    assert 20 in nodes  # 终点在路径中
    assert len(links) >= 2
    print(f"Composed path: {' → '.join(map(str, nodes))} (links: {links})")