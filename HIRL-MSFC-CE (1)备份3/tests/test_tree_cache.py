# tests/test_tree_cache.py
from sfc_backup_system.tree_cache import TreeCache


def test_tree_cache_invalidation():
    cache = TreeCache()
    cache.set(1, 10, 5)
    assert cache.get(1, 10) == 5

    cache.invalidate()
    assert cache.get(1, 10) is None