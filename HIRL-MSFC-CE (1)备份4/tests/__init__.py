# tests/conftest.py 或 tests/__init__.py
import sys
from pathlib import Path

# 自动把项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# 现在就可以正常导入
from sfc_backup_system.backup_policy import BackupPolicy