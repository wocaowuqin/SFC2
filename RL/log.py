# -*- coding: utf-8 -*-
# @File    : log.py
# @Date    : 2022-05-18
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
# @From    :
import logging
import time
import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd


class MyLog:
    """
    增强版日志记录工具类

    示例:
        mylog = MyLog(Path(__file__), name="SFC_Env")
        logger = mylog.logger
        logger.info("信息日志")
        logger.warning("警告日志")
    """

    def __init__(self,
                 path: Path,
                 filesave: bool = False,
                 consoleprint: bool = True,
                 name: Optional[str] = None,
                 log_level: int = logging.INFO,
                 max_log_files: int = 10):
        """
        初始化日志记录器

        Args:
            path: 运行日志的当前文件 Path(__file__)
            filesave: 是否存储日志到文件
            consoleprint: 是否打印到终端
            name: 日志记录器名称
            log_level: 日志级别
            max_log_files: 最大保留日志文件数
        """
        self.path = path
        self.filesave = filesave
        self.consoleprint = consoleprint
        self.name = name or path.stem
        self.log_level = log_level
        self.max_log_files = max_log_files

        # 日志格式
        self.formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 创建日志记录器
        self.logger = logging.getLogger(self.name)
        self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        # 避免重复添加handler
        if self.logger.handlers:
            self.logger.handlers.clear()

        self.logger.setLevel(self.log_level)
        self.logger.propagate = False

        # 创建日志目录
        self.log_path = Path.joinpath(self.path.parent, 'Logs')
        self._create_log_dir()

        # 生成日志文件名
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if self.name:
            self.log_file = self.log_path / f"{self.name}_{timestamp}.log"
        else:
            self.log_file = self.log_path / f"{self.path.stem}_{timestamp}.log"

        # 添加处理器
        if self.filesave:
            self._add_file_handler()

        if self.consoleprint:
            self._add_console_handler()

        # 清理旧日志文件
        self._cleanup_old_logs()

    def _create_log_dir(self):
        """创建日志目录"""
        try:
            self.log_path.mkdir(exist_ok=True)
        except Exception as e:
            print(f"创建日志目录失败: {e}")
            # 回退到当前目录
            self.log_path = Path.cwd()

    def _add_file_handler(self):
        """添加文件处理器"""
        try:
            fh = logging.FileHandler(
                self.log_file,
                mode='w',
                encoding='utf-8'
            )
            fh.setLevel(self.log_level)
            fh.setFormatter(self.formatter)
            self.logger.addHandler(fh)
            self.logger.info(f"日志文件已创建: {self.log_file}")
        except Exception as e:
            print(f"创建文件处理器失败: {e}")

    def _add_console_handler(self):
        """添加控制台处理器"""
        ch = logging.StreamHandler()
        ch.setLevel(self.log_level)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

    def _cleanup_old_logs(self):
        """清理旧日志文件"""
        try:
            if not self.log_path.exists():
                return

            log_files = sorted(
                self.log_path.glob("*.log"),
                key=os.path.getmtime,
                reverse=True
            )

            # 删除超过数量限制的旧文件
            for old_log in log_files[self.max_log_files:]:
                try:
                    old_log.unlink()
                    self.logger.debug(f"删除旧日志文件: {old_log}")
                except Exception as e:
                    self.logger.warning(f"删除旧日志文件失败 {old_log}: {e}")

        except Exception as e:
            self.logger.warning(f"清理旧日志文件失败: {e}")

    def set_level(self, level: int):
        """动态设置日志级别"""
        self.log_level = level
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)

    def get_log_file_path(self) -> Path:
        """获取日志文件路径"""
        return self.log_file

    def pd_to_csv(self, dataframe: pd.DataFrame, filename: Optional[str] = None):
        """
        保存DataFrame到CSV文件

        Args:
            dataframe: 要保存的DataFrame
            filename: 文件名（不包含扩展名）
        """
        try:
            if filename:
                csv_file = self.log_path / f"{filename}.csv"
            else:
                csv_file = self.log_path / f"{self.path.stem}_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.csv"

            dataframe.to_csv(csv_file, index=False, encoding='utf-8')
            self.logger.info(f"CSV文件已保存: {csv_file}")
        except Exception as e:
            self.logger.error(f"保存CSV文件失败: {e}")

    def save_dict_to_csv(self, data_dict: dict, filename: Optional[str] = None):
        """
        保存字典数据到CSV文件

        Args:
            data_dict: 要保存的字典数据
            filename: 文件名（不包含扩展名）
        """
        try:
            df = pd.DataFrame([data_dict])
            self.pd_to_csv(df, filename)
        except Exception as e:
            self.logger.error(f"保存字典到CSV失败: {e}")

    def close(self):
        """关闭日志记录器"""
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


# 便捷函数
def create_logger(
        module_path: str,
        name: Optional[str] = None,
        level: int = logging.INFO,
        save_to_file: bool = True
) -> logging.Logger:
    """
    快速创建日志记录器的便捷函数

    Args:
        module_path: 模块路径，通常使用 __file__
        name: 日志记录器名称
        level: 日志级别
        save_to_file: 是否保存到文件

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    mylog = MyLog(
        path=Path(module_path),
        name=name,
        filesave=save_to_file,
        consoleprint=True,
        log_level=level
    )
    return mylog.logger


# 使用示例
if __name__ == "__main__":
    # 使用方法1: 直接实例化
    mylog = MyLog(Path(__file__), name="TestLogger")
    logger = mylog.logger

    logger.debug("调试信息")
    logger.info("普通信息")
    logger.warning("警告信息")
    logger.error("错误信息")

    # 保存数据到CSV
    import pandas as pd

    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    mylog.pd_to_csv(df, "test_data")

    # 使用方法2: 使用便捷函数
    test_logger = create_logger(__file__, "QuickLogger")
    test_logger.info("这是通过便捷函数创建的日志记录器")

    mylog.close()