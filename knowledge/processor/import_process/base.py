"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""
import datetime
from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
from knowledge.utils.task_util import add_running_task, add_done_task, add_node_duration

T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):

    name: str = "base_node"

    def __init__(self, config: Optional[ImportConfig] = None):

        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:

        task_id = state.get("task_id", "")
        try:
            # 1. 开始准备执行节点
            self.logger.info(f"--- {self.name} 开始 ---")
            if task_id:
                # 1.1 更新节点状态
                add_running_task(task_id, self.name)

            start=datetime.datetime.now()
            # 2. 执行节点
            result = self.process(state)
            duration=(datetime.datetime.now() - start).total_seconds()
            # 3. 执行节点成功
            self.logger.info(f"--- {self.name} 完成 ---")
            if task_id:
                # 3.1 更新状态
                add_done_task(task_id, self.name)
                add_node_duration(task_id, self.name, duration)
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:

        pass

    def log_step(self, step_name: str, message: str = ""):

        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)

# 配置日志格式
def setup_logging(level: int = logging.INFO):

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )