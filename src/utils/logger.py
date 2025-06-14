"""
日志工具类 - Logger Utility Class
提供统一的日志管理功能
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class Logger:
    """
    统一的日志管理工具类
    支持多级日志、文件输出、控制台输出等功能
    """
    
    _instances = {}  # 单例模式存储
    
    def __new__(cls, name: str = "InteractAI", **kwargs):
        """单例模式，确保同名logger只创建一次"""
        if name not in cls._instances:
            cls._instances[name] = super(Logger, cls).__new__(cls)
            cls._instances[name]._initialized = False
        return cls._instances[name]
    
    def __init__(self, 
                 name: str = "InteractAI",
                 log_dir: str = "log",
                 log_level: int = logging.INFO,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 console_output: bool = True):
        """
        初始化日志器
        
        Args:
            name: 日志器名称
            log_dir: 日志目录路径
            log_level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            max_file_size: 最大文件大小 (字节)
            backup_count: 备份文件数量
            console_output: 是否输出到控制台
        """
        if self._initialized:
            return
            
        self.name = name
        self.log_dir = log_dir
        self.log_level = log_level
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.console_output = console_output
        
        self._setup_logger()
        self._initialized = True
        
    def _setup_logger(self):
        """设置日志器"""
        # 获取项目根目录路径
        current_file = Path(__file__).resolve()  # 当前文件的绝对路径
        project_root = current_file.parent.parent.parent  # 从 src/utils/logger.py 回到项目根目录
        
        # 创建日志目录路径
        if os.path.isabs(self.log_dir):
            # 如果是绝对路径，直接使用
            log_path = Path(self.log_dir)
        else:
            # 如果是相对路径，相对于项目根目录
            log_path = project_root / self.log_dir
            
        log_path.mkdir(exist_ok=True, parents=True)
        
        # 创建日志器
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.log_level)
        
        # 清除已有的处理器（避免重复）
        self.logger.handlers.clear()
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 文件处理器 - 使用RotatingFileHandler支持文件轮转
        from logging.handlers import RotatingFileHandler
        log_file = log_path / f"{self.name.lower()}.log"
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 控制台处理器
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            # 控制台使用简化格式
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
        # 记录初始化信息
        self.info(f"日志系统初始化完成 - {self.name}")
        self.info(f"日志文件: {log_file.resolve()}")  # 显示绝对路径
        self.info(f"日志目录: {log_path.resolve()}")  # 显示绝对目录路径
        self.info(f"日志级别: {logging.getLevelName(self.log_level)}")
        
    def debug(self, message: str, extra_data: Optional[dict] = None):
        """调试级别日志"""
        self._log(logging.DEBUG, message, extra_data)
        
    def info(self, message: str, extra_data: Optional[dict] = None):
        """信息级别日志"""
        self._log(logging.INFO, message, extra_data)
        
    def warning(self, message: str, extra_data: Optional[dict] = None):
        """警告级别日志"""
        self._log(logging.WARNING, message, extra_data)
        
    def error(self, message: str, extra_data: Optional[dict] = None):
        """错误级别日志"""
        self._log(logging.ERROR, message, extra_data)
        
    def critical(self, message: str, extra_data: Optional[dict] = None):
        """严重错误级别日志"""
        self._log(logging.CRITICAL, message, extra_data)
        
    def _log(self, level: int, message: str, extra_data: Optional[dict] = None):
        """内部日志记录方法"""
        if extra_data:
            # 如果有额外数据，格式化为字符串
            extra_str = " | ".join([f"{k}={v}" for k, v in extra_data.items()])
            message = f"{message} | {extra_str}"
        
        self.logger.log(level, message)
        
    def log_interaction(self, interaction_type: str, details: dict):
        """记录用户交互日志"""
        self.info(f"用户交互 - {interaction_type}", details)
        
    def log_emotion_change(self, old_emotion: float, new_emotion: float, 
                          old_state: str, new_state: str):
        """记录情绪变化日志"""
        details = {
            "old_emotion": old_emotion,
            "new_emotion": new_emotion,
            "old_state": old_state,
            "new_state": new_state,
            "change": new_emotion - old_emotion
        }
        self.info("情绪状态变化", details)
        
    def log_animation_event(self, event_type: str, parameters: dict):
        """记录动画事件日志"""
        self.info(f"动画事件 - {event_type}", parameters)
        
    def log_performance(self, function_name: str, execution_time: float, 
                       additional_metrics: Optional[dict] = None):
        """记录性能日志"""
        metrics = {"execution_time_ms": execution_time * 1000}
        if additional_metrics:
            metrics.update(additional_metrics)
        self.debug(f"性能监控 - {function_name}", metrics)
        
    def log_error_with_traceback(self, error: Exception, context: str = ""):
        """记录错误和堆栈跟踪"""
        import traceback
        error_details = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc()
        }
        self.error("异常发生", error_details)
        
    def set_level(self, level: Union[int, str]):
        """动态设置日志级别"""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
        self.info(f"日志级别已更改为: {logging.getLevelName(level)}")
        
    def create_session_log(self, session_id: str = None):
        """创建会话日志文件"""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 使用与主日志相同的目录路径逻辑
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        
        if os.path.isabs(self.log_dir):
            log_path = Path(self.log_dir)
        else:
            log_path = project_root / self.log_dir
            
        session_file = log_path / f"session_{session_id}.log"
        session_handler = logging.FileHandler(session_file, encoding='utf-8')
        session_handler.setLevel(self.log_level)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        session_handler.setFormatter(formatter)
        
        self.logger.addHandler(session_handler)
        self.info(f"会话日志已创建: {session_file}")
        return session_file


# 便捷的全局日志器实例
def get_logger(name: str = "InteractAI", **kwargs) -> Logger:
    """
    获取日志器实例
    
    Args:
        name: 日志器名称
        **kwargs: 其他初始化参数
        
    Returns:
        Logger实例
    """
    return Logger(name, **kwargs)


# 装饰器：自动记录函数执行时间
def log_performance(logger_name: str = "InteractAI"):
    """性能监控装饰器"""
    import time  # 在装饰器内部导入
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.log_performance(func.__name__, execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.log_performance(func.__name__, execution_time, 
                                     {"status": "error"})
                logger.log_error_with_traceback(e, f"执行函数 {func.__name__}")
                raise
        return wrapper
    return decorator


# 装饰器：自动记录函数调用
def log_function_call(logger_name: str = "InteractAI", level: str = "DEBUG"):
    """函数调用记录装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            # 记录函数调用
            getattr(logger, level.lower())(
                f"调用函数 {func.__name__}",
                {"args_count": len(args), "kwargs_count": len(kwargs)}
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator
