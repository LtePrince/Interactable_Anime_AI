#!/usr/bin/env python3
"""
日志工具类测试脚本
Logger Utility Class Test Script

测试日志系统的各种功能，包括：
- 基本日志级别
- 专用日志方法
- 性能监控装饰器
- 错误处理
- 文件路径正确性
"""

import sys
import os
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.logger import get_logger, log_performance, log_function_call


class LoggerTest:
    """日志测试类"""
    
    def __init__(self):
        self.test_logger = get_logger("LoggerTest", log_dir="log")
        self.passed_tests = 0
        self.total_tests = 0
        
    def run_test(self, test_name: str, test_func):
        """运行单个测试"""
        self.total_tests += 1
        try:
            print(f"\n🧪 测试 {self.total_tests}: {test_name}")
            test_func()
            self.passed_tests += 1
            print(f"✅ {test_name} - 通过")
            self.test_logger.info(f"测试通过: {test_name}")
        except Exception as e:
            print(f"❌ {test_name} - 失败: {e}")
            self.test_logger.error(f"测试失败: {test_name}", {"error": str(e)})
    
    def test_basic_logging_levels(self):
        """测试基本日志级别"""
        logger = self.test_logger
        
        logger.debug("这是调试信息 - DEBUG")
        logger.info("这是普通信息 - INFO")
        logger.warning("这是警告信息 - WARNING")
        logger.error("这是错误信息 - ERROR")
        logger.critical("这是严重错误信息 - CRITICAL")
        
        print("  基本日志级别测试完成")
    
    def test_logging_with_extra_data(self):
        """测试带额外数据的日志"""
        logger = self.test_logger
        
        extra_data = {
            "user_id": 12345,
            "action": "test_action",
            "timestamp": time.time(),
            "session_id": "test_session_001"
        }
        
        logger.info("带额外数据的日志测试", extra_data)
        logger.warning("警告级别的额外数据测试", {"warning_code": 404, "details": "Not Found"})
        
        print("  额外数据日志测试完成")
    
    def test_specialized_logging_methods(self):
        """测试专用日志方法"""
        logger = self.test_logger
        
        # 测试交互日志
        logger.log_interaction("点击测试", {
            "position": (150, 250),
            "button": "left",
            "duration": 0.123,
            "target": "emotion_image"
        })
        
        # 测试情绪变化日志
        logger.log_emotion_change(5.0, 8.5, "neutral", "happy")
        logger.log_emotion_change(8.5, 3.2, "happy", "angry")
        
        # 测试动画事件日志
        logger.log_animation_event("跳跃动画", {
            "height": 30,
            "duration": 25,
            "type": "bounce",
            "trigger": "emotion_change"
        })
        
        logger.log_animation_event("淡入淡出", {
            "start_alpha": 255,
            "end_alpha": 128,
            "duration": 15,
            "easing": "linear"
        })
        
        # 测试性能日志
        logger.log_performance("render_frame", 0.0167, {
            "fps": 60,
            "objects_rendered": 5,
            "memory_usage": "45MB"
        })
        
        print("  专用日志方法测试完成")
    
    def test_error_logging(self):
        """测试错误记录功能"""
        logger = self.test_logger
        
        # 测试普通错误记录
        try:
            result = 10 / 0
        except ZeroDivisionError as e:
            logger.log_error_with_traceback(e, "测试除零错误")
        
        # 测试文件不存在错误
        try:
            with open("non_existent_file.txt", 'r') as f:
                content = f.read()
        except FileNotFoundError as e:
            logger.log_error_with_traceback(e, "测试文件不存在错误")
        
        print("  错误记录测试完成")
    
    def test_log_level_changes(self):
        """测试动态日志级别变更"""
        logger = self.test_logger
        
        # 测试设置为DEBUG级别
        logger.set_level("DEBUG")
        logger.debug("DEBUG级别测试信息")
        
        # 测试设置为WARNING级别
        logger.set_level("WARNING")
        logger.info("这条INFO信息应该不会显示")
        logger.warning("这条WARNING信息应该显示")
        
        # 恢复为INFO级别
        logger.set_level("INFO")
        logger.info("恢复INFO级别测试")
        
        print("  日志级别变更测试完成")
    
    def test_file_path_correctness(self):
        """测试日志文件路径正确性"""
        # 检查日志文件是否在正确的目录中
        expected_log_dir = project_root / "log"
        expected_log_file = expected_log_dir / "loggertest.log"
        
        if expected_log_dir.exists():
            print(f"  ✅ 日志目录存在: {expected_log_dir}")
        else:
            raise Exception(f"日志目录不存在: {expected_log_dir}")
        
        if expected_log_file.exists():
            print(f"  ✅ 日志文件存在: {expected_log_file}")
        else:
            raise Exception(f"日志文件不存在: {expected_log_file}")
        
        # 检查文件内容
        with open(expected_log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "日志系统初始化完成" in content:
                print("  ✅ 日志文件内容正确")
            else:
                raise Exception("日志文件内容不完整")
    
    def test_session_logging(self):
        """测试会话日志功能"""
        logger = self.test_logger
        
        # 创建会话日志
        session_file = logger.create_session_log("test_session_001")
        logger.info("这是会话日志测试消息")
        
        if session_file.exists():
            print(f"  ✅ 会话日志文件创建成功: {session_file}")
        else:
            raise Exception(f"会话日志文件创建失败: {session_file}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始日志工具类完整测试...")
        print(f"📁 项目根目录: {project_root}")
        print(f"📝 预期日志目录: {project_root / 'log'}")
        
        # 运行各项测试
        self.run_test("基本日志级别", self.test_basic_logging_levels)
        self.run_test("额外数据日志", self.test_logging_with_extra_data)
        self.run_test("专用日志方法", self.test_specialized_logging_methods)
        self.run_test("错误记录", self.test_error_logging)
        self.run_test("日志级别变更", self.test_log_level_changes)
        self.run_test("文件路径正确性", self.test_file_path_correctness)
        self.run_test("会话日志", self.test_session_logging)
        
        # 输出测试结果
        print(f"\n📊 测试结果:")
        print(f"  总测试数: {self.total_tests}")
        print(f"  通过测试: {self.passed_tests}")
        print(f"  失败测试: {self.total_tests - self.passed_tests}")
        print(f"  成功率: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        if self.passed_tests == self.total_tests:
            print("🎉 所有测试通过!")
            self.test_logger.info("所有日志测试通过")
        else:
            print("⚠️  部分测试失败，请检查日志文件")
            self.test_logger.warning(f"测试完成，{self.total_tests - self.passed_tests}个测试失败")


@log_performance("LoggerTest")
def performance_test_function():
    """测试性能监控装饰器的函数"""
    import time
    time.sleep(0.01)  # 模拟10ms的工作
    result = sum(range(1000))  # 一些计算工作
    return result


@log_function_call("LoggerTest", "INFO")
def function_call_test(param1, param2="default"):
    """测试函数调用记录装饰器的函数"""
    return f"处理参数: {param1}, {param2}"


def test_decorators():
    """测试装饰器功能"""
    print("\n🎯 测试装饰器功能...")
    
    # 测试性能监控装饰器
    print("  测试性能监控装饰器...")
    result = performance_test_function()
    print(f"  函数返回值: {result}")
    
    # 测试函数调用记录装饰器
    print("  测试函数调用记录装饰器...")
    result = function_call_test("test_value", param2="test_param")
    print(f"  函数返回值: {result}")
    
    print("✅ 装饰器测试完成")


if __name__ == "__main__":
    try:
        # 创建并运行测试
        test_suite = LoggerTest()
        test_suite.run_all_tests()
        
        # 测试装饰器
        test_decorators()
        
        print(f"\n📁 日志文件位置: {project_root / 'log'}")
        print("💡 你可以查看日志文件来验证所有日志是否正确记录")
        
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
