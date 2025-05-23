import unittest
import logging
import os
from datetime import datetime, timezone, timedelta
import time
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logger import initialize_logging
from app.logger.settings import LoggerSettings, logger_setup, set_logger_settings
from app.logger.config import disable_logging
from app.logger.queue import is_queue_listener_running, stop_queue_listener, start_queue_listener
from app.repository import get_sync_table_manager

# 전역 변수로 keep_logs 설정
keep_logs = False

class TestLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 시작 전에 데이터베이스 초기화"""
        manager = get_sync_table_manager()
        print(f"manager: {manager.logging}")
        manager.drop_tables()
        manager.create_tables()

    def setUp(self):
        """각 테스트 시작 전 설정"""
        # 데이터베이스 초기화
        manager = get_sync_table_manager()
        manager.drop_tables()
        manager.create_tables()

        # 테스트용 로거 설정
        self.test_settings = LoggerSettings(
            logger_level="DEBUG",
            logger_db_path="./database.db",
            logger_db_logging=True
        )

        print(f"test_settings: {self.test_settings}")
        
        # 설정 적용
        self.logger_settings = set_logger_settings(self.test_settings)
        
        # 디버깅: 설정 적용 후 상태 확인
        print("\nAfter settings applied:")
        print(f"logger_db_logging: {self.logger_settings.logger_db_logging}")
        print(f"logger_level: {self.logger_settings.logger_level}")
        print(f"logger_db_path: {self.logger_settings.logger_db_path}")
        
        # 설정이 제대로 적용되었는지 확인
        self.assertTrue(self.logger_settings.logger_db_logging, "DB logging setting was not applied")
        
        # 큐 리스너 시작
        if not is_queue_listener_running():
            start_queue_listener()
        
        # 테스트용 로거 생성
        self.logger = logging.getLogger("app.logger_test")
        self.logger.setLevel(logging.DEBUG)
        
        # 디버깅: 로거 핸들러 확인
        print("\nLogger handlers after setup:")
        for handler in self.logger.handlers:
            print(f"- {handler.__class__.__name__}")
        
        # 테스트 메시지
        self.test_message = "Test log message"
        
        # KST 시간 설정
        self.KST = timezone(timedelta(hours=9))

    def tearDown(self):
        """각 테스트 종료 후 정리"""
        # 큐 리스너 정리
        if is_queue_listener_running():
            stop_queue_listener()

    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 종료 후 정리"""
        if not keep_logs:
            manager = get_sync_table_manager()
            manager.drop_tables()
            manager.create_tables()

    def test_disable_logging(self):
        """로깅 비활성화가 제대로 동작하는지 테스트"""
        # 로깅 비활성화
        disable_logging(self.logger)
        
        # 로그 메시지 생성
        self.logger.info(self.test_message)
        
        # 데이터베이스에서 로그 확인
        manager = get_sync_table_manager()
        logs = manager.get_all("log")
        
        # 로그가 저장되지 않았는지 확인
        self.assertEqual(len(logs), 0)

    def test_database_logging(self):
        """데이터베이스에 로그가 제대로 저장되는지 테스트"""
        # 큐 리스너가 실행 중인지 확인
        self.assertTrue(is_queue_listener_running(), "Queue listener is not running")
        
        # DB 핸들러가 추가되었는지 확인
        self.assertTrue(self.logger_settings.logger_db_logging, "DB logging is not enabled")
        
        # 디버깅: 현재 설정 상태 확인
        print("\nCurrent settings:")
        print(f"logger_db_logging: {self.logger_settings.logger_db_logging}")
        print(f"logger_level: {self.logger_settings.logger_level}")
        print(f"logger_db_path: {self.logger_settings.logger_db_path}")
        
        # 로그 메시지 생성
        self.logger.info(self.test_message)
        
        # 디버깅: 현재 로거의 핸들러 확인
        print("\nCurrent logger handlers:")
        for handler in self.logger.handlers:
            print(f"- {handler.__class__.__name__}")
        
        # 디버깅: 큐 리스너 상태 확인
        print(f"\nQueue listener running: {is_queue_listener_running()}")
        
        # 로그가 데이터베이스에 저장될 때까지 대기 (최대 1초)
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            manager = get_sync_table_manager()
            logs = manager.get_all("log")
            if len(logs) > 0:
                break
            time.sleep(0.1)
            print(f"Waiting for logs... ({time.time() - start_time:.1f}s)")
        
        # 데이터베이스에서 로그 확인
        manager = get_sync_table_manager()
        logs = manager.get_all("log")
        
        # 디버깅: 데이터베이스 상태 확인
        print(f"\nDatabase path: {self.logger_settings.logger_db_path}")
        print(f"Number of logs in database: {len(logs)}")
        if len(logs) > 0:
            print(f"Latest log: {logs[-1]}")
        
        # 로그가 존재하는지 확인
        self.assertTrue(len(logs) > 0, "No logs found in database")
        
        # 가장 최근 로그 확인
        latest_log = logs[-1]
        self.assertEqual(latest_log["message"], self.test_message, 
                        f"Expected message '{self.test_message}', got '{latest_log['message']}'")
        self.assertEqual(latest_log["level"], "INFO", 
                        f"Expected level 'INFO', got '{latest_log['level']}'")
        
        # 타임스탬프가 KST인지 확인
        timestamp = datetime.fromisoformat(latest_log["timestamp"].replace('Z', '+00:00'))
        self.assertEqual(timestamp.tzinfo, self.KST, 
                        f"Expected timezone {self.KST}, got {timestamp.tzinfo}")

    def test_queue_listener_control(self):
        """큐 리스너 제어가 제대로 동작하는지 테스트"""
        # 리스너 시작
        start_queue_listener()
        self.assertTrue(is_queue_listener_running(), "Queue listener failed to start")
        
        # 리스너 중지
        stop_queue_listener()
        self.assertFalse(is_queue_listener_running(), "Queue listener failed to stop")

    def test_log_levels(self):
        """다양한 로그 레벨이 제대로 저장되는지 테스트"""
        # 큐 리스너가 실행 중인지 확인
        self.assertTrue(is_queue_listener_running(), "Queue listener is not running")
        
        # 각 로그 레벨별 메시지 생성
        self.logger.debug("Debug message")
        self.logger.info("Info message")
        self.logger.warning("Warning message")
        self.logger.error("Error message")
        self.logger.critical("Critical message")
        
        # 로그가 데이터베이스에 저장될 때까지 대기 (최대 1초)
        max_wait = 1.0
        start_time = time.time()
        while time.time() - start_time < max_wait:
            manager = get_sync_table_manager()
            logs = manager.get_all("log")
            if len(logs) >= 5:  # 모든 레벨의 로그가 저장되었는지 확인
                break
            time.sleep(0.1)
        
        # 데이터베이스에서 로그 확인
        manager = get_sync_table_manager()
        logs = manager.get_all("log")
        
        # 각 로그 레벨별 메시지가 존재하는지 확인
        levels = [log["level"] for log in logs]
        self.assertIn("DEBUG", levels, "DEBUG level log not found")
        self.assertIn("INFO", levels, "INFO level log not found")
        self.assertIn("WARNING", levels, "WARNING level log not found")
        self.assertIn("ERROR", levels, "ERROR level log not found")
        self.assertIn("CRITICAL", levels, "CRITICAL level log not found")

if __name__ == '__main__':
    # 커맨드 라인 인자에서 --keep-logs 옵션 확인
    if '--keep-logs' in sys.argv:
        keep_logs = True
        sys.argv.remove('--keep-logs')
    
    unittest.main()
