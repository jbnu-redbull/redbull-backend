import os
import sys
import asyncio
import unittest
from datetime import datetime

# 경로 설정 (app/ 상위 경로를 PYTHONPATH에 추가)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.repository.client import SyncSQLiteClient, AsyncSQLiteClient
from app.repository.table import SyncTableManager, AsyncTableManager
from app.repository.schema import STTResult
from app.repository import cleanup_sync_table_manager, cleanup_async_table_manager

class TestSyncTableManager(unittest.TestCase):
    def setUp(self):
        self.client = SyncSQLiteClient(":memory:")
        self.client.connect()
        self.manager = SyncTableManager(self.client)
        self.manager.create_tables()

    def tearDown(self):
        self.client.close()
        cleanup_sync_table_manager()
    
    def test_create_table(self):
        tables = self.manager.get_tables()
        self.assertEqual(tables, ["stt_result", "meeting_analysis", "redmine_issue_log", "log"])

    def test_duplicated_table_creation(self):
        self.manager.create_tables()
        tables = self.manager.get_tables()
        self.assertEqual(tables, ["stt_result", "meeting_analysis", "redmine_issue_log", "log"])

    def test_insert_and_get(self):
        model = STTResult(audio_file_path="test.wav", stt_text="hello")
        row_id = self.manager.insert("stt_result", model)
        result = self.manager.get_by_id("stt_result", row_id)

        self.assertEqual(result["audio_file_path"], "test.wav")
        self.assertEqual(result["stt_text"], "hello")

    def test_update_and_delete(self):
        model = STTResult(audio_file_path="update.wav", stt_text="original")
        row_id = self.manager.insert("stt_result", model)

        self.manager.update_by_id("stt_result", row_id, {"stt_text": "updated"})
        updated = self.manager.get_by_id("stt_result", row_id)
        self.assertEqual(updated["stt_text"], "updated")

        self.manager.delete_by_id("stt_result", row_id)
        self.assertIsNone(self.manager.get_by_id("stt_result", row_id))


class TestAsyncTableManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncSQLiteClient(":memory:")
        await self.client.connect()
        self.manager = AsyncTableManager(self.client)
        await self.manager.create_tables()

    async def asyncTearDown(self):
        await self.client.close()
        await cleanup_async_table_manager()

    async def test_insert_and_get(self):
        model = STTResult(audio_file_path="async.wav", stt_text="hello async")
        row_id = await self.manager.insert("stt_result", model)
        result = await self.manager.get_by_id("stt_result", row_id)

        self.assertEqual(result["audio_file_path"], "async.wav")
        self.assertEqual(result["stt_text"], "hello async")

    async def test_update_and_delete(self):
        model = STTResult(audio_file_path="async.wav", stt_text="to be updated")
        row_id = await self.manager.insert("stt_result", model)

        await self.manager.update_by_id("stt_result", row_id, {"stt_text": "now updated"})
        updated = await self.manager.get_by_id("stt_result", row_id)
        self.assertEqual(updated["stt_text"], "now updated")

        await self.manager.delete_by_id("stt_result", row_id)
        deleted = await self.manager.get_by_id("stt_result", row_id)
        self.assertIsNone(deleted)

if __name__ == "__main__":
    unittest.main()
