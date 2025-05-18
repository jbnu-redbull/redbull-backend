from app.repository.table import AsyncTableManager, TABLE_DEFINITIONS

# stt_result 테이블 정의 추가
TABLE_DEFINITIONS["stt_result"] = """
    CREATE TABLE IF NOT EXISTS stt_result (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_file_path TEXT NOT NULL,
        stt_text TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
"""

class STTTableManager(AsyncTableManager):
    """STT 테이블 관리자 클래스"""
    
    async def create_tables(self):
        """Create necessary tables if they don't exist."""
        for table, sql in TABLE_DEFINITIONS.items():
            await self.client.execute(sql)