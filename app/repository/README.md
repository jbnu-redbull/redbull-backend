# SQLite Client & Table Manager

이 프로젝트는 `sqlite3` 및 `aiosqlite`를 활용하여 동기 및 비동기 SQLite 데이터베이스 작업을 수행하는 클라이언트와 테이블 관리 도구를 제공합니다. `Pydantic` 모델을 기반으로 자동 테이블 생성 및 CRUD 기능을 수행할 수 있습니다.

---

## 📁 모듈 구성

* `settings.py`: SQLite 설정을 관리합니다.
* `client.py`: 동기/비동기 SQLite 클라이언트를 정의합니다.
* `schema.py`: 테이블에 매핑되는 Pydantic 모델을 정의합니다.
* `table.py`: 테이블 생성 및 CRUD 로직을 관리하는 TableManager를 제공합니다.

---

## ⚙️ 설정 (`settings.py`)

### `SQLiteSettings`

* `db_path (str)`: SQLite DB 파일 경로 (기본: `./database.db`)
* `.env` 파일로부터 환경 변수 로드 가능

### `set_sqlite_settings()`

* 런타임에 새로운 설정 인스턴스를 주입할 수 있습니다.

---

## 🔌 클라이언트 (`client.py`)

### `SyncSQLiteClient`

동기 SQLite 클라이언트

* `connect()`

  * DB 연결을 생성하고 row factory를 설정합니다.

* `execute(query, params)`

  * SQL 실행 및 커밋 수행 후 cursor 반환

* `fetchall(query, params)`

  * `execute` 후 `fetchall` 결과 반환

* `close()`

  * 연결 종료

### `AsyncSQLiteClient`

비동기 SQLite 클라이언트 (aiosqlite 기반)

* `connect()`

  * 비동기 DB 연결 생성 및 row factory 설정

* `execute(query, params)`

  * SQL 실행 및 커밋 수행 후 cursor 반환

* `fetchall(query, params)`

  * `execute` 후 `fetchall` 결과 반환

* `close()`

  * 연결 종료 (await 필요)

---

## 🧱 스키마 모델 (`schema.py`)

### 공통

* `created_at` 필드는 기본값으로 KST 기준 현재 시간이 할당됩니다.

### `STTResult`

* `audio_file_path`: 오디오 파일 경로
* `stt_text`: STT 텍스트 결과

### `MeetingAnalysis`

* `transcript`: 회의록 정리본
* `summary`: 요약
* `issue_list`: JSON 이슈 목록
* `user_id`: 사용자 ID

### `RedmineIssueLog`

* `meeting_analysis_id`: 연결된 분석 ID
* `issue_text`: 이슈 내용
* `redmine_issue_id`: 레드마인 이슈 ID (옵션)
* `status`: 상태 (`created`, `failed`, `pending`)
* `timestamp`: 기록 시각

---

## 🧩 테이블 매니저 (`table.py`)

### SQL 생성 도구

* `generate_create_sql(model, table_name)`

  * Pydantic 모델을 기반으로 CREATE TABLE 구문 생성

---

### `SyncTableManager`

동기 방식 테이블 매니저

* `create_tables()` / `drop_tables()`
* `insert(table, model)` → row ID 반환
* `get_all(table)` → 전체 레코드 반환
* `get_by_id(table, id)` → 특정 레코드 반환
* `update_by_id(table, id, new_data)`
* `delete_by_id(table, id)`

### `AsyncTableManager`

비동기 방식 테이블 매니저

* 위의 동기 메서드와 동일하지만 모두 `await` 필요

---

## 📌 예시 코드

```python
# 동기 클라이언트 예시
from client import SyncSQLiteClient
from table import SyncTableManager
from schema import STTResult

client = SyncSQLiteClient()
client.connect()
manager = SyncTableManager(client)
manager.create_tables()

result = STTResult(audio_file_path="a.wav", stt_text="안녕하세요")
row_id = manager.insert("stt_result", result)
print("Inserted ID:", row_id)
client.close()
```

```python
# 비동기 클라이언트 예시
from client import AsyncSQLiteClient
from table import AsyncTableManager
from schema import STTResult

async def main():
    client = AsyncSQLiteClient()
    await client.connect()
    manager = AsyncTableManager(client)
    await manager.create_tables()

    result = STTResult(audio_file_path="a.wav", stt_text="안녕하세요")
    row_id = await manager.insert("stt_result", result)
    print("Inserted ID:", row_id)
    await client.close()
```

---

## 📎 참고

* `TABLE_MODELS` dict: 테이블 이름과 모델의 매핑
* 모든 테이블은 `id INTEGER PRIMARY KEY AUTOINCREMENT` 필드를 가집니다.

---

## ✅ ToDo

* Where 조건 기반 검색 기능 추가
* 스키마 마이그레이션 지원
* 인덱스 및 제약조건 생성 기능

---

## 🧑‍💻 개발자

이 저장소는 자동 테이블 생성을 통해 빠른 SQLite 기반 API 또는 테스트 환경 구축을 지원합니다.
