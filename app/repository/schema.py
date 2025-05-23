# app/repository/schema.py

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any
from datetime import datetime, timezone, timedelta

KST = timezone(timedelta(hours=9))

def now_kst() -> datetime:
    return datetime.now(tz=KST)

class STTResult(BaseModel):
    id: Optional[int] = None
    audio_file_path: str = Field(..., description="오디오 파일 경로")
    stt_text: str = Field(..., description="STT 변환 결과 텍스트")
    created_at: Optional[datetime] = Field(default_factory=now_kst)

class MeetingAnalysis(BaseModel):
    id: Optional[int] = None
    transcript: str = Field(..., description="STT 텍스트 정리본")
    summary: str = Field(..., description="요약본")
    issue_list: List[Any] = Field(..., description="이슈 리스트 (JSON 형태)")
    created_at: Optional[datetime] = Field(default_factory=now_kst)
    user_id: str = Field(..., description="작성자 또는 요청자 ID")

class RedmineIssueLog(BaseModel):
    id: Optional[int] = None
    meeting_analysis_id: int = Field(..., description="연결된 회의분석 ID")
    issue_text: str = Field(..., description="이슈 텍스트")
    redmine_issue_id: Optional[str] = Field(None, description="레드마인에서 반환받은 이슈 ID")
    status: Literal["created", "failed", "pending"] = Field(..., description="상태")
    timestamp: Optional[datetime] = Field(default_factory=now_kst)

class Log(BaseModel):
    id: Optional[int] = Field(None, description="로그 ID")
    module: str = Field(..., description="모듈 이름")
    level: str = Field(..., description="로그 레벨")
    message: str = Field(..., description="로그 메시지")
    timestamp: datetime = Field(..., description="로그 시간")

TABLE_MODELS = {
    "stt_result": STTResult,
    "meeting_analysis": MeetingAnalysis,
    "redmine_issue_log": RedmineIssueLog,
    "log": Log
}
