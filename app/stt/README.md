# Speech-to-Text (STT) Module

이 모듈은 오디오 파일을 텍스트로 변환하는 STT(Speech-to-Text) 기능을 제공합니다. Whisper ASR 모델을 사용하여 음성을 텍스트로 변환하고, 결과를 파일과 데이터베이스에 저장합니다.

---

## 📁 모듈 구성

* `settings.py`: STT 관련 설정을 관리합니다.
* `model.py`: Whisper ASR 모델 초기화 및 처리 로직을 정의합니다.
* `utils.py`: 결과 저장 및 유틸리티 함수를 제공합니다.
* `table.py`: STT 결과를 저장할 데이터베이스 테이블 정의를 포함합니다.

---

## ⚙️ 설정 (`settings.py`)

### `STTSettings`

* `whisper_model_path (str)`: Whisper 모델 경로 (기본: `openai/whisper-small`)
* `chunk_length_s (int)`: 오디오 청크 길이 (초)
* `stride_length_s (int)`: 청크 간 겹침 길이 (초)
* `device (str)`: 실행 디바이스 (`cpu` 또는 `cuda`)
* `.env` 파일로부터 환경 변수 로드 가능

### `set_stt_settings()`

* 런타임에 새로운 설정 인스턴스를 주입할 수 있습니다.

---

## 🎯 모델 (`model.py`)

### `initialize_asr()`

* Whisper ASR 파이프라인을 초기화합니다.
* 비동기 실행을 지원합니다.

### `process_audio()`

* 오디오 파일을 처리하여 텍스트로 변환합니다.
* 타임스탬프와 함께 변환된 텍스트를 반환합니다.

---

## 🛠️ 유틸리티 (`utils.py`)

### `save_results()`

* STT 결과를 파일과 데이터베이스에 저장합니다.
* 파일 저장: 타임스탬프와 텍스트를 포함
* DB 저장: `stt_result` 테이블에 저장

---

## 📊 테이블 (`table.py`)

### `STTResult`

* `audio_file_path`: 오디오 파일 경로
* `stt_text`: STT 텍스트 결과

---

## 📌 예시 코드

```python
from stt.model import initialize_asr, process_audio
from stt.utils import save_results

async def process_audio_file(file_path: str):
    # ASR 파이프라인 초기화
    asr_pipeline = await initialize_asr()
    
    # 오디오 처리
    results = await process_audio(file_path, asr_pipeline=asr_pipeline)
    
    # 결과 저장
    await save_results(results, "output.txt")
```

---

## 📎 참고

* Whisper 모델 크기:
  * `tiny`: ~75MB
  * `small`: ~500MB
  * `medium`: ~1.5GB
  * `large-v3`: ~3GB

* 성능과 속도의 균형을 위해 `small` 모델을 기본으로 사용

---

## ✅ ToDo

* 더 나은 한국어 인식을 위한 모델 최적화
* db연결

---

## 🧑‍💻 개발자

이 모듈은 음성 인식 기능을 쉽게 통합할 수 있도록 설계되었습니다. 파일 기반 저장과 데이터베이스 저장을 모두 지원하여 다양한 사용 사례에 대응할 수 있습니다. 