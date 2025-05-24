import asyncio
from app.stt.model import initialize_asr, process_audio
from app.stt.settings import stt_settings
from app.stt.utils import save_results
import os
import torch

async def main():
    # GPU 사용 가능 여부 확인
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # ASR 파이프라인 초기화
    asr_pipeline = await initialize_asr(device=device)
    
    # 오디오 파일 경로
    audio_file = "tests/test_data/redbull.wav"
    
    if not os.path.exists(audio_file):
        print(f"Error: {audio_file} not found!")
        return
    
    # STT 처리
    diarization, result = await process_audio(audio_file, asr_pipeline=asr_pipeline)
    
    # 결과 저장 (파일 + DB)
    await save_results((diarization, result), "results/redbull_stt.txt")
    
    # 결과 출력
    print("\nSTT 결과:")
    if isinstance(result, dict) and 'text' in result:
        print(f"전체 텍스트: {result['text']}")
    else:
        print("결과를 찾을 수 없습니다.")

if __name__ == "__main__":
    asyncio.run(main())
