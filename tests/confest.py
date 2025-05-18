import os
from dotenv import load_dotenv

  # .env 파일 로드
load_dotenv()

  # PYTHONPATH 설정
if os.getenv("PYTHONPATH"):
      import sys
      sys.path.append(os.getenv("PYTHONPATH"))