# src/common/config.py
SERVER_IP = '127.0.0.1'
TCP_PORT = 9997
UDP_PORT = 9998
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHUNK_SIZE = 4096
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# 자세 가이드 임계값
FACE_CENTER_THRESHOLD = 18.0   # 얼굴 중앙 정렬 허용 편차(%) - 기존 10.0에서 완화
SHOULDER_TILT_THRESHOLD = 5.0  # 어깨 수평 허용 기울기(%)

# ── 음성 분석 설정 ──
ANALYZE_SR = 16000             # 분석용 리샘플 레이트(Hz). 44100 전체는 너무 무거워 16k로 낮춰 분석
SILENCE_TOP_DB = 30            # librosa 묵음 판정 기준(dB). 피크 대비 이보다 작으면 묵음으로 간주
F0_MIN = 80                    # F0(음정) 분석 하한(Hz)
F0_MAX = 400                   # F0(음정) 분석 상한(Hz)
MIN_ANALYZE_SEC = 1.0          # 이보다 짧은 음성은 분석하지 않음(초)

# ── TTS 설정 (gTTS) ──
TTS_LANG = 'ko'               # gTTS 언어 코드
TTS_CACHE_DIR = 'data/tts_cache'  # 생성한 TTS 샘플 캐시 경로 (Git 제외 대상)