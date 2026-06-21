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

# ── 발음 정확도(DTW) 설정 ──
# TTS(기계음)와 사람 목소리는 성대 구조가 달라, 같은 문장을 정확히 말해도
# MFCC-DTW 거리가 0이 되지 않고 일정 수준의 "기저 거리"가 항상 남는다.
# 정답 발음과 오답 발음의 avg_cost 간격이 좁기 때문에(기계-사람 격차가 워낙 커서),
# 그 좁은 구간을 점수축에 펴주는 변환이 필요하다. 두 개의 거리 기준점을 둔다.
#   - PRON_COST_FULL : 이 거리 이하면 100점 (정답 수준)
#   - PRON_COST_ZERO : 이 거리 이상이면 0점 (완전 다른 발음/잡음)
# 그 사이는 감마 커브로 변환한다. gamma>1이면 중간 구간을 더 가파르게 떨어뜨려
# "정답이 무조건 100, 오답이 80% 먹고 들어감" 현상을 막는다.
#
# TODO: 현재 값은 '안녕하세요' 1문장 + 정답 1·오답 1 샘플 기준 임시값.
#       재활 대상에 가까운 발화자/문장으로 표본을 늘려 재측정 필요.
# 주의: 정답(good)의 실측 cost가 약 6.3, 오답(bad)이 약 6.7로 간격이 0.4뿐이라
#       이 한 쌍만으로는 정답/오답을 깔끔히 분리하기 어렵다. FULL을 정답 cost보다
#       아래로 두어 "어눌하면 정답이라도 감점"되도록 빡빡하게 잡았다. 그 대가로
#       정답조차 만점은 안 나온다. 표본이 늘면 FULL/ZERO를 다시 조정할 것.
PRON_COST_FULL = 5.8          # 이하면 100점 (정답 cost 6.3보다 낮게 둬 만점 천장 제거)
PRON_COST_ZERO = 7.4          # 이상이면 0점 (다른 발음/잡음)
PRON_GAMMA = 1.6              # 점수 커브 기울기. >1이면 중간 구간을 더 가파르게 감점
PRON_MFCC_N = 13              # MFCC 차수 (c0는 코드에서 제거)
PRON_MIN_SEC = 0.5            # 이보다 짧으면(트림 후) 비교 불가로 None 반환