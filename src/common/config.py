# src/common/config.py
# 서버/클라이언트 공통 설정 (포트번호, IP 등)

# 로컬 테스트시 127.0.0.1, 다른 기기 연결시 서버 IP로 변경
SERVER_IP = '127.0.0.1'

TCP_PORT = 9997
UDP_AUDIO_PORT = 9998
UDP_VIDEO_PORT = 9999

AUDIO_SAMPLE_RATE = 44100
AUDIO_CHUNK_SIZE = 4096

# 영상 해상도 (클라이언트 카메라 설정과 동일하게)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480