# src/server/audio_analyzer.py
# UDP 음성 수신 및 재생

import socket
import pyaudio
from src.common.config import SERVER_IP, UDP_AUDIO_PORT, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, UDP_AUDIO_PORT))

# PyAudio 출력 스트림 초기화
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=AUDIO_SAMPLE_RATE,
    output=True
)

def receive_audio():
    print(f"UDP 음성 대기 중 - 포트 {UDP_AUDIO_PORT}")

    while True:
        try:
            data, addr = sock.recvfrom(AUDIO_CHUNK_SIZE * 2)
            # 수신 음성 재생
            stream.write(data)
            # TODO: voice_analyzer.py 연동 (분석 버퍼에 누적)
        except OSError:
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    sock.close()

if __name__ == "__main__":
    receive_audio()