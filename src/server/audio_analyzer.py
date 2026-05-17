# src/server/audio_analyzer.py
# UDP 음성 수신 및 재생

import socket
import numpy as np
import sounddevice as sd
from src.common.config import SERVER_IP, UDP_AUDIO_PORT, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE
from src.server.session_recorder import save_audio

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, UDP_AUDIO_PORT))

# 출력 스트림 초기화
stream = sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.int16)
stream.start()

def receive_audio(patient_name: str = "환자"):
    print(f"UDP 음성 대기 중 - 포트 {UDP_AUDIO_PORT}")
    buffer = b''
    audio_buffer = b''  # 세션 음성 누적

    while True:
        try:
            data, addr = sock.recvfrom(AUDIO_CHUNK_SIZE * 2)
            buffer += data
            audio_buffer += data

            # 버퍼에 충분히 쌓이면 재생
            if len(buffer) >= AUDIO_CHUNK_SIZE * 8:
                audio = np.frombuffer(buffer, dtype=np.int16)
                stream.write(audio)
                buffer = b''
            # TODO: voice_analyzer.py 연동 (분석 버퍼에 누적)
        except OSError:
            break

    # 세션 종료 시 음성 저장
    save_audio(patient_name, audio_buffer, AUDIO_SAMPLE_RATE)
    stream.stop()
    sock.close()

if __name__ == "__main__":
    receive_audio()