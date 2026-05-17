# src/client/audio_sender.py
# 마이크 음성 캡처 및 UDP 송신

import socket
import numpy as np
import sounddevice as sd
from src.common.config import SERVER_IP, UDP_AUDIO_PORT, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def send_audio():
    print(f"음성 송신 시작 - {SERVER_IP}:{UDP_AUDIO_PORT}")

    while True:
        try:
            # 마이크 입력 캡처
            audio = sd.rec(AUDIO_CHUNK_SIZE, samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.int16)
            sd.wait()
            # 바이트로 변환 후 송신
            data = audio.tobytes()
            sock.sendto(data, (SERVER_IP, UDP_AUDIO_PORT))
        except OSError:
            break

    sock.close()

if __name__ == "__main__":
    send_audio()