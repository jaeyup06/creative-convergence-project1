# src/client/audio_sender.py
# 마이크 음성 캡처 및 UDP 송신

import threading
import numpy as np
import sounddevice as sd
from src.common.config import SERVER_IP, UDP_PORT, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE
from src.common.packet_format import PKT_PATIENT_AUDIO
from src.client.video_sender import sock


def send_audio(stop_event: threading.Event = None):
    while True:
        if stop_event and stop_event.is_set():
            break
        try:
            audio = sd.rec(AUDIO_CHUNK_SIZE, samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.int16)
            sd.wait()
            if stop_event and stop_event.is_set():
                break
            data = bytes([PKT_PATIENT_AUDIO]) + audio.tobytes()
            sock.sendto(data, (SERVER_IP, UDP_PORT))
        except OSError:
            break


if __name__ == "__main__":
    send_audio()
