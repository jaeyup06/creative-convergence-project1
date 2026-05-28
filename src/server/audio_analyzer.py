# src/server/audio_analyzer.py
# UDP 음성 수신 및 재생 (Queue 기반)

import queue
import threading
import numpy as np
import sounddevice as sd
from src.common.config import AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE
from src.server.session_recorder import save_audio

stream = sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.int16)
stream.start()


def receive_audio(audio_queue: queue.Queue, patient_name: str = "환자",
                  gui=None, mute_event: threading.Event = None):
    print("음성 수신 대기 중...")
    buffer = b''
    audio_buffer = b''

    while True:
        try:
            data = audio_queue.get(timeout=1.0)

            payload = data[1:]
            buffer += payload
            audio_buffer += payload

            if len(buffer) >= AUDIO_CHUNK_SIZE * 8:
                audio = np.frombuffer(buffer, dtype=np.int16)

                muted = mute_event is not None and mute_event.is_set()
                if not muted:
                    stream.write(audio)

                volume = 0
                if not muted:
                    rms = int(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
                    volume = min(int(rms / 300 * 100), 100)
                if gui:
                    gui.root.after(0, lambda v=volume: gui.update_patient_volume(v))

                buffer = b''
        except queue.Empty:
            continue
        except OSError:
            break

    save_audio(patient_name, audio_buffer, AUDIO_SAMPLE_RATE)
    stream.stop()
