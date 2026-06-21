# src/server/audio_analyzer.py
# UDP 음성 수신 · 재생 · 세션 버퍼 관리
# - 실시간: 수신 음성 재생 + 음량(RMS) 미터를 GUI로 전달
# - 세션 단위: 누적 버퍼를 start_session()으로 비우고 snapshot_session()으로 회수
#   (실제 분석/저장은 server.py의 세션 종료 핸들러가 담당)

import queue
import threading
import numpy as np
import sounddevice as sd
from src.common.config import AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE

# 재생 스트림은 지연 초기화: 출력장치 없는 환경/테스트에서 import만으로 죽지 않도록
_stream = None


def _get_stream():
    global _stream
    if _stream is None:
        _stream = sd.OutputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype=np.int16)
        _stream.start()
    return _stream


# 현재 세션 동안 누적되는 환자 음성 (분석/저장용)
_buffer_lock = threading.Lock()
_session_audio = bytearray()


def start_session():
    """세션 시작 시 호출 — 누적 버퍼 초기화."""
    with _buffer_lock:
        _session_audio.clear()


def snapshot_session() -> bytes:
    """현재까지 누적된 세션 음성을 복사해 반환 (분석/저장 입력으로 사용)."""
    with _buffer_lock:
        return bytes(_session_audio)


def receive_audio(audio_queue: queue.Queue, gui=None, mute_event: threading.Event = None):
    print("음성 수신 대기 중...")
    buffer = b''

    while True:
        try:
            data = audio_queue.get(timeout=1.0)
            payload = data[1:]
            buffer += payload

            # 세션 버퍼에는 매 패킷 전부 누적 (분석 정확도 위해 손실 없이)
            with _buffer_lock:
                _session_audio.extend(payload)

            if len(buffer) >= AUDIO_CHUNK_SIZE * 8:
                audio = np.frombuffer(buffer, dtype=np.int16)

                muted = mute_event is not None and mute_event.is_set()
                if not muted:
                    try:
                        _get_stream().write(audio)
                    except Exception:
                        pass  # 출력장치 문제로 재생 실패해도 수신/분석은 계속

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