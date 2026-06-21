# src/client/audio_sender.py
# 마이크 음성 캡처 및 UDP 송신

import threading
import numpy as np
import sounddevice as sd
from src.common.config import SERVER_IP, UDP_PORT, AUDIO_SAMPLE_RATE, AUDIO_CHUNK_SIZE
from src.common.packet_format import PKT_PATIENT_AUDIO
from src.client.video_sender import sock


def send_audio(stop_event: threading.Event = None, volume_callback=None):
    def callback(indata, frames, time, status):
        if stop_event and stop_event.is_set():
            raise sd.CallbackStop()
        
        # 음량 계산 후 GUI 화면 게이지 업데이트용 콜백 호출
        if volume_callback:
            rms = int(np.sqrt(np.mean(indata.astype(np.float32)**2)))
            volume = min(int(rms / 300 * 100), 100)
            volume_callback(volume)
            
        # 서버로 보낼 오디오 패킷 묶음
        data = bytes([PKT_PATIENT_AUDIO]) + indata.tobytes()
        try:
            sock.sendto(data, (SERVER_IP, UDP_PORT))
        except OSError:
            pass

    try:
        # InputStream을 사용하여 끊김 없이 백그라운드에서 오디오 스트리밍 전송
        with sd.InputStream(samplerate=AUDIO_SAMPLE_RATE, channels=1, 
                            dtype=np.int16, blocksize=AUDIO_CHUNK_SIZE, 
                            callback=callback):
            while not (stop_event and stop_event.is_set()):
                sd.sleep(100)
    except Exception as e:
        print(f"[AudioSender] 오류 발생: {e}")


if __name__ == "__main__":
    send_audio()