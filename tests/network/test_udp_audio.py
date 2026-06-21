# tests/network/test_udp_audio.py
# 마이크 하드웨어 없이 음성 패킷 와이어 포맷만 검증한다.
# (audio_sender는 sd.rec로 실제 마이크를 쓰므로 CI에서 직접 호출 불가)
import socket
import numpy as np
from src.common.packet_format import PKT_PATIENT_AUDIO
from src.common.config import AUDIO_CHUNK_SIZE


def test_audio_packet_roundtrip():
    recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv.bind(("127.0.0.1", 0))
    port = recv.getsockname()[1]

    send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    audio = (np.zeros(AUDIO_CHUNK_SIZE, dtype=np.int16))
    packet = bytes([PKT_PATIENT_AUDIO]) + audio.tobytes()
    send.sendto(packet, ("127.0.0.1", port))

    recv.settimeout(2.0)
    data, _ = recv.recvfrom(65535)
    send.close()
    recv.close()

    # 첫 바이트는 타입, 나머지는 페이로드
    assert data[0] == PKT_PATIENT_AUDIO
    payload = np.frombuffer(data[1:], dtype=np.int16)
    assert len(payload) == AUDIO_CHUNK_SIZE


if __name__ == "__main__":
    test_audio_packet_roundtrip()
    print("udp_audio 테스트 통과")