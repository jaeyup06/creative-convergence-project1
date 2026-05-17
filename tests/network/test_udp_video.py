# tests/network/test_udp_video.py
# UDP 영상 송수신 테스트

import socket
import numpy as np
from src.common.config import SERVER_IP, UDP_VIDEO_PORT, VIDEO_WIDTH, VIDEO_HEIGHT

# 패킷 크기 및 분할 수
PACKET_SIZE = VIDEO_WIDTH * VIDEO_HEIGHT * 3 // 20
PACKET_COUNT = 20

def test_udp_video():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP 영상 송신 테스트 - {SERVER_IP}:{UDP_VIDEO_PORT}")

    # 더미 프레임 생성 (640x480 검정 화면)
    dummy_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    s = dummy_frame.flatten().tobytes()

    # 20개 패킷으로 나눠서 송신 (5_21p 참고)
    for i in range(PACKET_COUNT):
        packet = bytes([i]) + s[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
        sock.sendto(packet, (SERVER_IP, UDP_VIDEO_PORT))

    print("UDP 영상 더미 패킷 전송 완료")
    sock.close()

if __name__ == "__main__":
    test_udp_video()