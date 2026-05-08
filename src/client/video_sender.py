# src/client/video_sender.py
# 카메라 영상 캡처 및 UDP 송신

import socket
import cv2
from src.common.config import SERVER_IP, UDP_VIDEO_PORT, VIDEO_WIDTH, VIDEO_HEIGHT

# 패킷 1개당 크기 및 분할 수 (5_21p 참고)
PACKET_SIZE = VIDEO_WIDTH * VIDEO_HEIGHT * 3 // 20
PACKET_COUNT = 20

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
cap = cv2.VideoCapture(0)

def send_video():
    print(f"영상 송신 시작 - {SERVER_IP}:{UDP_VIDEO_PORT}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        d = frame.flatten()
        s = d.tobytes()

        # 20개 패킷으로 나눠서 송신 (5_21p 참고)
        for i in range(PACKET_COUNT):
            # 첫 바이트 = 패킷 인덱스
            packet = bytes([i]) + s[i * PACKET_SIZE:(i + 1) * PACKET_SIZE]
            sock.sendto(packet, (SERVER_IP, UDP_VIDEO_PORT))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    sock.close()

if __name__ == "__main__":
    send_video()