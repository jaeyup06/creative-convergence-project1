# src/server/video_receiver.py
# UDP 영상 수신 및 출력

import socket
import numpy as np
import cv2
from src.common.config import SERVER_IP, UDP_VIDEO_PORT, VIDEO_WIDTH, VIDEO_HEIGHT

# 패킷 1개당 크기 및 분할 수 (5_20p 참고)
PACKET_SIZE = VIDEO_WIDTH * VIDEO_HEIGHT * 3 // 20
PACKET_COUNT = 20

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((SERVER_IP, UDP_VIDEO_PORT))

# 패킷 조립 버퍼 초기화 (5_20p 참고)
s = [b'\xff' * PACKET_SIZE for x in range(PACKET_COUNT)]

def receive_video():
    picture = b''

    while True:
        data, addr = sock.recvfrom(PACKET_SIZE + 1)

        # 첫 바이트 = 패킷 인덱스, 나머지 = 실제 데이터 (5_20p 참고)
        s[data[0]] = data[1:PACKET_SIZE + 1]

        # 마지막 패킷 수신 시 프레임 조립 (5_20p 참고)
        if data[0] == PACKET_COUNT - 1:
            for i in range(PACKET_COUNT):
                picture += s[i]

            # frombuffer로 프레임 복원, fromstring deprecated (5_16p 참고)
            frame = np.frombuffer(picture, dtype=np.uint8)
            frame = frame.reshape(VIDEO_HEIGHT, VIDEO_WIDTH, 3)
            cv2.imshow("환자 영상", frame)
            picture = b''

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    receive_video()