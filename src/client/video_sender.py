import cv2
import socket
import struct
import threading
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.config import SERVER_IP, UDP_VIDEO_PORT

# UDP 패킷 최대 크기 (65507 bytes가 UDP 이론 한계, 여유있게 설정)
MAX_PACKET_SIZE = 60000


class VideoSender:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (SERVER_IP, UDP_VIDEO_PORT)
        self.cap = None
        self.running = False

    def start(self):
        """카메라 열고 전송 시작"""
        self.cap = cv2.VideoCapture(0)  # 0 = 기본 웹캠

        if not self.cap.isOpened():
            print("[ERROR] 카메라를 열 수 없습니다.")
            return

        # 해상도 설정 (너무 크면 UDP 전송 부하 증가)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        print(f"[VideoSender] 전송 시작 → {SERVER_IP}:{UDP_VIDEO_PORT}")

        thread = threading.Thread(target=self._send_loop, daemon=True)
        thread.start()

    def _send_loop(self):
        """프레임을 읽어서 UDP로 전송하는 루프"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] 프레임을 읽을 수 없습니다.")
                break

            # JPEG으로 압축 (품질 50 = 용량/품질 균형)
            ret, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if not ret:
                continue

            data = encoded.tobytes()
            size = len(data)

            # 패킷이 MAX_PACKET_SIZE보다 크면 분할 전송
            if size <= MAX_PACKET_SIZE:
                # 단일 패킷: 헤더(total=1, index=0) + 데이터
                header = struct.pack('>HH', 1, 0)  # total_chunks, chunk_index
                self.sock.sendto(header + data, self.server_addr)
            else:
                # 분할 패킷
                chunks = [data[i:i+MAX_PACKET_SIZE] for i in range(0, size, MAX_PACKET_SIZE)]
                total = len(chunks)
                for idx, chunk in enumerate(chunks):
                    header = struct.pack('>HH', total, idx)
                    self.sock.sendto(header + chunk, self.server_addr)

            # 약 30fps 목표 (0.033초 대기)
            time.sleep(0.033)

    def stop(self):
        """전송 중지 및 자원 해제"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.sock.close()
        print("[VideoSender] 전송 중지")


# 단독 실행 테스트용
if __name__ == '__main__':
    sender = VideoSender()
    sender.start()

    print("전송 중... 종료하려면 'q' 입력.")
    try:
        while True:
            key = input()
            if key.strip().lower() == 'q':
                break
    except KeyboardInterrupt:
        pass

    sender.stop()
