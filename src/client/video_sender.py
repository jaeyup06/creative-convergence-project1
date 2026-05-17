# src/client/video_sender.py
# 카메라에서 프레임을 캡처하여 UDP로 서버에 전송
# face_asymmetry, pose_guide 오버레이 포함

import cv2
import socket
import struct
import threading
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from common.config import SERVER_IP, UDP_VIDEO_PORT
from recognition.face_asymmetry import FaceAsymmetryAnalyzer
from pose_guide import PoseGuide

# UDP 패킷 최대 크기 (65507 bytes가 UDP 이론 한계, 여유있게 설정)
MAX_PACKET_SIZE = 60000


class VideoSender:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.server_addr = (SERVER_IP, UDP_VIDEO_PORT)
        self.cap = None
        self.running = False

        # 분석 모듈 초기화
        self.asymmetry_analyzer = FaceAsymmetryAnalyzer()
        self.pose_guide = PoseGuide()

        # 세션 상태
        # True: 자세 유도 단계 / False: 재활 측정 단계
        self.pose_mode = True

    def start(self):
        """카메라 열고 전송 시작"""
        self.cap = cv2.VideoCapture(0)  # 0 = 기본 웹캠

        if not self.cap.isOpened():
            print("[ERROR] Camera not available.")
            return

        # 해상도 설정 (너무 크면 UDP 전송 부하 증가)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.running = True
        print(f"[VideoSender] Stream started → {SERVER_IP}:{UDP_VIDEO_PORT}")
        print("Press 'c' to calibrate shoulders, 'b' to save baseline, 's' to start session, 'q' to quit")

        thread = threading.Thread(target=self._send_loop, daemon=True)
        thread.start()

    def _process_frame(self, frame):
        """
        프레임에 오버레이 적용
        - pose_mode=True : 자세 유도 가이드 표시
        - pose_mode=False: 안면 비대칭 측정 표시
        반환: (annotated_frame, asymmetry_index or None)
        """
        asymmetry = None

        if self.pose_mode:
            # 자세 유도 단계
            frame, is_centered, is_level = self.pose_guide.analyze(frame)
            cv2.putText(frame, "MODE: Pose Guide | C: Shoulder cal | B: Face baseline | S: Start",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            # 재활 측정 단계: 안면 비대칭 오버레이
            frame, asymmetry = self.asymmetry_analyzer.analyze(frame)
            cv2.putText(frame, "MODE: Rehabilitation Session",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame, asymmetry

    def _send_frame(self, frame):
        """프레임을 JPEG 압축 후 UDP 전송"""
        ret, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            return

        data = encoded.tobytes()
        size = len(data)

        if size <= MAX_PACKET_SIZE:
            header = struct.pack('>HH', 1, 0)
            self.sock.sendto(header + data, self.server_addr)
        else:
            chunks = [data[i:i+MAX_PACKET_SIZE] for i in range(0, size, MAX_PACKET_SIZE)]
            total = len(chunks)
            for idx, chunk in enumerate(chunks):
                header = struct.pack('>HH', total, idx)
                self.sock.sendto(header + chunk, self.server_addr)

    def _send_loop(self):
        """프레임을 읽고 오버레이 적용 후 UDP로 전송하는 루프"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Cannot read frame.")
                break

            # 오버레이 적용
            frame, asymmetry = self._process_frame(frame)

            # 로컬 미리보기
            cv2.imshow('Client Preview', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.stop()
                break
            elif key == ord('c'):
                # 어깨 캘리브레이션
                self.pose_guide.calibrate_shoulders(frame)
            elif key == ord('b'):
                # 안면 비대칭 기준점 저장
                baseline = self.asymmetry_analyzer.calibrate(frame)
                if baseline is not None:
                    print(f"[VideoSender] Baseline saved: {baseline:.4f}")
                else:
                    print("[VideoSender] Baseline failed - face not detected")
            elif key == ord('s'):
                # 자세 유도 → 재활 세션 전환
                self.pose_mode = False
                print("[VideoSender] Session started - Asymmetry mode ON")

            # 서버로 전송
            self._send_frame(frame)

            # 약 30fps 목표
            time.sleep(0.033)

        cv2.destroyAllWindows()

    def stop(self):
        """전송 중지 및 자원 해제"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.sock.close()
        print("[VideoSender] Stream stopped")


# 단독 실행 테스트용
if __name__ == '__main__':
    sender = VideoSender()
    sender.start()

    try:
        while sender.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    sender.stop()