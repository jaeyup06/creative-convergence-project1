# src/client/video_sender.py
# 카메라 영상 캡처 및 UDP 송신
# face_asymmetry, pose_guide 오버레이 포함

import socket
import threading
import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.common.config import SERVER_IP, UDP_PORT, VIDEO_WIDTH, VIDEO_HEIGHT
from src.common.packet_format import PKT_PATIENT_VIDEO, VIDEO_PACKET_COUNT, VIDEO_PACKET_SIZE
from src.recognition.face_asymmetry import FaceAsymmetryAnalyzer
from src.client.pose_guide import check_face_center, check_shoulder_level

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 0))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

analyzer = FaceAsymmetryAnalyzer()
pose_mode = True


def apply_overlay(frame):
    global pose_mode

    landmarks = analyzer.get_landmarks(frame)
    nose_x = int(landmarks[30][0]) if landmarks is not None else None

    if pose_mode:
        if nose_x is not None:
            face_result = check_face_center(frame, nose_x)
            is_centered = face_result["중앙 정렬"]
            offset = face_result["편차"]
            color = (0, 255, 0) if is_centered else (0, 0, 255)
            cv2.putText(frame, f"Face center: {'OK' if is_centered else f'off {offset:.1f}%'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        shoulder_result = check_shoulder_level(frame)
        if shoulder_result.get("설정됨"):
            is_level = shoulder_result["수평"]
            tilt = shoulder_result["기울기"]
            color = (0, 255, 0) if is_level else (0, 0, 255)
            cv2.putText(frame, f"Shoulder: {'OK' if is_level else f'tilt {tilt:.1f}%'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "Shoulder: set_baseline() needed",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.putText(frame, "MODE: Pose Guide | B: face baseline | R: start session",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        frame, asymmetry = analyzer.analyze(frame)
        cv2.putText(frame, "MODE: Rehabilitation Session",
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def send_video(frame_callback=None, stop_event: threading.Event = None):
    global pose_mode

    cap = cv2.VideoCapture(0)
    gui_mode = frame_callback is not None

    if not gui_mode:
        print(f"영상 송신 시작 - {SERVER_IP}:{UDP_PORT}")
        print("B: 안면 baseline 저장 | R: 재활 세션 시작 | Q: 종료")

    while True:
        if stop_event and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        frame = apply_overlay(frame)

        if gui_mode:
            frame_callback(frame.copy())
        else:
            cv2.imshow("client preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                analyzer.calibrate(frame)
            elif key == ord('r'):
                pose_mode = False
                print("[VideoSender] Session started")

        d = frame.flatten().tobytes()
        for i in range(VIDEO_PACKET_COUNT):
            packet = bytes([PKT_PATIENT_VIDEO, i]) + d[i * VIDEO_PACKET_SIZE:(i + 1) * VIDEO_PACKET_SIZE]
            sock.sendto(packet, (SERVER_IP, UDP_PORT))

    cap.release()
    if not gui_mode:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    send_video()
