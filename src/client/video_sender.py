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
from src.client.pose_guide import check_face_center, check_shoulder_level, auto_set_baseline, set_baseline

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', 0))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)

analyzer = FaceAsymmetryAnalyzer()
pose_mode = True

_request_set_shoulder = False
_request_save_baseline = False
_shoulder_coords = None


def request_set_shoulder(left: tuple = None, right: tuple = None):
    global _request_set_shoulder, _shoulder_coords
    _request_set_shoulder = True
    _shoulder_coords = (left, right) if (left is not None and right is not None) else None


def request_save_baseline():
    global _request_save_baseline
    _request_save_baseline = True


def apply_overlay(frame, draw_text=True):
    """
    프레임에 오버레이 적용 + 분석 결과 반환
    draw_text: True면 디버그용 텍스트를 프레임에 직접 그림.
               GUI 모드에서는 자세 가이드 패널에 별도로 수치를 표시하므로 False로 둠
               (환자 화면에 중복된 글자가 떠 있던 문제 수정)
    반환: (frame, analysis)
    """
    global pose_mode

    analysis = {
        "face_ok": None, "face_offset": None,
        "shoulder_ok": None, "shoulder_tilt": None,
        "asymmetry": None, "asym_diff": None,
    }

    landmarks = analyzer.get_landmarks(frame)
    nose_x = int(landmarks[30][0]) if landmarks is not None else None

    if pose_mode:
        if nose_x is not None:
            face_result = check_face_center(frame, nose_x)
            is_centered = face_result["중앙 정렬"]
            offset = face_result["편차"]
            analysis["face_ok"] = is_centered
            analysis["face_offset"] = offset
            if draw_text:
                color = (0, 255, 0) if is_centered else (0, 0, 255)
                cv2.putText(frame, f"Face center: {'OK' if is_centered else f'off {offset:.1f}%'}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif draw_text:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        shoulder_result = check_shoulder_level(frame)
        if shoulder_result.get("설정됨"):
            is_level = shoulder_result["수평"]
            tilt = shoulder_result["기울기"]
            analysis["shoulder_ok"] = is_level
            analysis["shoulder_tilt"] = tilt
            if draw_text:
                color = (0, 255, 0) if is_level else (0, 0, 255)
                cv2.putText(frame, f"Shoulder: {'OK' if is_level else f'tilt {tilt:.1f}%'}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif draw_text:
            cv2.putText(frame, "Shoulder: set_baseline() needed",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        if draw_text:
            cv2.putText(frame, "MODE: Pose Guide | B: face baseline | R: start session",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        if landmarks is not None:
            asymmetry = analyzer.calculate_asymmetry(landmarks)
            analysis["asymmetry"] = asymmetry
            if analyzer.baseline is not None:
                analysis["asym_diff"] = round(asymmetry - analyzer.baseline, 4)
        frame, _ = analyzer.analyze(frame)
        if draw_text:
            cv2.putText(frame, "MODE: Rehabilitation Session",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, analysis


def send_video(frame_callback=None, stop_event: threading.Event = None,
               analysis_callback=None):
    global pose_mode, _request_set_shoulder, _request_save_baseline, _shoulder_coords

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

        if _request_set_shoulder:
            if _shoulder_coords:
                set_baseline(_shoulder_coords[0], _shoulder_coords[1])
            else:
                auto_set_baseline(frame)
            _request_set_shoulder = False
            _shoulder_coords = None
        if _request_save_baseline:
            analyzer.calibrate(frame)
            _request_save_baseline = False

        frame, analysis = apply_overlay(frame, draw_text=not gui_mode)

        if gui_mode:
            frame_callback(frame.copy())
            if analysis_callback:
                analysis_callback(analysis)
        else:
            cv2.imshow("client preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                analyzer.calibrate(frame)
            elif key == ord('s'):
                auto_set_baseline(frame)
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