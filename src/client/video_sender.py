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

# 의료진 버튼 → 다음 프레임에 처리할 요청 플래그
_request_set_shoulder = False
_request_save_baseline = False
_request_start_session = False
_request_stop_session = False
_shoulder_coords = None  # (left, right) 의료진이 클릭한 좌표. 없으면 자동 추정으로 fallback


def request_set_shoulder(left: tuple = None, right: tuple = None):
    """
    의료진이 '어깨 기준점 설정' 버튼 누르면 호출
    left, right: 의료진이 환자 화면에서 클릭한 (x, y) 좌표.
                 좌표가 없으면 다음 프레임에서 화면비 추정(auto_set_baseline)으로 fallback
    """
    global _request_set_shoulder, _shoulder_coords
    _request_set_shoulder = True
    _shoulder_coords = (left, right) if (left is not None and right is not None) else None


def request_save_baseline():
    """의료진이 '베이스라인 저장' 버튼 누르면 호출"""
    global _request_save_baseline
    _request_save_baseline = True


def request_start_session():
    """의료진의 '재활 세션 시작' 버튼 → 자세 가이드 모드 종료, 재활 측정 모드로 전환"""
    global _request_start_session
    _request_start_session = True


def request_stop_session():
    """의료진의 '세션 종료' 버튼 → 자세 가이드 모드로 복귀"""
    global _request_stop_session
    _request_stop_session = True


def apply_overlay(frame, draw_text=True):
    """
    프레임에 오버레이 적용 + 분석 결과 반환

    - 비대칭 지수(analysis["asymmetry"])는 pose_mode(자세 가이드/재활 측정) 여부와
      무관하게 매 프레임 항상 계산함. 세션을 시작하지 않아도 값이 끊기지 않음.
    - draw_text=False(GUI 모드)일 때는 얼굴/어깨 디버그 텍스트와 dlib 랜드마크
      시각화를 프레임에 굽지 않음 (환자 화면을 깨끗하게 유지). 수치는 GUI의
      자세 가이드/분석 수치 패널에서 따로 표시됨.
    - draw_text=True(독립 실행 cv2 미리보기 모드)일 때만 디버그용으로 시각화함.

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

    # 비대칭 지수는 모드와 관계없이 항상 계산 (자세 가이드 단계에서도 표시 필요)
    if landmarks is not None:
        asymmetry = analyzer.calculate_asymmetry(landmarks)
        analysis["asymmetry"] = asymmetry
        if analyzer.baseline is not None:
            analysis["asym_diff"] = round(asymmetry - analyzer.baseline, 4)

    if pose_mode:
        # 자세 유도 모드
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
        # 재활 측정 모드 - dlib 랜드마크 시각화는 독립 실행(미리보기) 모드에서만 그림
        if draw_text:
            frame, _ = analyzer.analyze(frame)
            cv2.putText(frame, "MODE: Rehabilitation Session",
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame, analysis


def send_video(frame_callback=None, stop_event: threading.Event = None,
               analysis_callback=None):
    """
    frame_callback: 오버레이 적용된 프레임 전달 (GUI 표시용)
    analysis_callback: 분석 결과 dict 전달 (자세 가이드/수치 표시용)
    stop_event: 종료 신호
    """
    global pose_mode, _request_set_shoulder, _request_save_baseline
    global _request_start_session, _request_stop_session, _shoulder_coords

    cap = cv2.VideoCapture(0)
    gui_mode = frame_callback is not None

    if not gui_mode:
        print(f"영상 송신 시작 - {SERVER_IP}:{UDP_PORT}")
        print("B: 안면 baseline 저장 | S: 어깨 기준점(자동) | R: 재활 세션 시작 | Q: 종료")

    while True:
        if stop_event and stop_event.is_set():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))

        # 의료진 버튼 요청 처리
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
        if _request_start_session:
            pose_mode = False
            _request_start_session = False
            print("[VideoSender] 세션 시작 (버튼) - 재활 측정 모드로 전환")
        if _request_stop_session:
            pose_mode = True
            _request_stop_session = False
            print("[VideoSender] 세션 종료 (버튼) - 자세 가이드 모드로 복귀")

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