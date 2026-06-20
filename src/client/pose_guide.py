# src/client/pose_guide.py
# 환자 자세 유도 가이드 - 어깨 수평 및 얼굴 중앙 정렬 감지

import cv2
import numpy as np
from src.common.config import FACE_CENTER_THRESHOLD, SHOULDER_TILT_THRESHOLD

baseline_left = None
baseline_right = None

prev_gray = None
prev_points = None


def set_baseline(left: tuple, right: tuple):
    global baseline_left, baseline_right, prev_gray, prev_points
    baseline_left = left
    baseline_right = right
    prev_gray = None
    prev_points = None
    print(f"기준점 설정 완료 - 왼쪽: {left}, 오른쪽: {right}")


def auto_set_baseline(frame: np.ndarray):
    """
    어깨 기준점을 화면 하단 좌우 지점으로 자동 설정
    (의료진 클릭 좌표를 받지 못했을 때의 예비 fallback)
    """
    h, w = frame.shape[:2]
    left = (w // 4, int(h * 0.7))
    right = (w * 3 // 4, int(h * 0.7))
    set_baseline(left, right)


def check_shoulder_level(frame: np.ndarray) -> dict:
    global prev_gray, prev_points

    if baseline_left is None or baseline_right is None:
        return {"설정됨": False}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or prev_points is None:
        prev_gray = gray
        prev_points = np.array([baseline_left, baseline_right], dtype=np.float32).reshape(-1, 1, 2)
        return {"설정됨": True, "수평": True, "기울기": 0.0}

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)

    if status[0] and status[1]:
        left_pos = next_points[0][0]
        right_pos = next_points[1][0]

        height_diff = abs(left_pos[1] - right_pos[1])
        shoulder_width = abs(left_pos[0] - right_pos[0])
        tilt = round(height_diff / shoulder_width * 100, 2) if shoulder_width > 0 else 0.0

        is_level = tilt < SHOULDER_TILT_THRESHOLD

        prev_gray = gray
        prev_points = next_points

        return {
            "설정됨": True,
            "수평": is_level,
            "기울기": tilt,
            "왼쪽": tuple(left_pos),
            "오른쪽": tuple(right_pos),
        }

    return {"설정됨": True, "수평": True, "기울기": 0.0}


def check_face_center(frame: np.ndarray, nose_x: int) -> dict:
    h, w, _ = frame.shape
    center_x = w // 2
    offset = nose_x - center_x
    offset_ratio = round(abs(offset) / center_x * 100, 2)
    is_centered = offset_ratio < FACE_CENTER_THRESHOLD

    return {
        "중앙 정렬": is_centered,
        "편차": offset_ratio,
        "방향": "왼쪽" if offset < 0 else "오른쪽"
    }


def reset_baseline():
    global baseline_left, baseline_right, prev_gray, prev_points
    baseline_left = None
    baseline_right = None
    prev_gray = None
    prev_points = None
    print("기준점 초기화 완료")