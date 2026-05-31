# src/client/pose_guide.py
# 환자 자세 유도 가이드 - 어깨 수평 및 얼굴 중앙 정렬 감지

import cv2
import numpy as np

# 기준점 저장 (왼쪽 어깨, 오른쪽 어깨)
baseline_left = None
baseline_right = None

# Optical Flow 추적용 이전 프레임
prev_gray = None
prev_points = None


def set_baseline(left: tuple, right: tuple):
    # 의료진이 클릭한 어깨 기준점 저장
    global baseline_left, baseline_right, prev_gray, prev_points
    baseline_left = left
    baseline_right = right
    prev_gray = None
    prev_points = None
    print(f"기준점 설정 완료 - 왼쪽: {left}, 오른쪽: {right}")


def auto_set_baseline(frame: np.ndarray):
    """
    어깨 기준점을 화면 하단 좌우 지점으로 자동 설정
    (클릭 없이 버튼 한 번으로 어깨 추적 시작)
    """
    h, w = frame.shape[:2]
    # 화면 하단(70% 높이), 좌우 1/4 지점을 어깨로 추정
    left = (w // 4, int(h * 0.7))
    right = (w * 3 // 4, int(h * 0.7))
    set_baseline(left, right)


def check_shoulder_level(frame: np.ndarray) -> dict:
    # Optical Flow로 어깨 기준점 추적 후 수평 여부 계산
    global prev_gray, prev_points

    if baseline_left is None or baseline_right is None:
        return {"설정됨": False}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray is None or prev_points is None:
        prev_gray = gray
        prev_points = np.array([baseline_left, baseline_right], dtype=np.float32).reshape(-1, 1, 2)
        return {"설정됨": True, "수평": True, "기울기": 0.0}

    # Lucas-Kanade Optical Flow로 특징점 추적
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)

    if status[0] and status[1]:
        left_pos = next_points[0][0]
        right_pos = next_points[1][0]

        # 높이 차이로 기울기 계산
        height_diff = abs(left_pos[1] - right_pos[1])
        shoulder_width = abs(left_pos[0] - right_pos[0])
        tilt = round(height_diff / shoulder_width * 100, 2) if shoulder_width > 0 else 0.0

        is_level = tilt < 5.0  # 5% 이하면 수평으로 판정

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
    # 코 중심이 화면 중앙에 있는지 확인
    # nose_x: dlib 68포인트 30번(코) x 좌표
    h, w, _ = frame.shape
    center_x = w // 2
    offset = nose_x - center_x
    offset_ratio = round(abs(offset) / center_x * 100, 2)
    is_centered = offset_ratio < 10.0  # 10% 이하면 중앙으로 판정

    return {
        "중앙 정렬": is_centered,
        "편차": offset_ratio,
        "방향": "왼쪽" if offset < 0 else "오른쪽"
    }


def reset_baseline():
    # 기준점 초기화
    global baseline_left, baseline_right, prev_gray, prev_points
    baseline_left = None
    baseline_right = None
    prev_gray = None
    prev_points = None
    print("기준점 초기화 완료")