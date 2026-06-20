# tests/data/test_calibration.py
# 자세 가이드 캘리브레이션(중앙 정렬 / 어깨 수평 기준점) 테스트

import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.client import pose_guide
from src.common.config import FACE_CENTER_THRESHOLD


def _blank_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_face_center_within_threshold():
    frame = _blank_frame()
    center_x = frame.shape[1] // 2
    offset_x = int(center_x * (FACE_CENTER_THRESHOLD - 1) / 100)
    result = pose_guide.check_face_center(frame, center_x + offset_x)
    assert result["중앙 정렬"] is True
    print(f"[OK] 중앙 정렬 판정 - 편차 {result['편차']}% (기준 {FACE_CENTER_THRESHOLD}%)")


def test_face_center_outside_threshold():
    frame = _blank_frame()
    center_x = frame.shape[1] // 2
    offset_x = int(center_x * (FACE_CENTER_THRESHOLD + 5) / 100)
    result = pose_guide.check_face_center(frame, center_x + offset_x)
    assert result["중앙 정렬"] is False
    print(f"[OK] 중앙 비정렬 판정 - 편차 {result['편차']}% (기준 {FACE_CENTER_THRESHOLD}%)")


def test_set_baseline_then_check_shoulder():
    pose_guide.reset_baseline()
    frame = _blank_frame()

    before = pose_guide.check_shoulder_level(frame)
    assert before["설정됨"] is False

    pose_guide.set_baseline((200, 300), (440, 300))

    after = pose_guide.check_shoulder_level(frame)
    assert after["설정됨"] is True
    assert after["기울기"] == 0.0
    print("[OK] 어깨 기준점 설정 및 초기 수평 판정 통과")

    pose_guide.reset_baseline()


def test_reset_baseline_clears_state():
    pose_guide.set_baseline((100, 100), (300, 100))
    pose_guide.reset_baseline()
    result = pose_guide.check_shoulder_level(_blank_frame())
    assert result["설정됨"] is False
    print("[OK] 기준점 초기화 확인")


if __name__ == "__main__":
    test_face_center_within_threshold()
    test_face_center_outside_threshold()
    test_set_baseline_then_check_shoulder()
    test_reset_baseline_clears_state()
    print("모든 캘리브레이션 테스트 통과")