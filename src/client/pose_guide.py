# src/client/pose_guide.py
# 얼굴 중앙 정렬 및 어깨 수평 유도 오버레이 모듈
# - 얼굴 중앙 정렬: Dlib으로 얼굴 위치 감지
# - 어깨 수평: Optical Flow (Lucas-Kanade)로 어깨 특징점 추적

import cv2
import dlib
import numpy as np
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'models',
    'shape_predictor_68_face_landmarks.dat'
)

# Optical Flow 파라미터
LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# 얼굴 가이드 원 크기 (프레임 높이 대비 비율)
FACE_GUIDE_RADIUS_RATIO = 0.25

# 어깨 수평 허용 오차 (픽셀)
SHOULDER_LEVEL_THRESHOLD = 15


class PoseGuide:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Dlib 모델 파일이 없습니다: {MODEL_PATH}\n"
                "python scripts/download_models.py 를 먼저 실행하세요."
            )

        self.detector = dlib.get_frontal_face_detector()

        # Optical Flow 상태
        self.prev_gray = None
        self.shoulder_points = None   # 초기 어깨 특징점
        self.is_calibrated = False    # 어깨 캘리브레이션 완료 여부

    # ── 얼굴 중앙 정렬 ───────────────────────────────────────────

    def _check_face_center(self, frame):
        """
        얼굴이 화면 중앙에 있는지 확인
        반환: (is_centered, face_center) 또는 (False, None)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return False, None

        face = faces[0]
        face_cx = (face.left() + face.right()) // 2
        face_cy = (face.top() + face.bottom()) // 2

        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # 중앙으로부터 허용 범위 (프레임의 15%)
        tolerance_x = int(w * 0.15)
        tolerance_y = int(h * 0.15)

        is_centered = (
            abs(face_cx - center_x) < tolerance_x and
            abs(face_cy - center_y) < tolerance_y
        )

        return is_centered, (face_cx, face_cy)

    def _draw_face_guide(self, frame, is_centered, face_center):
        """얼굴 가이드 원 오버레이"""
        h, w = frame.shape[:2]
        center = (w // 2, h // 2)
        radius = int(h * FACE_GUIDE_RADIUS_RATIO)

        color = (0, 255, 0) if is_centered else (0, 165, 255)
        cv2.circle(frame, center, radius, color, 2)

        if face_center:
            cv2.circle(frame, face_center, 5, color, -1)

        # 안내 텍스트
        if face_center is None:
            msg = "Face not detected"
        elif is_centered:
            msg = "Face OK"
        else:
            msg = "Move face to center"

        cv2.putText(frame, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    # ── 어깨 수평 (Optical Flow) ─────────────────────────────────

    def calibrate_shoulders(self, frame):
        """
        세션 시작 전 어깨 특징점 초기화 (캘리브레이션)
        환자가 가만히 있을 때 호출
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # 어깨 위치 추정: 화면 하단 1/3, 좌우 1/4 지점
        left_shoulder = np.array([[w // 4, int(h * 0.7)]], dtype=np.float32)
        right_shoulder = np.array([[w * 3 // 4, int(h * 0.7)]], dtype=np.float32)

        # 해당 위치 주변에서 추적하기 좋은 특징점 찾기
        left_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=3,
            qualityLevel=0.01,
            minDistance=10,
            mask=self._make_roi_mask(gray, left_shoulder[0], radius=40)
        )
        right_pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=3,
            qualityLevel=0.01,
            minDistance=10,
            mask=self._make_roi_mask(gray, right_shoulder[0], radius=40)
        )

        # 특징점 못 찾으면 추정 위치 그대로 사용
        left_pts = left_pts if left_pts is not None else left_shoulder.reshape(-1, 1, 2)
        right_pts = right_pts if right_pts is not None else right_shoulder.reshape(-1, 1, 2)

        self.shoulder_points = np.vstack([left_pts, right_pts])
        self.prev_gray = gray
        self.is_calibrated = True
        print("[PoseGuide] Shoulder calibration complete")

    def _make_roi_mask(self, gray, center, radius):
        """특징점 탐색 영역 마스크 생성"""
        mask = np.zeros_like(gray)
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        return mask

    def _track_shoulders(self, frame):
        """
        Optical Flow로 어깨 특징점 추적
        반환: (is_level, left_y, right_y)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.shoulder_points is None:
            return False, None, None

        # Lucas-Kanade Optical Flow
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.shoulder_points, None, **LK_PARAMS
        )

        # 추적 성공한 점만 필터링
        good_new = new_pts[status.flatten() == 1]
        good_old = self.shoulder_points[status.flatten() == 1]

        if len(good_new) < 2:
            return False, None, None

        # x 좌표 기준으로 좌/우 어깨 분류
        h, w = frame.shape[:2]
        center_x = w // 2

        good_new_2d = good_new.reshape(-1, 2)
        left_pts = good_new_2d[good_new_2d[:, 0] < center_x]
        right_pts = good_new_2d[good_new_2d[:, 0] >= center_x]

        left_y = float(np.mean(left_pts[:, 1])) if len(left_pts) > 0 else None
        right_y = float(np.mean(right_pts[:, 1])) if len(right_pts) > 0 else None

        # 다음 프레임을 위해 업데이트
        self.shoulder_points = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray

        if left_y is None or right_y is None:
            return False, left_y, right_y

        is_level = abs(left_y - right_y) < SHOULDER_LEVEL_THRESHOLD
        return is_level, left_y, right_y

    def _draw_shoulder_guide(self, frame, is_level, left_y, right_y):
        """어깨 수평 가이드 오버레이"""
        h, w = frame.shape[:2]

        if not self.is_calibrated:
            cv2.putText(frame, "Press C to calibrate shoulders", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            return frame

        if left_y is None or right_y is None:
            cv2.putText(frame, "Shoulder not detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

        color = (0, 255, 0) if is_level else (0, 0, 255)

        # 어깨 추정 위치에 점 표시
        cv2.circle(frame, (w // 4, int(left_y)), 8, color, -1)
        cv2.circle(frame, (w * 3 // 4, int(right_y)), 8, color, -1)

        # 어깨 연결선
        cv2.line(frame, (w // 4, int(left_y)), (w * 3 // 4, int(right_y)), color, 2)

        # 수평 기준선 (점선 효과)
        avg_y = int((left_y + right_y) / 2)
        for x in range(0, w, 20):
            cv2.line(frame, (x, avg_y), (min(x + 10, w), avg_y), (200, 200, 200), 1)

        msg = "Shoulder level OK" if is_level else "Level your shoulders"
        cv2.putText(frame, msg, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    # ── 통합 분석 ────────────────────────────────────────────────

    def analyze(self, frame):
        """
        프레임 입력 → 얼굴 중앙 + 어깨 수평 동시 분석
        반환: (annotated_frame, is_face_centered, is_shoulder_level)
        """
        is_centered, face_center = self._check_face_center(frame)
        frame = self._draw_face_guide(frame, is_centered, face_center)

        is_level, left_y, right_y = self._track_shoulders(frame)
        frame = self._draw_shoulder_guide(frame, is_level, left_y, right_y)

        # 자세 종합 판정
        is_ready = is_centered and is_level
        status_msg = "Ready to start" if is_ready else "Adjust your posture"
        status_color = (0, 255, 0) if is_ready else (0, 0, 255)
        cv2.putText(frame, status_msg, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return frame, is_centered, is_level


# 단독 실행 테스트용
if __name__ == '__main__':
    guide = PoseGuide()
    cap = cv2.VideoCapture(0)

    calibrated = False
    print("실행 중... 'c'를 누르면 어깨 캘리브레이션, 'q'를 누르면 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not calibrated:
            cv2.putText(frame, "Press C to calibrate", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        frame, is_centered, is_level = guide.analyze(frame)
        cv2.imshow('Pose Guide Test', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            guide.calibrate_shoulders(frame)
            calibrated = True

    cap.release()
    cv2.destroyAllWindows()