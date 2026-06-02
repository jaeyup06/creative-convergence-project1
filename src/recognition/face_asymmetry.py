# src/recognition/face_asymmetry.py
# Dlib 68포인트 랜드마크 기반 안면 비대칭 측정 모듈

import cv2
import dlib
import numpy as np
import os

# 모델 파일 경로 (프로젝트 루트 기준)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'models',
    'shape_predictor_68_face_landmarks.dat'
)

# ── 비대칭 판정 임계치 ──────────────────────────────
# 비대칭 지수가 이 값 이상이면 단계별로 경고 표시 (임의 기준, 실측 후 조정 필요)
THRESHOLD_NORMAL = 0.10   # 이 미만: 정상(초록)
THRESHOLD_WARNING = 0.20  # 이 미만: 경계(주황), 이상: 위험(빨강)

# ── Dlib 68포인트 랜드마크 인덱스 ──────────────────────────────
# 좌우 대칭 쌍: (왼쪽 인덱스, 오른쪽 인덱스)
SYMMETRIC_PAIRS = [
    # 눈썹
    (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),
    # 눈
    (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),
    # 입
    (48, 54), (49, 53), (50, 52),
]


class FaceAsymmetryAnalyzer:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Dlib 모델 파일이 없습니다: {MODEL_PATH}\n"
                "python scripts/download_models.py 를 먼저 실행하세요."
            )

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_PATH)

        # 캘리브레이션 기준점 (무표정 상태의 비대칭 지수)
        self.baseline = None

        print("[FaceAsymmetry] Model loaded")

    def calibrate(self, frame):
        """
        무표정 상태를 기준점(Baseline)으로 저장
        반환: 기준 비대칭 지수 또는 None (얼굴 미감지)
        """
        landmarks = self.get_landmarks(frame)
        if landmarks is None:
            print("[FaceAsymmetry] Calibration failed - No face detected")
            return None

        self.baseline = self.calculate_asymmetry(landmarks)
        print(f"[FaceAsymmetry] Baseline saved: {self.baseline:.4f}")
        return self.baseline

    def get_landmarks(self, frame):
        """
        프레임에서 68개 랜드마크 좌표 추출
        반환: numpy array (68, 2) 또는 None (얼굴 미감지)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None  # 얼굴 감지 실패

        # 첫 번째 얼굴만 사용
        shape = self.predictor(gray, faces[0])
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        return landmarks

    def calculate_asymmetry(self, landmarks):
        """
        좌우 대칭 랜드마크 쌍의 X축 거리 차이 + Y축 높이 차이로 비대칭 지수 계산
        - X축: 중심선으로부터 좌우 거리 차이 (얼굴이 한쪽으로 쏠림)
        - Y축: 좌우 대칭점의 높이 차이 (한쪽 입꼬리/눈썹이 처짐 → 마비 특징)
        반환: 비대칭 지수 (0.0 ~ 1.0, 낮을수록 대칭)
        """
        # 얼굴 중심선: 코 끝(30번 포인트) 기준
        nose_x = landmarks[30][0]

        # 얼굴 크기 기준 (정규화용): 양 눈 바깥 끝(36-45) 거리
        face_width = abs(landmarks[45][0] - landmarks[36][0])
        if face_width == 0:
            face_width = 1

        diffs = []
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            left_pt = landmarks[left_idx]
            right_pt = landmarks[right_idx]

            # X축 비대칭: 중심선으로부터의 거리 차이 비율
            left_dist = abs(left_pt[0] - nose_x)
            right_dist = abs(right_pt[0] - nose_x)
            total = left_dist + right_dist
            x_diff = abs(left_dist - right_dist) / total if total > 0 else 0.0

            # Y축 비대칭: 좌우 대칭점의 높이 차이를 얼굴 폭으로 정규화
            y_diff = abs(left_pt[1] - right_pt[1]) / face_width

            # X축과 Y축 평균
            diffs.append((x_diff + y_diff) / 2)

        if not diffs:
            return 0.0

        asymmetry_index = float(np.mean(diffs))
        return round(asymmetry_index, 4)

    def _level_color(self, asymmetry):
        """비대칭 지수에 따른 단계별 색상 (BGR)"""
        if asymmetry < THRESHOLD_NORMAL:
            return (0, 255, 0)      # 초록: 정상
        elif asymmetry < THRESHOLD_WARNING:
            return (0, 165, 255)    # 주황: 경계
        else:
            return (0, 0, 255)      # 빨강: 위험

    def draw_landmarks(self, frame, landmarks):
        """
        프레임에 랜드마크 점과 비대칭 수치를 오버레이로 그리기
        baseline이 있으면 기준 대비 변화량도 함께 표시
        """
        asymmetry = self.calculate_asymmetry(landmarks)

        for i, (x, y) in enumerate(landmarks):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 비대칭 지수 텍스트 (임계치 기준 색상)
        color = self._level_color(asymmetry)
        cv2.putText(frame, f"Asymmetry: {asymmetry:.4f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 기준점 대비 변화량 표시
        if self.baseline is not None:
            delta = asymmetry - self.baseline
            delta_color = (0, 255, 0) if delta <= 0 else (0, 0, 255)
            delta_sign = "+" if delta > 0 else ""
            cv2.putText(frame, f"Delta: {delta_sign}{delta:.4f} (Baseline: {self.baseline:.4f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, delta_color, 2)
        else:
            cv2.putText(frame, "No baseline - Press B to calibrate",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        return frame, asymmetry

    def analyze(self, frame):
        """
        프레임 입력 → 랜드마크 추출 → 비대칭 지수 반환
        반환: (annotated_frame, asymmetry_index) 또는 (frame, None)
        """
        landmarks = self.get_landmarks(frame)

        if landmarks is None:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame, None

        frame, asymmetry = self.draw_landmarks(frame, landmarks)
        return frame, asymmetry


# 단독 실행 테스트용
if __name__ == '__main__':
    analyzer = FaceAsymmetryAnalyzer()
    cap = cv2.VideoCapture(0)

    print("실행 중... 'b' 기준점 저장, 'q' 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, asymmetry = analyzer.analyze(frame)

        if asymmetry is not None:
            print(f"비대칭 지수: {asymmetry:.4f}")

        cv2.imshow('Face Asymmetry Test', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            analyzer.calibrate(frame)

    cap.release()
    cv2.destroyAllWindows()