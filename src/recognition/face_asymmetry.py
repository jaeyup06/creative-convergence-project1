import cv2
import dlib
import numpy as np
import os

# 모델 파일 경로 (프로젝트 루트 기준)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'models',
    'shape_predictor_68_face_landmarks.dat'
)

# ── Dlib 68포인트 랜드마크 인덱스 ──────────────────────────────
# 좌우 대칭 쌍: (왼쪽 인덱스, 오른쪽 인덱스)
# 눈썹, 눈, 입을 기준으로 비대칭 측정
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
        print("[FaceAsymmetry] 모델 로드 완료")

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
        좌우 대칭 랜드마크 쌍의 거리 차이로 비대칭 지수 계산
        반환: 비대칭 지수 (0.0 ~ 1.0, 낮을수록 대칭)
        """
        # 얼굴 중심선: 코 끝(30번 포인트) 기준
        nose_x = landmarks[30][0]

        diffs = []
        for left_idx, right_idx in SYMMETRIC_PAIRS:
            left_pt = landmarks[left_idx]
            right_pt = landmarks[right_idx]

            # 각 점의 코 중심선으로부터의 거리
            left_dist = abs(left_pt[0] - nose_x)
            right_dist = abs(right_pt[0] - nose_x)

            # 거리 차이 비율
            total = left_dist + right_dist
            if total > 0:
                diff = abs(left_dist - right_dist) / total
                diffs.append(diff)

        if not diffs:
            return 0.0

        asymmetry_index = float(np.mean(diffs))
        return round(asymmetry_index, 4)

    def draw_landmarks(self, frame, landmarks):
        """
        프레임에 랜드마크 점과 비대칭 수치를 오버레이로 그리기
        """
        asymmetry = self.calculate_asymmetry(landmarks)

        for i, (x, y) in enumerate(landmarks):
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # 비대칭 지수 텍스트 출력
        text = f"Asymmetry: {asymmetry:.4f}"
        color = (0, 255, 0) if asymmetry < 0.1 else (0, 165, 255) if asymmetry < 0.2 else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

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

    print("실행 중... 종료하려면 'q' 입력.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, asymmetry = analyzer.analyze(frame)

        if asymmetry is not None:
            print(f"비대칭 지수: {asymmetry:.4f}")

        cv2.imshow('Face Asymmetry Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()