import dlib
from skimage import io
from glob import glob
import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', '..', '..', 'data', 'models', 'shape_predictor_68_face_landmarks.dat')
IMG_DIR = BASE_DIR
CROP_DIR = os.path.join(BASE_DIR, 'croped_images')

# 크롭 저장 폴더 없으면 생성
if not os.path.exists(CROP_DIR):
    os.makedirs(CROP_DIR)

print("작업 폴더:", BASE_DIR)
print("모델 경로:", MODEL_PATH)
print("결과 저장:", CROP_DIR)

# 모델 로드
face_detector = dlib.get_frontal_face_detector()
face_68_landmark = dlib.shape_predictor(r'C:\Users\User\Documents\GitHub\creative-convergence-project1\data\models\shape_predictor_68_face_landmarks.dat')

i = 100
for file in glob(IMG_DIR + "/*.jpg"):
    image = io.imread(file)
    print("이미지:", file)
    print("크기:", image.shape)

    win = dlib.image_window()
    win.set_image(image)

    faces = face_detector(image, 1)
    print("인식한 얼굴 수:", len(faces))

    for face in faces:
        i += 1
        print(face)
        print(f'왼쪽: {face.left()}, 위: {face.top()}, 오른쪽: {face.right()}, 아래: {face.bottom()}')

        win.add_overlay(face)

        face_landmark = face_68_landmark(image, face)
        win.add_overlay(face_landmark)

        # 크롭 저장 (필요 시 주석 해제)
        # crop = image[face.top():face.bottom(), face.left():face.right()]
        # io.imsave(os.path.join(CROP_DIR, "croped" + str(i) + ".jpg"), crop)

dlib.hit_enter_to_continue()