import urllib.request
import bz2
import os

MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
OUTPUT_DIR = "data/models"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "shape_predictor_68_face_landmarks.dat")
COMPRESSED_PATH = OUTPUT_PATH + ".bz2"

def download_dlib_model():
    if os.path.exists(OUTPUT_PATH):
        print("이미 모델 파일이 존재합니다:", OUTPUT_PATH)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("dlib 68 모델 다운로드 중")
    urllib.request.urlretrieve(MODEL_URL, COMPRESSED_PATH)
    print("다운로드 완료")

    print("압축 해제 중")
    with bz2.open(COMPRESSED_PATH, "rb") as f_in:
        with open(OUTPUT_PATH, "wb") as f_out:
            f_out.write(f_in.read())

    os.remove(COMPRESSED_PATH)
    print("완료:", OUTPUT_PATH)

if __name__ == "__main__":
    download_dlib_model()