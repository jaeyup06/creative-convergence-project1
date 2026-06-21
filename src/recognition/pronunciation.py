# src/recognition/pronunciation.py
# 발음 정확도 측정: TTS 표준 샘플 vs 환자 발음을 DTW(MFCC)로 비교
# - FFT(주파수 스펙트럼 비교)는 TTS-사람 목소리 간 파형 구조 차이로 수치 왜곡이 큼.
#   DTW는 시간축 패턴 흐름 기준이라 그 차이를 어느 정도 흡수 -> librosa 내장 DTW 사용.

import os
import hashlib

import numpy as np
import librosa

from src.common.config import AUDIO_SAMPLE_RATE, ANALYZE_SR, TTS_LANG, TTS_CACHE_DIR


def _to_float_mono(audio) -> np.ndarray:
    a = np.asarray(audio)
    if a.ndim > 1:
        a = a.reshape(-1)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    else:
        a = a.astype(np.float32)
    return a


def _tts_sample(text: str):
    """text의 표준 발음 TTS(mp3)를 생성/캐시하고 (y, sr=ANALYZE_SR) 반환.
    gTTS는 mp3로만 출력되므로 librosa.load(mp3) 디코딩에 ffmpeg(또는 최신 libsndfile)가 필요."""
    from gtts import gTTS  # 런타임 import: 인터넷 없을 때 모듈 전체가 죽지 않도록

    os.makedirs(TTS_CACHE_DIR, exist_ok=True)
    key = hashlib.md5(text.strip().encode("utf-8")).hexdigest()
    mp3_path = os.path.join(TTS_CACHE_DIR, f"{key}.mp3")
    if not os.path.exists(mp3_path):
        gTTS(text=text, lang=TTS_LANG).save(mp3_path)
    y, _ = librosa.load(mp3_path, sr=ANALYZE_SR)
    return y, ANALYZE_SR


def _similarity(y_ref: np.ndarray, y_pat: np.ndarray, sr: int = ANALYZE_SR) -> float:
    """두 음성의 MFCC를 DTW로 정렬해 0~100 유사도 반환.
    TTS 없이도 단독 테스트 가능하도록 분리한 순수 함수."""
    mfcc_ref = librosa.feature.mfcc(y=y_ref, sr=sr, n_mfcc=13)
    mfcc_pat = librosa.feature.mfcc(y=y_pat, sr=sr, n_mfcc=13)

    # cosine 거리 기반 DTW. 누적비용을 경로길이로 나눠 step당 평균비용(0~2)을 구함
    D, wp = librosa.sequence.dtw(X=mfcc_ref, Y=mfcc_pat, metric="cosine")
    avg_cost = float(D[-1, -1]) / len(wp)
    accuracy = max(0.0, 100.0 * (1.0 - avg_cost / 2.0))
    return round(accuracy, 1)


def score_pronunciation(patient_audio, target_text: str, sr: int = AUDIO_SAMPLE_RATE):
    """
    환자 발음과 target_text의 TTS 표준 샘플을 비교해 0~100 정확도 반환.
    분석 불가 시 None.

    주의: patient_audio가 세션 전체 버퍼면 한 문장이 아니므로 앞뒤 묵음을 잘라
          비교 신뢰도를 높인다. 정석은 문장 단위 분할(추후 과제).
    """
    if not target_text or not target_text.strip():
        return None

    y_pat = _to_float_mono(patient_audio)
    if sr != ANALYZE_SR:
        y_pat = librosa.resample(y_pat, orig_sr=sr, target_sr=ANALYZE_SR)

    # 앞뒤 묵음 제거
    y_pat, _ = librosa.effects.trim(y_pat, top_db=30)
    if len(y_pat) < ANALYZE_SR // 2:  # 0.5초 미만이면 비교 불가
        return None

    try:
        y_ref, _ = _tts_sample(target_text)
    except Exception as e:
        # TTS 생성/디코딩 실패(인터넷·ffmpeg 문제 등) -> 점수 없음
        print(f"[pronunciation] TTS 실패: {e}")
        return None

    return _similarity(y_ref, y_pat, ANALYZE_SR)


if __name__ == "__main__":
    # 자기검증: 동일 음성끼리는 높은 점수, 잡음은 낮은 점수가 나와야 함
    sr = ANALYZE_SR
    t = np.linspace(0, 1.0, sr, endpoint=False)
    a = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = np.random.randn(sr).astype(np.float32)
    print("same:", _similarity(a, a, sr))
    print("noise:", _similarity(a, noise, sr))