# src/recognition/pronunciation.py
# 발음 정확도 측정: TTS 표준 샘플 vs 환자 발음을 DTW(MFCC)로 비교
#
# [개선 배경]
# 기존 구현(MFCC 13차 전체 + cosine DTW)은 변별력이 거의 없었다.
#   1) MFCC c0(=에너지/음량 성분)가 다른 계수보다 값이 압도적으로 커서
#      cosine 거리가 사실상 "음량이 비슷한가"만 보게 됨 -> 문장이 달라도 점수가 뭉침.
#   2) 화자/마이크 게인 차이가 정규화 없이 그대로 거리에 섞임.
#   3) 무엇보다 TTS(기계음)와 사람 목소리는 성대 구조가 달라, 정확히 같은 문장을
#      말해도 일정한 "기저 거리"가 항상 남는다. 이 기저 거리가 발음 오류로 인한
#      차이보다 훨씬 커서, 발음 정확도 신호가 그 안에 묻혀버린다.
#
# [개선 방식]
#   - MFCC에서 c0 제거 (음량 성분 차단)
#   - 발화별 정규화(CMVN: 평균0/분산1)로 게인 차이 흡수
#   - delta/delta2(1·2차 미분) 추가 -> 절대 음색보다 "변화 패턴"을 보게 해
#     TTS-사람 간 음색 차이에 덜 민감, 조음 패턴 차이에 더 민감하게
#   - euclidean DTW (정규화했으므로 cosine 불필요)
#   - baseline 보정: TTS-사람 간 기저 거리를 100점 기준선으로 잡고 초과분만 감점

import os
import hashlib

import numpy as np
import librosa

from src.common.config import (
    AUDIO_SAMPLE_RATE, ANALYZE_SR, SILENCE_TOP_DB,
    TTS_LANG, TTS_CACHE_DIR,
    PRON_COST_FULL, PRON_COST_ZERO, PRON_GAMMA, PRON_MFCC_N, PRON_MIN_SEC,
)


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


def _feature(y: np.ndarray, sr: int = ANALYZE_SR) -> np.ndarray:
    """MFCC(c0 제거) + delta + delta2를 쌓고 발화별 정규화(CMVN)한 특징 행렬 반환.
    반환 shape: (3*(PRON_MFCC_N-1), 프레임수)"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=PRON_MFCC_N)
    mfcc = mfcc[1:]  # c0(에너지/음량 성분) 제거 -> 발음 패턴만 남김

    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, d1, d2])

    # CMVN: 발화별 평균0/분산1 정규화 -> 화자·마이크 게인 차이 흡수
    feat = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-8)
    return feat


def _avg_cost(y_ref: np.ndarray, y_pat: np.ndarray, sr: int = ANALYZE_SR) -> float:
    """두 음성 특징을 euclidean DTW로 정렬해 step당 평균 비용 반환. 낮을수록 유사."""
    X = _feature(y_ref, sr)
    Y = _feature(y_pat, sr)
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    return float(D[-1, -1]) / len(wp)


def _cost_to_score(avg_cost: float) -> float:
    """avg_cost를 0~100 점수로 변환.
    PRON_COST_FULL 이하=100점, PRON_COST_ZERO 이상=0점, 그 사이는 감마 커브.

    TTS-사람 기저 거리 때문에 정답/오답의 거리 간격이 좁다. 그 좁은 구간을
    점수축에 펴주되, gamma>1로 중간을 가파르게 깎아 "정답=무조건 100,
    오답=80% 먹고 들어감" 현상을 막는다."""
    if avg_cost <= PRON_COST_FULL:
        return 100.0
    if avg_cost >= PRON_COST_ZERO:
        return 0.0
    x = (avg_cost - PRON_COST_FULL) / (PRON_COST_ZERO - PRON_COST_FULL)  # 0(정답)~1(오답)
    score = 100.0 * (1.0 - x) ** PRON_GAMMA
    return round(max(0.0, min(100.0, score)), 1)


def _similarity(y_ref: np.ndarray, y_pat: np.ndarray, sr: int = ANALYZE_SR) -> float:
    """두 음성의 발음 유사도를 0~100으로 반환.
    TTS 없이도 단독 테스트 가능하도록 분리한 순수 함수."""
    return _cost_to_score(_avg_cost(y_ref, y_pat, sr))


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
    y_pat, _ = librosa.effects.trim(y_pat, top_db=SILENCE_TOP_DB)
    if len(y_pat) < int(ANALYZE_SR * PRON_MIN_SEC):
        return None

    try:
        y_ref, _ = _tts_sample(target_text)
    except Exception as e:
        # TTS 생성/디코딩 실패(인터넷·ffmpeg 문제 등) -> 점수 없음
        print(f"[pronunciation] TTS 실패: {e}")
        return None

    # 레퍼런스도 동일 전처리(트림) -> 침묵 길이 차이로 인한 왜곡 방지
    y_ref, _ = librosa.effects.trim(y_ref, top_db=SILENCE_TOP_DB)

    return _similarity(y_ref, y_pat, ANALYZE_SR)


if __name__ == "__main__":
    # 자기검증:
    #   - 자기 자신과 비교하면 거리 0 -> 100점
    #   - 잡음은 멀어야 -> 낮은 점수
    # 주의: 합성 사인파/잡음만으로는 c0·TTS 격차 문제를 못 잡는다.
    #       실제 변별력 검증은 tests/recognition/test_pronunciation.py에서
    #       (TTS 샘플 vs 정상발화) > (TTS 샘플 vs 오발화) 로 확인할 것.
    sr = ANALYZE_SR
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    voiced = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    noise = (np.random.randn(int(sr * 1.2)) * 0.1).astype(np.float32)
    print("self (동일):", _similarity(voiced, voiced, sr))   # 100.0 기대
    print("noise (상이):", _similarity(voiced, noise, sr))   # 낮게 기대