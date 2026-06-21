# src/recognition/voice_analyzer.py
# 환자 발화 음성에서 발화 속도 / 음정 안정성(F0) / 음량 / 묵음 구간 추출
# - CNN/DNN 등 AI 미사용. librosa 신호 처리만 사용.
# - 세션 종료 시 누적 버퍼를 한 번에 분석하는 배치 방식.

import numpy as np
import librosa

from src.common.config import (
    AUDIO_SAMPLE_RATE, ANALYZE_SR, SILENCE_TOP_DB,
    F0_MIN, F0_MAX, MIN_ANALYZE_SEC,
)


def _to_float_mono(audio) -> np.ndarray:
    """int16/float, 다채널 입력을 float32 mono로 정규화."""
    a = np.asarray(audio)
    if a.ndim > 1:
        a = a.reshape(-1)
    if a.dtype == np.int16:
        a = a.astype(np.float32) / 32768.0
    else:
        a = a.astype(np.float32)
    return a


def analyze_voice(audio, sr: int = AUDIO_SAMPLE_RATE) -> dict:
    """
    누적 음성을 분석해 수치 dict 반환.

    audio : np.int16 또는 float ndarray (mono 또는 (-1,1))
    sr    : 입력 샘플레이트

    반환 키:
      speech_rate   발화 속도 (음절 근사 개수 / 발화시간[초])
      f0_stability  음정 안정성 (0~100, 높을수록 안정)
      volume        평균 음량 (0~100)
      silence_sec   묵음 총합 (초)
      duration_sec  전체 길이 (초)
    값이 측정 불가하면 None.
    """
    result = {
        "speech_rate": None,
        "f0_stability": None,
        "volume": None,
        "silence_sec": None,
        "duration_sec": None,
    }

    y = _to_float_mono(audio)
    if y.size == 0:
        return result

    # 분석은 16k로 통일 (44.1k 전체는 pyin이 너무 느림)
    if sr != ANALYZE_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=ANALYZE_SR)
        sr = ANALYZE_SR

    total_sec = len(y) / sr
    result["duration_sec"] = round(total_sec, 2)
    if total_sec < MIN_ANALYZE_SEC:
        return result  # 너무 짧으면 분석 신뢰도 없음

    # 음량: RMS 평균을 0~100 스케일로
    rms = librosa.feature.rms(y=y)[0]
    result["volume"] = round(float(np.mean(rms)) * 100.0, 1)

    # 묵음 구간: 비묵음(voiced) 구간을 빼서 묵음 총합 산출
    intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB)
    voiced_sec = float(sum((e - s) for s, e in intervals)) / sr
    result["silence_sec"] = round(max(total_sec - voiced_sec, 0.0), 2)

    # 발화 속도: onset(음절 시작점) 개수 / 실제 발화시간
    #   -> 정밀 음절 분할이 아닌 근사치. 실측 후 보정 필요.
    if voiced_sec > 0:
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        result["speech_rate"] = round(len(onsets) / voiced_sec, 2)
    else:
        result["speech_rate"] = 0.0

    # 음정 안정성: 유효 F0의 표준편차가 작을수록 안정 -> 100점에 가깝게
    try:
        f0, _, _ = librosa.pyin(y, fmin=F0_MIN, fmax=F0_MAX, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        if f0_valid.size > 1:
            std = float(np.std(f0_valid))
            # 표준편차 50Hz 이상이면 0점, 0이면 100점 (선형, 휴리스틱)
            result["f0_stability"] = round(max(0.0, 100.0 - std / 50.0 * 100.0), 1)
    except Exception:
        # pyin 실패 시 음정 안정성만 비움 (나머지 수치는 유지)
        pass

    return result


if __name__ == "__main__":
    # 간단 자기검증: 220Hz 사인 + 묵음
    sr = AUDIO_SAMPLE_RATE
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    tone = (np.sin(2 * np.pi * 220 * t) * 8000).astype(np.int16)
    silence = np.zeros(int(sr * 1.0), dtype=np.int16)
    sample = np.concatenate([tone, silence])
    print(analyze_voice(sample, sr))