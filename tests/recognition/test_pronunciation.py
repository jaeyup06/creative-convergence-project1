# tests/recognition/test_pronunciation.py
# 네트워크/ffmpeg 의존을 피하기 위해 TTS 없는 순수 DTW 함수(_similarity)와
# 입력 가드(score_pronunciation의 빈 텍스트 처리)만 검증한다.
import numpy as np
from src.recognition.pronunciation import _similarity, score_pronunciation
from src.common.config import ANALYZE_SR


def _sine(freq=220, dur=1.0, sr=ANALYZE_SR):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_identical_audio_high_score():
    a = _sine()
    assert _similarity(a, a, ANALYZE_SR) > 90


def test_noise_low_score():
    a = _sine()
    noise = np.random.randn(ANALYZE_SR).astype(np.float32)
    assert _similarity(a, noise, ANALYZE_SR) < 50


def test_empty_text_returns_none():
    a = _sine()
    assert score_pronunciation(a, "", ANALYZE_SR) is None
    assert score_pronunciation(a, "   ", ANALYZE_SR) is None


if __name__ == "__main__":
    test_identical_audio_high_score()
    test_noise_low_score()
    test_empty_text_returns_none()
    print("pronunciation 테스트 통과")