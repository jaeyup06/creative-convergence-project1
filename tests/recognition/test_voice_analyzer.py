# tests/recognition/test_voice_analyzer.py
import numpy as np
from src.recognition.voice_analyzer import analyze_voice
from src.common.config import AUDIO_SAMPLE_RATE


def _make_tone(freq=220, dur=2.0, sr=AUDIO_SAMPLE_RATE):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)


def test_keys_present():
    r = analyze_voice(_make_tone(), AUDIO_SAMPLE_RATE)
    for k in ["speech_rate", "f0_stability", "volume", "silence_sec", "duration_sec"]:
        assert k in r


def test_too_short_returns_none_metrics():
    sr = AUDIO_SAMPLE_RATE
    short = (np.zeros(int(sr * 0.3))).astype(np.int16)  # 0.3초 < MIN_ANALYZE_SEC
    r = analyze_voice(short, sr)
    assert r["speech_rate"] is None
    assert r["volume"] is None


def test_silence_detected():
    sr = AUDIO_SAMPLE_RATE
    tone = _make_tone(dur=2.0, sr=sr)
    silence = np.zeros(int(sr * 1.0), dtype=np.int16)
    r = analyze_voice(np.concatenate([tone, silence]), sr)
    # 1초 묵음을 붙였으니 0.5초 이상은 묵음으로 잡혀야 함
    assert r["silence_sec"] >= 0.5


def test_stable_tone_high_f0_stability():
    r = analyze_voice(_make_tone(dur=2.0), AUDIO_SAMPLE_RATE)
    # 순음은 음정이 일정하므로 안정성 높게 나와야 함
    assert r["f0_stability"] is not None and r["f0_stability"] > 80


if __name__ == "__main__":
    test_keys_present()
    test_too_short_returns_none_metrics()
    test_silence_detected()
    test_stable_tone_high_f0_stability()
    print("voice_analyzer 테스트 통과")