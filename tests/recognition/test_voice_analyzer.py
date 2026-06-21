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
    short = (np.zeros(int(sr * 0.3))).astype(np.int16)
    r = analyze_voice(short, sr)
    assert r["speech_rate"] is None
    assert r["volume"] is None


def test_silence_detected():
    sr = AUDIO_SAMPLE_RATE
    tone = _make_tone(dur=2.0, sr=sr)
    silence = np.zeros(int(sr * 1.0), dtype=np.int16)
    r = analyze_voice(np.concatenate([tone, silence]), sr)
    assert r["silence_sec"] >= 0.5


def test_stable_tone_high_f0_stability():
    r = analyze_voice(_make_tone(dur=2.0), AUDIO_SAMPLE_RATE)
    assert r["f0_stability"] is not None and r["f0_stability"] > 80


if __name__ == "__main__":
    test_keys_present()
    test_too_short_returns_none_metrics()
    test_silence_detected()
    test_stable_tone_high_f0_stability()
    print("voice_analyzer 테스트 통과")