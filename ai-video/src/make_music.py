"""Lo-fi beat synthesized from scratch with numpy (no samples).

90 BPM, 12 bars = exactly 32 s, so the track loops seamlessly with the video.
Fmaj7 - Em7 - Dm7 - Cmaj7, electric-piano-like chords, bass, swung drums,
vinyl crackle and rain ambience. Includes a soft cat purr in bars 5-8.
"""
import sys
import wave

import numpy as np

SR = 44100
BPM = 90
BEAT = 60 / BPM
BAR = 4 * BEAT
LOOP = 12 * BAR  # 32.0 s
N = int(LOOP * SR)
rng = np.random.default_rng(3)

# render 3 loops, keep the middle one -> tails wrap around seamlessly
buf = np.zeros((3 * N, 2))


def midi(m):
    return 440 * 2 ** ((m - 69) / 12)


def add(sig, t0, pan=0.0):
    i = int(t0 * SR)
    j = min(i + len(sig), len(buf))
    if i < 0 or i >= len(buf):
        return
    l, r = np.sqrt(0.5 * (1 - pan)), np.sqrt(0.5 * (1 + pan))
    buf[i:j, 0] += sig[: j - i] * l
    buf[i:j, 1] += sig[: j - i] * r


def epiano(f, dur, vel=0.18):
    t = np.arange(int((dur + 1.5) * SR)) / SR
    env = np.exp(-t * 1.6) * np.minimum(t / 0.01, 1)
    env *= np.where(t < dur, 1, np.exp(-(t - dur) * 6))
    s = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t * 4)
         + 0.12 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t * 6))
    trem = 1 + 0.15 * np.sin(2 * np.pi * 4.5 * t)
    return vel * env * s * trem


def bass(f, dur, vel=0.30):
    t = np.arange(int((dur + 0.3) * SR)) / SR
    env = np.minimum(t / 0.02, 1) * np.exp(-t * 0.9) * np.where(t < dur, 1, np.exp(-(t - dur) * 12))
    return vel * env * (np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t))


def kick():
    t = np.arange(int(0.45 * SR)) / SR
    f = 50 + 70 * np.exp(-t * 28)
    return 0.55 * np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 8)


def lowpass(x, a):
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc += a * (x[i] - acc)
        y[i] = acc
    return y


def snare():
    t = np.arange(int(0.3 * SR)) / SR
    n = rng.normal(0, 1, len(t))
    n = n - lowpass(n, 0.08)
    return 0.16 * n * np.exp(-t * 16) + 0.08 * np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25)


def hat(vel=0.05):
    t = np.arange(int(0.08 * SR)) / SR
    n = rng.normal(0, 1, len(t))
    n = n - lowpass(n, 0.5)
    return vel * n * np.exp(-t * 60)


CHORDS = [  # Fmaj7, Em7, Dm7, Cmaj7 (voicings around middle C) + bass root
    ([53, 57, 60, 64], 41),
    ([52, 55, 59, 62], 40),
    ([50, 53, 57, 60], 38),
    ([48, 52, 55, 59], 36),
]
MELODY = [  # (bar-in-cycle, beat, midi, beats) — sparse, lazy top line
    (0, 1.5, 72, 1), (0, 3, 69, 1), (1, 0.5, 71, 1.5), (1, 2.5, 67, 1),
    (2, 1, 69, 1), (2, 2.5, 65, 1.5), (3, 0, 67, 2), (3, 2.5, 64, 1),
]
K, S, HT = kick(), snare(), None
SWING = 0.08 * BEAT

for rep in range(3):
    base = rep * LOOP
    for bar in range(12):
        t0 = base + bar * BAR
        notes, root = CHORDS[bar % 4]
        for k, m in enumerate(notes):  # slight strum
            add(epiano(midi(m), BAR * 0.95), t0 + k * 0.018, pan=-0.3 + 0.2 * k)
        add(bass(midi(root), BEAT * 1.6), t0)
        add(bass(midi(root + 7), BEAT * 0.8), t0 + 2.5 * BEAT)
        if bar >= 4:  # melody enters after 4 bars (intro is chords+rain only)
            for b, beat, m, ln in MELODY:
                if b == bar % 4:
                    add(epiano(midi(m), ln * BEAT, 0.10), t0 + beat * BEAT + (SWING if beat % 1 else 0), pan=0.35)
        if bar >= 2:  # drums enter on bar 3
            add(K, t0)
            add(K, t0 + 2.5 * BEAT + SWING)
            add(S, t0 + BEAT)
            add(S, t0 + 3 * BEAT)
            for e in range(8):
                add(hat(0.05 if e % 2 == 0 else 0.03), t0 + e * BEAT / 2 + (SWING if e % 2 else 0), pan=0.4)

mid = buf[N:2 * N].copy()
t = np.arange(N) / SR

# purr: ~25 Hz amplitude-modulated low noise, bars 5-8 (when the cat is on screen)
noise = rng.normal(0, 1, N)
purr = lowpass(noise, 0.02) * (0.5 + 0.5 * np.sin(2 * np.pi * 24 * t)) ** 2
pw = np.clip(np.minimum((t - 8) / 1.0, (16.5 - t) / 1.0), 0, 1)
mid += (purr * pw * 0.5)[:, None]

# tape-like warmth: low-pass the music
for c in range(2):
    mid[:, c] = lowpass(mid[:, c], 0.25)

# rain ambience (pink-ish noise), periodic by construction (same noise both ends is fine)
rain = lowpass(rng.normal(0, 1, N), 0.35) - lowpass(rng.normal(0, 1, N), 0.01)
mid[:, 0] += rain * 0.035
mid[:, 1] += np.roll(rain, 5000) * 0.035
# vinyl crackle
crack = np.zeros(N)
idx = rng.integers(0, N, 900)
crack[idx] = rng.normal(0, 1, len(idx)) * rng.uniform(0.05, 0.25, len(idx))
crack = crack - lowpass(crack, 0.3)
mid += crack[:, None] * 0.6

# smooth the loop seam (rain/crackle are not periodic): 0.25 s crossfade with the start
x = int(0.25 * SR)
fade = np.linspace(0, 1, x)[:, None]
mid[:x] = mid[:x] * fade + mid[-x:] * (1 - fade)
mid = mid[:]

mid /= np.max(np.abs(mid)) * 1.12
pcm = (mid * 32767).astype(np.int16)
with wave.open(sys.argv[1], "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("ok", LOOP)
