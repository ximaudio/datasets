#!/usr/bin/env python3
"""Write six short test songs for check_stems.py: one clean, five broken on purpose.

  python3 tools/make_test_songs.py test-songs
  python3 tools/check_stems.py test-songs --all

clean     three stems that sum to the mix
missing   the mix holds a part that is not among the stems
gain      the mix was turned down 6 dB after the stems were printed
silent    one stem is digital silence
rate      one stem is at 48 kHz, the rest at 44.1 kHz
clip      one stem is clipped

Noise and a sine, four seconds each: no music in here. Needs numpy and soundfile. MIT licence.
"""
import os
import sys

import numpy as np
import soundfile as sf

SR, SECONDS = 44100, 4


def make(folder, fault=None, seed=1):
    rng = np.random.default_rng(seed)
    n = SR * SECONDS
    os.makedirs(os.path.join(folder, "drums"), exist_ok=True)
    os.makedirs(os.path.join(folder, "bass"), exist_ok=True)
    drums = 0.1 * rng.standard_normal((n, 2))
    bass = 0.2 * np.sin(np.linspace(0, 2000, n))[:, None]
    sub = 0.05 * rng.standard_normal((n, 1))
    mix = drums + np.repeat(bass, 2, axis=1) + np.repeat(sub, 2, axis=1)
    if fault == "missing":
        mix = mix + 0.05 * rng.standard_normal((n, 2))
    if fault == "gain":
        mix = mix * 0.5
    sf.write(os.path.join(folder, "drums", "01 - drums.wav"), drums, SR, subtype="PCM_24")
    sf.write(os.path.join(folder, "bass", "02 - bass.wav"), bass, SR, subtype="PCM_24")
    sf.write(os.path.join(folder, "bass", "03 - sub.wav"), sub, SR, subtype="PCM_24")
    if fault == "silent":
        sf.write(os.path.join(folder, "bass", "04 - empty.wav"), np.zeros((n, 2)), SR, subtype="PCM_24")
    if fault == "rate":
        sf.write(os.path.join(folder, "bass", "04 - other rate.wav"), np.full((48000 * SECONDS, 2), 1e-3), 48000, subtype="PCM_24")
    if fault == "clip":
        sf.write(os.path.join(folder, "bass", "04 - hot.wav"), np.clip(12 * drums, -1, 1), SR, subtype="PCM_24")
    sf.write(os.path.join(folder, "mix.wav"), mix, SR, subtype="FLOAT")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "test-songs"
    for i, fault in enumerate((None, "missing", "gain", "silent", "rate", "clip")):
        make(os.path.join(out, fault or "clean"), fault, seed=i + 1)
    print("written to", out)
