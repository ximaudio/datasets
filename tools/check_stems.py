#!/usr/bin/env python3
"""Check a multitrack stem set before you train on it.

  python3 check_stems.py SONG_DIR            one song: a folder with a mix file and its stems (any depth)
  python3 check_stems.py ROOT --all          every folder under ROOT that holds a mix file
  python3 check_stems.py SONG_DIR --json     the same report as JSON

What it checks, per song:
  1. every file opens, and every sample is a number (a float file can hold NaN or Inf)
  2. every file has the same sample rate
  3. every stem is as long as the mix (within --slack seconds, each file timed at its own rate)
  4. silent stems (peak under --silence dBFS)
  5. clipped stems (runs of 3 or more samples at full scale)
  6. the stems sum to the mix: residual = rms(mix - sum of stems) / rms(mix), in dB.
     Under about -60 dB the stems are the mix. Around -20 dB or higher something is missing, a bus was
     processed after the stems were printed, or the stems came out of a separation model.
     The sum is only compared when checks 1 and 2 pass. Otherwise the residual is left empty and
     "sum_skipped" says why: samples at different rates do not line up, and one NaN makes every total NaN.

The mix file is mix.wav or mixture.wav unless you pass --mix NAME. Other files whose name starts with "mix" or
"master" are left out of the sum. Works on any folder that holds a mix file and its stems as WAV, FLAC or AIFF: the
MUSDB18-HQ layout (mixture.wav and four stems), or stems in sub-folders by group next to a mix.wav.

Needs: numpy, soundfile.  Exit code 0 when every song passes, 1 when any check fails.  MIT licence.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import soundfile as sf

AUDIO = (".wav", ".flac", ".aif", ".aiff")
MIX_NAMES = ("mix.wav", "mixture.wav", "mix.flac", "mixture.flac")
BLOCK = 1 << 16
FULL_SCALE = 0.9999


def db(x):
    return -math.inf if x <= 0 else 20 * math.log10(x)


def find_mix(folder, name=None):
    for fn in sorted(os.listdir(folder)):
        if (name and fn == name) or (not name and fn.lower() in MIX_NAMES):
            return os.path.join(folder, fn)
    return None


def stems_of(folder, mix):
    out = []
    for root, _, files in os.walk(folder):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            low = fn.lower()
            if p == mix or not low.endswith(AUDIO) or fn.startswith("."):
                continue
            if root == folder and (low.startswith("mix") or low.startswith("master")):
                continue
            out.append(p)
    return sorted(out)


def stereo(block):
    """(frames, channels) -> (frames, 2): mono is copied to both sides, extra channels are dropped."""
    if block.shape[1] == 1:
        return np.repeat(block, 2, axis=1)
    return block[:, :2]


def clip_runs(x):
    """Count of runs of 3 or more consecutive full-scale samples in one block (per channel, summed)."""
    hot = np.abs(x) >= FULL_SCALE
    if not hot.any():
        return 0
    runs = 0
    for ch in range(hot.shape[1]):
        h = hot[:, ch].astype(np.int8)
        edges = np.diff(np.concatenate(([0], h, [0])))
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        runs += int(np.sum((ends - starts) >= 3))
    return runs


def check_song(folder, mix_name=None, silence_db=-80.0, slack_s=0.05, fail_db=-40.0):
    mix_path = find_mix(folder, mix_name)
    if not mix_path:
        return {"song": folder, "ok": False, "problems": ["no mix file found"], "stems": []}
    paths = stems_of(folder, mix_path)
    if not paths:
        return {"song": folder, "ok": False, "problems": ["no stems found"], "stems": []}
    problems, files, unread = [], [], []
    try:
        mix = sf.SoundFile(mix_path)
    except Exception as e:
        return {"song": folder, "ok": False, "problems": [f"cannot read the mix {os.path.basename(mix_path)}: {e}"], "stems": []}
    for p in list(paths):
        try:
            files.append(sf.SoundFile(p))
        except Exception as e:
            unread.append(os.path.relpath(p, folder))
            problems.append(f"cannot read {os.path.relpath(p, folder)}: {e}")
            paths.remove(p)
    if not files:
        mix.close()
        return {"song": folder, "ok": False, "problems": problems, "stems": []}
    rates = {mix.samplerate} | {f.samplerate for f in files}
    if len(rates) > 1:
        problems.append(f"sample rates differ: {sorted(rates)}")
    mix_s = mix.frames / mix.samplerate
    info = [{"file": os.path.relpath(p, folder), "sample_rate": f.samplerate, "channels": f.channels, "frames": f.frames,
             "duration_s": round(f.frames / f.samplerate, 3), "peak": 0.0, "clip_runs": 0, "non_finite": 0}
            for p, f in zip(paths, files)]
    # Each file is timed at its own rate: 4 seconds at 48 kHz and 4 seconds at 44.1 kHz are the same length.
    for row in info:
        diff = row["frames"] / row["sample_rate"] - mix_s
        if abs(diff) > slack_s:
            problems.append(f"length differs from the mix by {diff:+.2f} s: {row['file']}")
    # Walk the song in blocks so a long song with many stems never sits in memory at once.
    e_mix = e_res = 0.0
    dot = e_sum = 0.0
    mix_bad = 0
    tails = [None] * len(files)
    total = max([mix.frames] + [f.frames for f in files])
    for _ in range(0, total, BLOCK):
        m = mix.read(BLOCK, dtype="float64", always_2d=True)
        bad = ~np.isfinite(m)
        if bad.any():
            mix_bad += int(bad.sum())
            m = np.where(bad, 0.0, m)
        m = stereo(m)
        acc = np.zeros((BLOCK, 2))
        n = len(m)
        for i, (row, f) in enumerate(zip(info, files)):
            b = f.read(BLOCK, dtype="float64", always_2d=True)
            if not len(b):
                continue
            bad = ~np.isfinite(b)
            if bad.any():
                row["non_finite"] += int(bad.sum())
                b = np.where(bad, 0.0, b)
            row["peak"] = max(row["peak"], float(np.max(np.abs(b))))
            # The last two frames of the block before ride along, so a clipped run cut by a block edge is still seen.
            row["clip_runs"] += clip_runs(b if tails[i] is None else np.concatenate((tails[i], b)))
            tails[i] = b[-2:]
            b = stereo(b)
            acc[:len(b)] += b
            n = max(n, len(b))
        mm = np.zeros((n, 2))
        mm[:len(m)] = m
        s = acc[:n]
        e_mix += float(np.sum(mm * mm))
        e_res += float(np.sum((mm - s) ** 2))
        dot += float(np.sum(mm * s))
        e_sum += float(np.sum(s * s))
    for f in files + [mix]:
        f.close()
    silent = [r["file"] for r in info if db(r["peak"]) < silence_db]
    clipped = [r["file"] for r in info if r["clip_runs"]]
    for r in info:
        r["peak_dbfs"] = None if r["peak"] <= 0 else round(db(r["peak"]), 2)
        del r["peak"]
    if silent:
        problems.append(f"{len(silent)} silent stem(s): {', '.join(silent[:5])}{' ...' if len(silent) > 5 else ''}")
    if clipped:
        problems.append(f"{len(clipped)} clipped stem(s): {', '.join(clipped[:5])}{' ...' if len(clipped) > 5 else ''}")
    broken = [r["file"] for r in info if r["non_finite"]]
    if mix_bad:
        problems.append(f"{mix_bad} NaN or Inf sample(s) in the mix")
    if broken:
        problems.append(f"NaN or Inf samples in {len(broken)} stem(s): {', '.join(broken[:5])}{' ...' if len(broken) > 5 else ''}")
    # The sum is compared sample by sample, so it only means something when every file is whole, at one rate, and every
    # sample is a number. Otherwise the residual stays empty: an undefined result must never read as a perfect sum.
    skipped = ("NaN or Inf samples" if mix_bad or broken else "sample rates differ" if len(rates) > 1
               else "a stem could not be read" if unread else None)
    residual = gain = None
    if skipped:
        problems.append(f"sum not compared: {skipped}")
    elif e_mix <= 0:
        problems.append("the mix is silent")
    else:
        residual = db(math.sqrt(e_res / e_mix)) if e_res > 0 else -math.inf
        # The gain that would best fit the sum to the mix: far from 0 dB means the mix was turned up or down after the sum.
        gain = None if e_sum <= 0 or dot <= 0 else round(db(dot / e_sum), 2)
    if residual is not None and residual > fail_db:
        problems.append(f"stems do not sum to the mix: residual {residual:.1f} dB" + (f" (best-fit gain {gain:+.2f} dB)" if gain is not None else ""))
    return {"song": folder, "mix": os.path.basename(mix_path), "n_stems": len(info), "sample_rate": mix.samplerate,
            "duration_s": round(mix_s, 2), "mix_non_finite": mix_bad,
            "sum_residual_db": None if residual is None else (-200.0 if residual == -math.inf else round(residual, 1)),
            "best_fit_gain_db": gain, "sum_skipped": skipped, "silent": silent, "clipped": clipped, "non_finite": broken,
            "unreadable": unread, "ok": not problems, "problems": problems, "stems": info}


def main():
    ap = argparse.ArgumentParser(description="Check a multitrack stem set: bad samples, rates, lengths, silence, clipping, sum to mix.")
    ap.add_argument("path")
    ap.add_argument("--all", action="store_true", help="check every folder under PATH that holds a mix file")
    ap.add_argument("--mix", help="file name of the mix (default: mix.wav or mixture.wav)")
    ap.add_argument("--silence", type=float, default=-80.0, help="a stem with a peak under this is silent (dBFS, default -80)")
    ap.add_argument("--slack", type=float, default=0.05, help="allowed length difference in seconds (default 0.05)")
    ap.add_argument("--fail", type=float, default=-40.0, help="a sum residual above this fails (dB, default -40)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.all:
        folders = sorted(r for r, _, _ in os.walk(a.path) if find_mix(r, a.mix))
    else:
        folders = [a.path]
    if not folders:
        sys.exit("no folder with a mix file under " + a.path)
    reports = [check_song(f, a.mix, a.silence, a.slack, a.fail) for f in folders]
    if a.json:
        json.dump(reports, sys.stdout, indent=1)
        print()
    else:
        for r in reports:
            head = "ok  " if r["ok"] else "FAIL"
            res = r.get("sum_residual_db")
            print(f"{head} {r['song']}  stems {r.get('n_stems', 0)}  residual {'not compared' if res is None else f'{res} dB'}")
            for p in r["problems"]:
                print("     - " + p)
        bad = sum(1 for r in reports if not r["ok"])
        print(f"{len(reports)} song(s), {bad} with problems")
    sys.exit(1 if any(not r["ok"] for r in reports) else 0)


if __name__ == "__main__":
    main()
