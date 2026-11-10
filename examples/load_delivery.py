#!/usr/bin/env python3
"""Read a XIM Audio stems delivery into plain Python rows. Standard library only. MIT licence.

A delivery looks like this:

  <root>/manifest.csv            one row per song (song_id, creator_id, split, daw, bpm, key, genre, sample_rate, ...)
  <root>/splits.json             train / val / test, split by creator
  <root>/data/ximaudio/<song_id>/
        data.json                the song's own record: per lane the instrument label, peak and activity
        mix.wav                  the unity-gain sum of the lanes, 32-bit float
        mix_flat_24.wav          the same sum at -0.3 dBFS peak, 24-bit
        MIDI/                    where the session had notes
        <stem group>/<NN - lane name>.wav

  from load_delivery import songs
  for s in songs("/path/to/delivery", split="train"):
      print(s["song_id"], s["bpm"], len(s["lanes"]), s["mix"])

  python3 load_delivery.py /path/to/delivery > songs.jsonl
"""
import csv
import json
import os
import sys

SKIP_DIRS = {"MIDI"}


def songs(root, split=None):
    """Yield one dict per song: the manifest row plus the paths of its mix, lanes and MIDI files."""
    with open(os.path.join(root, "manifest.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if split and row.get("split") != split:
            continue
        folder = os.path.join(root, "data", "ximaudio", row["song_id"])
        if not os.path.isdir(folder):
            continue
        lanes, midi = [], []
        for group in sorted(os.listdir(folder)):
            sub = os.path.join(folder, group)
            if not os.path.isdir(sub):
                continue
            for fn in sorted(os.listdir(sub)):
                p = os.path.join(sub, fn)
                if group in SKIP_DIRS:
                    if fn.lower().endswith((".mid", ".midi")):
                        midi.append(p)
                elif fn.lower().endswith(".wav"):
                    lanes.append({"group": group, "name": os.path.splitext(fn)[0], "path": p})
        mix = os.path.join(folder, "mix.wav")
        data = os.path.join(folder, "data.json")
        yield {**row, "folder": folder, "mix": mix if os.path.exists(mix) else None,
               "data_json": data if os.path.exists(data) else None, "lanes": lanes, "midi": midi}


def song_record(song):
    """The song's data.json as a dict (instrument label, peak and activity per lane), or None."""
    if not song.get("data_json"):
        return None
    with open(song["data_json"], encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for s in songs(sys.argv[1], split=sys.argv[2] if len(sys.argv) > 2 else None):
        print(json.dumps(s, ensure_ascii=False))
