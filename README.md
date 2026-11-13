# XIM Audio datasets

Licensed music production data for AI training: multitrack stems printed from the producer's own session, the DAW
sessions themselves, MIDI, presets and more. Modern music made in the DAW by working producers, from pop and hip-hop to
house and trance.

**This repository holds no audio.** It holds what an engineer needs before asking for the data:

- [`catalogue.json`](catalogue.json): the measured numbers of every set
- [`licences/`](licences): 18 open music datasets, 6 DAW project datasets and 16 MIDI datasets, each with what its licence says about commercial use and the page that says it
- [`tools/check_stems.py`](tools/check_stems.py): a checker for any stem set, ours or anyone's
- [`examples/load_delivery.py`](examples/load_delivery.py): reads a delivery into rows

The data is licensed per set, non-exclusive, on request: **[ximaudio.com/request](https://ximaudio.com/request?ref=github)**.
Sample first, then a quote.

## The sets, measured 2026-11-13

| Set | Size | Files | Hours of audio | |
|---|---:|---:|---:|---|
| **Stems** | 4,408 stems packs | 117,622 | 9,669.2 | [page](https://ximaudio.com/data/stems?ref=github) · [card](https://huggingface.co/datasets/ximaudio/stems) |
| **Paired production sets** | 3,637 songs | 137,485 | 7,790.2 | [page](https://ximaudio.com/data/paired-sets?ref=github) · [card](https://huggingface.co/datasets/ximaudio/paired-sets) |
| **Vocal and instrumental pairs** | 2,006 pairs | 4,012 | 221.5 | [page](https://ximaudio.com/data/acapella-instrumental-pairs?ref=github) · [card](https://huggingface.co/datasets/ximaudio/acapella-instrumental-pairs) |
| **DAW sessions** | 9,463 sessions |  | 691.2 | [page](https://ximaudio.com/data/daw-sessions?ref=github) · [card](https://huggingface.co/datasets/ximaudio/daw-sessions) |
| **Tutorial sessions** | 378 episodes |  | 224.2 | [page](https://ximaudio.com/data/tutorial-sessions?ref=github) · [card](https://huggingface.co/datasets/ximaudio/tutorial-sessions) |
| **MIDI files** | 1,633 packs | 289,034 |  | [page](https://ximaudio.com/data/midi?ref=github) · [card](https://huggingface.co/datasets/ximaudio/midi) |
| **Synth presets** | 445 packs | 167,387 |  | [page](https://ximaudio.com/data/presets?ref=github) · [card](https://huggingface.co/datasets/ximaudio/presets) |
| **Loops and one-shots** | 1,695 packs | 458,371 |  | [page](https://ximaudio.com/data/loops-one-shots?ref=github) · [card](https://huggingface.co/datasets/ximaudio/loops-one-shots) |
| **Instrumentals** | 3,351 tracks |  | 184.3 | [page](https://ximaudio.com/data/instrumentals?ref=github) · [card](https://huggingface.co/datasets/ximaudio/instrumentals) |
| **Stock music** | 1,927 tracks |  | 81.4 | [page](https://ximaudio.com/data/stock-music?ref=github) · [card](https://huggingface.co/datasets/ximaudio/stock-music) |
| **Marketplace data** | 54,446 rows |  |  | [page](https://ximaudio.com/data/marketplace-data?ref=github) · [card](https://huggingface.co/datasets/ximaudio/marketplace-data) |

Stems, by DAW: Logic 2,258, Ableton 1,485, Fl studio 117, Cubase 14, FL Studio 6. Median 24 lanes per song; eight songs in ten have
between 12 and 45. Each lane carries one of 53 instrument labels in the manifest.

## How the stems are made

Every creator in this set accepted terms that name AI training. Every stem is printed from the creator's own session, part by part, and the manifest says which session it came from. No separation model touched the files. Each WAV is one lane of the session, printed with its
plugins loaded, and every song ships with the unity-gain sum of its lanes, so the parts and the mix agree to the sample.

Every creator sells on WeMakeDanceMusic, a music production marketplace online since 2012, under terms that name
machine learning, keeps ownership and is paid on every licence. Nothing is scraped. Deliveries are anonymised:
pseudonymous IDs, no titles, no artist or producer names.

The rights of every set, one by one: [ximaudio.com/rights](https://ximaudio.com/rights?ref=github).

## Open datasets and what their licences allow

Before licensing anything, check what the open sets allow. We read each dataset's own page and wrote down what it says.

| File | Rows | Checked |
|---|---:|---|
| [`licences/open-music-datasets.csv`](licences/open-music-datasets.csv) | 18 | 2026-11-01 |
| [`licences/daw-project-datasets.csv`](licences/daw-project-datasets.csv) | 6 | 2026-11-02 |
| [`licences/midi-datasets.csv`](licences/midi-datasets.csv) | 16 | 2026-11-02 |

Columns: `dataset, holds, licence, commercial_use, note, source, checked_on`. `commercial_use` is `yes`, `no`,
`per track` (the licence differs by track) or `not stated`. This is our reading, not legal advice; the `source` column
is the page to read yourself. Found a mistake or a missing set? Open an issue or a pull request.

## Check a stem set

```
pip install numpy soundfile
python3 tools/check_stems.py /path/to/song            # one song folder: a mix file and its stems
python3 tools/check_stems.py /path/to/dataset --all   # every song under a folder
```

It reports sample-rate mismatches, stems shorter or longer than the mix, silent stems, clipped stems, and how far
the stems are from summing to the mix (in dB). Under about -60 dB the stems are the mix. Around -20 dB or higher, a
part is missing, a bus was processed after the stems were printed, or the stems came out of a separation model.
It works on any folder that holds a mix file and its stems as WAV, FLAC or AIFF: the MUSDB18-HQ layout
(`mixture.wav` and four stems), or stems in sub-folders by group next to a `mix.wav`.

Try it without any dataset: `python3 tools/make_test_songs.py test-songs` writes one clean song and five broken ones.

## Read a delivery

```python
from load_delivery import songs
for s in songs("/path/to/delivery", split="train"):
    print(s["song_id"], s["bpm"], s["key"], len(s["lanes"]), s["mix"])
```

One folder per song in the MoisesDB layout: one WAV per lane under its stem group, `mix.wav`, `data.json`, a `MIDI/`
folder where the session had notes, and `manifest.csv` with a creator-level train, val and test split.

## Guides

- [Open music datasets and what their licences allow](https://ximaudio.com/guides/open-music-datasets-licences?ref=github)
- [A MUSDB18 alternative for commercial use](https://ximaudio.com/guides/musdb18-alternative-commercial-use?ref=github)
- [What licensed training data means](https://ximaudio.com/guides/licensed-training-data?ref=github)
- [Stems from the session, or separated stems](https://ximaudio.com/guides/stems-vs-separated-stems?ref=github)
- [DAW project file datasets: what exists, what a session carries](https://ximaudio.com/guides/daw-project-files-dataset?ref=github)
- [MIDI datasets and what their licences allow](https://ximaudio.com/guides/midi-datasets-licences?ref=github)

## Cite

```bibtex
@misc{ximaudio_catalogue_2026,
  title = {XIM Audio catalogue: licensed music production data for AI training},
  author = {XIM Audio Inc.},
  year = {2026},
  url = {https://ximaudio.com},
  doi = {10.5281/zenodo.23883162},
  note = {Catalogue measured 2026-11-13; audio licensed on request}
}
```

The catalogue datasheet is on Zenodo: [doi.org/10.5281/zenodo.23883162](https://doi.org/10.5281/zenodo.23883162).
The dataset cards are on Hugging Face: [huggingface.co/ximaudio](https://huggingface.co/ximaudio).

## Licence of this repository

Code (`tools/`, `examples/`): MIT, see [LICENSE-CODE](LICENSE-CODE). Text, numbers and the licence tables: CC BY 4.0,
see [LICENSE](LICENSE). The datasets themselves are not in this repository and are not under these licences.

XIM Audio Inc., [ximaudio.com](https://ximaudio.com/?ref=github), mikas@ximaudio.com
