#!/usr/bin/env python3
"""
prepare_metadata.py
Parses Kaldi-style files (wav.scp, text, utt2spk, spk2gender, spk2age)
and writes a single data/metadata.csv
"""

import argparse
from pathlib import Path
import pandas as pd

def load_kaldi_map(path: Path):
    d = {}
    if not path.exists():
        return d
    with open(path, "r", encoding="utf8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln: 
                continue
            parts = ln.split(maxsplit=1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d

def process(base_dir: Path, out_csv: Path):
    rows = []
    base_dir = base_dir.expanduser().resolve()
    for split in ["train", "test"]:
        split_dir = base_dir / split
        if not split_dir.exists():
            print(f"Skipping missing split dir: {split_dir}")
            continue

        wav_scp = load_kaldi_map(split_dir / "wav.scp")
        texts = load_kaldi_map(split_dir / "text")
        utt2spk = load_kaldi_map(split_dir / "utt2spk")
        spk2gender = load_kaldi_map(split_dir / "spk2gender")
        spk2age = load_kaldi_map(split_dir / "spk2age")

        for utt, wav_path in wav_scp.items():
            transcript = texts.get(utt, "")
            spk = utt2spk.get(utt, "")
            gender = spk2gender.get(spk, "")
            age = spk2age.get(spk, "")
            # If wav_path is relative in the dataset, make it absolute relative to dataset root
            wav_abspath = (split_dir.parent / wav_path).resolve() if not Path(wav_path).is_absolute() else Path(wav_path).resolve()
            rows.append({
                "utt_id": utt,
                "wav_path": str(wav_abspath),
                "transcript": transcript,
                "speaker_id": spk,
                "gender": gender,
                "age": age,
                "split": split
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Wrote metadata {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default="../data/speechocean762", help="Path to speechocean762 folder")
    p.add_argument("--out", default="../data/metadata.csv", help="Output CSV path")
    args = p.parse_args()
    process(Path(args.dataset_dir), Path(args.out))
