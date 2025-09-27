#!/usr/bin/env python3
"""
convert_wavs.py
Reads data/metadata.csv, converts/resamples audio into data/wavs/ (16kHz mono, 16-bit PCM).
Adds a new column `wav_16k_path` and writes metadata_with_wavs.csv.
"""

import argparse
from pathlib import Path
import soundfile as sf
import librosa
import pandas as pd
from tqdm import tqdm

def convert_file(src: Path, dst: Path, sr=16000):
    """Load audio, resample to 16kHz mono, normalize, and save as 16-bit PCM WAV."""
    y, _ = librosa.load(str(src), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError("Empty audio file")

    # Normalize to peak = 0.99
    peak = max(abs(y.max()), abs(y.min()))
    if peak > 0:
        y = y / peak * 0.99

    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dst), y, sr, subtype="PCM_16")

def main(meta_csv: Path, out_dir: Path):
    df = pd.read_csv(meta_csv)

    new_paths = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src = Path(row["wav_path"])

        # Ensure speaker_id and utt_id are strings
        speaker = str(row.get("speaker_id", ""))
        utt = str(row["utt_id"])

        # Save inside speaker subfolder if available
        if speaker:
            dst = out_dir / speaker / f"{utt}.wav"
        else:
            dst = out_dir / f"{utt}.wav"

        try:
            convert_file(src, dst)
            new_paths.append(str(dst.resolve()))
        except Exception as e:
            print(f"⚠️ Failed converting {src}: {e}")
            new_paths.append("")

    # Add new column
    df["wav_16k_path"] = new_paths
    out_meta = meta_csv.parent / "metadata_with_wavs.csv"
    df.to_csv(out_meta, index=False)
    print(f"✅ Wrote updated metadata with resampled wav paths: {out_meta}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="../data/metadata.csv")
    parser.add_argument("--out-wav-dir", default="../data/wavs")
    args = parser.parse_args()
    main(Path(args.meta), Path(args.out_wav_dir))
