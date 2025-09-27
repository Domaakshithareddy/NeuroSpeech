#!/usr/bin/env python3
"""
tools/make_manifests.py

Regenerates train/val/test JSONL manifests from CSV split files in experiments/splits/.
- Always prefers 'wav_16k_path' if present and non-empty, else falls back to 'audio_path' or 'wav_path'.
- Applies a deterministic text normalization so manifests are consistent with training.
- Writes audio_filepath, text, utt_id fields per line (one JSON object per line).
"""

import argparse
import json
import re
from pathlib import Path
import pandas as pd

# ---- Text normalization (use same rules as training later) ----
def normalize_text(s: str) -> str:
    # Lowercase, keep a-z and apostrophes and spaces only, collapse multi-space
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"[^a-z' ]+", " ", s)    # remove everything except letters, apostrophe and space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def choose_audio_path(row) -> str:
    # Prefer wav_16k_path (resampled), fallback to audio_path (split), otherwise wav_path
    for key in ("wav_16k_path", "audio_path", "wav_path"):
        if key in row and isinstance(row[key], str) and len(row[key].strip()) > 0:
            return row[key]
    # as a last resort: try building path from speaker/utt if possible (not mandatory)
    return ""

def make_manifest_from_csv(csv_path: Path, out_path: Path, normalize=True):
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"Warning: {csv_path} is empty; wrote empty manifest {out_path}")
        out_path.write_text("", encoding="utf8")
        return 0

    rows_written = 0
    with out_path.open("w", encoding="utf8") as fh:
        for _, r in df.iterrows():
            audio = choose_audio_path(r)
            if not audio:
                # skip rows missing audio
                continue
            text = r.get("transcript", "")
            if normalize:
                text = normalize_text(text)
            utt = str(r.get("utt_id", r.get("utt", "")))
            obj = {"audio_filepath": str(audio), "text": text, "utt_id": utt}
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"Wrote {out_path} ({rows_written} lines)")
    return rows_written

def main(args):
    root = Path(args.splits_dir)
    if not root.exists():
        raise SystemExit(f"Splits folder not found: {root}")

    # map CSV -> output manifest path
    mapping = {
        "train": root / "train.csv",
        "validation": root / "val.csv",
        "test": root / "test.csv"
    }
    out_prefix = root
    # create manifests
    total = 0
    for split, csvp in mapping.items():
        if not csvp.exists():
            print(f"Skipping missing CSV for split {split}: {csvp}")
            continue
        # out name: train_manifest.jsonl, val_manifest.jsonl, test_manifest.jsonl
        out_name = f"{split.replace('validation','val')}_manifest.jsonl" if split == "validation" else f"{split}_manifest.jsonl"
        outp = out_prefix / out_name
        total += make_manifest_from_csv(csvp, outp, normalize=not args.no_normalize)

    print(f"✅ Finished. Total rows across manifests: {total}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits-dir", default="experiments/splits", help="Folder containing train.csv, val.csv, test.csv")
    p.add_argument("--no-normalize", action="store_true", help="Disable text normalization (not recommended)")
    args = p.parse_args()
    main(args)
