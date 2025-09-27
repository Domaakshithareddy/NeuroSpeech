#!/usr/bin/env python3
"""
split_dataset.py
Split dataset into train/val/test ensuring speaker separation.
Writes CSVs into experiments/splits/
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def split(meta_csv: Path, out_dir: Path, val_size=0.1, test_size=0.1, random_state=42):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_csv)

    # Choose which audio column to use
    if "wav_16k_path" in df.columns and df["wav_16k_path"].notnull().any():
        df = df[df["wav_16k_path"].str.len() > 0].copy()
        df["audio_path"] = df["wav_16k_path"]
    else:
        df["audio_path"] = df["wav_path"]

    # Group by speaker to avoid overlap between splits
    groups = df["speaker_id"].fillna("unknown")

    # Step 1: split off test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss1.split(df, groups=groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Step 2: split validation from remaining train
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_size / (1.0 - test_size),
        random_state=random_state,
    )
    tr_idx, val_idx = next(gss2.split(train_df, groups=train_df["speaker_id"].fillna("unknown")))
    final_train = train_df.iloc[tr_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    # Save splits
    final_train.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)

    print("✅ Wrote splits to", out_dir)
    print("   train/val/test rows:", len(final_train), len(val_df), len(test_df))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="../data/metadata_with_wavs.csv")
    parser.add_argument("--out", default="../experiments/splits")
    args = parser.parse_args()
    split(Path(args.meta), Path(args.out))