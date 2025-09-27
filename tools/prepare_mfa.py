#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import shutil

def prepare(meta_csv: Path, mfa_dir: Path):
    df = pd.read_csv(meta_csv)
    mfa_dir.mkdir(parents=True, exist_ok=True)
    mfa_wavs = mfa_dir / "wavs"
    mfa_wavs.mkdir(parents=True, exist_ok=True)
    transcripts = []
    for _, r in df.iterrows():
        utt = r['utt_id']
        # prefer converted wav_16k_path if available
        wav = r.get('wav_16k_path') if 'wav_16k_path' in r and isinstance(r['wav_16k_path'], str) and r['wav_16k_path'] else r['wav_path']
        if not wav or not Path(wav).exists():
            print("Skipping missing wav for", utt)
            continue
        dst_path = mfa_wavs / f"{utt}.wav"
        shutil.copyfile(wav, dst_path)
        transcripts.append(f"{utt}\t{r['transcript']}")
    with open(mfa_dir / "transcripts.txt", "w", encoding="utf8") as fh:
        fh.write("\n".join(transcripts))
    print("Prepared MFA workspace at", mfa_dir, "with", len(transcripts), "utterances")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="../data/metadata_with_wavs.csv")
    parser.add_argument("--mfa-dir", default="../mfa")
    args = parser.parse_args()
    prepare(Path(args.meta), Path(args.mfa_dir))