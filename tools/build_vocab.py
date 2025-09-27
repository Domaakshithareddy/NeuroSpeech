#!/usr/bin/env python3
# tools/build_vocab.py
import json, re, argparse
from pathlib import Path
from collections import Counter
import pandas as pd

def normalize_text(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build(manifests, out="tools/vocab.json"):
    chars = set()
    for m in manifests:
        df = pd.read_json(m, lines=True)
        for t in df['text'].astype(str):
            t = normalize_text(t)
            chars.update(list(t))
    chars = sorted(chars)
    tokens = ["|"] + chars  # '|' as word separator
    vocab = {c: i for i,c in enumerate(tokens)}
    vocab["<unk>"] = len(vocab)
    vocab["<pad>"] = len(vocab)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, indent=2)
    print("Wrote vocab:", out, "size:", len(vocab))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifests", nargs="+", default=["experiments/splits/train_manifest.jsonl","experiments/splits/val_manifest.jsonl"])
    p.add_argument("--out", default="tools/vocab.json")
    args = p.parse_args()
    build(args.manifests, args.out)
