#!/usr/bin/env python3
"""
Fine-tune Wav2Vec2 on child speech manifests.
Supports CPU, CUDA (NVIDIA), and MPS (Apple Silicon GPU).

This version includes HybridCTCTrainer which computes the CTC loss on CPU
while keeping the forward/backbone on the selected device (MPS/CUDA/CPU).
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
import evaluate

# ---- Device detection ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("✅ Using device:", device)

# print transformers/trainingargs info (helpful debug)
try:
    import transformers, inspect, sys
    print("Transformers version:", transformers.__version__)
    ta = TrainingArguments
    print("TrainingArguments from:", ta.__module__)
    print("TrainingArguments.__init__ signature snippet:", str(inspect.signature(ta.__init__))[:200])
except Exception:
    pass

# ---- Text normalization ----
def normalize_text(t: str) -> str:
    t = "" if t is None else str(t)
    t = t.lower()
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def prepare_dataset(dataset, processor):
    def map_fn(batch):
        # the datasets Audio column provides {"array": np.array, "sampling_rate": int}
        audio = batch["audio_filepath"]["array"]
        inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate,
                           return_tensors="np", padding=False)
        batch["input_values"] = inputs.input_values[0]
        # labels
        with processor.as_target_processor():
            labels = processor.tokenizer(normalize_text(batch["text"])).input_ids
        batch["labels"] = labels
        return batch
    return dataset.map(map_fn, remove_columns=dataset.column_names)

def data_collator_factory(processor):
    from dataclasses import dataclass
    from typing import List, Dict, Any
    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        def __call__(self, features: List[Dict[str,Any]]) -> Dict[str,Any]:
            input_values = [f["input_values"] for f in features]
            labels = [f["labels"] for f in features]
            batch = self.processor.feature_extractor.pad({"input_values": input_values},
                                                         return_tensors="pt")
            max_len = max(len(l) for l in labels)
            padded_labels = [l + [self.processor.tokenizer.pad_token_id] *
                             (max_len - len(l)) for l in labels]
            batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
            return batch
    return DataCollatorCTCWithPadding(processor=processor)

# ---- Hybrid Trainer: compute CTC loss on CPU ----
# ---- Hybrid Trainer: compute CTC loss on CPU ----
class HybridCTCTrainer(Trainer):
    """
    Override compute_loss to compute CTC loss on CPU while keeping model forward on MPS/CUDA.
    """
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, return_dict=True)
        logits = outputs.logits  # (B, T, V)

        if labels is None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # log-probs
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # lengths
        batch_size, T, _ = log_probs.size()
        input_lengths = torch.full((batch_size,), T, dtype=torch.long)

        pad_id = getattr(model.config, "pad_token_id", 0)
        target_lengths = (labels != pad_id).sum(dim=1).to(dtype=torch.long)

        # flatten labels
        labels_flat = [labels[i][labels[i] != pad_id] for i in range(batch_size)]
        labels_flat = [l for l in labels_flat if l.numel() > 0]
        labels_concat = torch.cat(labels_flat) if labels_flat else torch.tensor([], dtype=torch.long)

        # move to CPU for loss
        log_probs_cpu = log_probs.permute(1, 0, 2).to("cpu")   # (T, B, V)
        labels_concat = labels_concat.to("cpu")
        input_lengths = input_lengths.to("cpu")
        target_lengths = target_lengths.to("cpu")

        blank_id = getattr(model.config, "blank_token_id", pad_id)

        # ctc_loss on CPU but keep graph connected
        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs_cpu,
            labels_concat,
            input_lengths,
            target_lengths,
            blank=blank_id,
            zero_infinity=True,
        )

        return (ctc_loss, outputs) if return_outputs else ctc_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="experiments/splits/train_manifest.jsonl")
    parser.add_argument("--val", default="experiments/splits/val_manifest.jsonl")
    parser.add_argument("--test", default="experiments/splits/test_manifest.jsonl")
    parser.add_argument("--vocab", default="tools/vocab.json")
    parser.add_argument("--model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--output_dir", default="experiments/wav2vec2_child")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Processor & tokenizer ----
    tokenizer = Wav2Vec2CTCTokenizer(args.vocab,
                                     unk_token="<unk>",
                                     pad_token="<pad>",
                                     word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=False
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # ---- Model ----
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer.get_vocab()),
        ignore_mismatched_sizes=True
    ).to(device)

    # freeze feature extractor for stability on small data
    try:
        model.freeze_feature_extractor()
    except Exception:
        pass

    # ---- Dataset ----
    data_files = {}
    if Path(args.train).exists(): data_files["train"] = args.train
    if Path(args.val).exists(): data_files["validation"] = args.val
    if Path(args.test).exists(): data_files["test"] = args.test

    ds = load_dataset("json", data_files=data_files)

    for split in list(ds.keys()):
        ds[split] = ds[split].cast_column("audio_filepath", Audio(sampling_rate=16000))
        ds[split] = prepare_dataset(ds[split], processor)

    data_collator = data_collator_factory(processor)

    # ---- Metrics ----
    wer_metric = evaluate.load("wer")
    def compute_metrics(pred):
        # pred.predictions is a numpy array
        pred_logits = torch.from_numpy(pred.predictions).to(device)
        pred_ids = torch.argmax(pred_logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids.cpu().numpy())
        label_ids = pred.label_ids
        label_ids[label_ids == processor.tokenizer.pad_token_id] = -100
        label_str = processor.tokenizer.batch_decode(label_ids, group_tokens=False)
        pred_norm = [normalize_text(s) for s in pred_str]
        label_norm = [normalize_text(s) for s in label_str]
        return {"wer": wer_metric.compute(predictions=pred_norm, references=label_norm)}

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        eval_strategy="epoch",        # compatible with your transformers version
        num_train_epochs=args.epochs,
        save_steps=500,
        save_total_limit=2,
        learning_rate=args.lr,
        warmup_steps=500,
        logging_steps=50,
        gradient_checkpointing=True,
        fp16=False,   # fp16 not supported on MPS; safe default
        seed=args.seed,
        push_to_hub=False,
    )

    # ---- Use HybridCTCTrainer ----
    trainer = HybridCTCTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=ds.get("train"),
        eval_dataset=ds.get("validation"),
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("✅ Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
