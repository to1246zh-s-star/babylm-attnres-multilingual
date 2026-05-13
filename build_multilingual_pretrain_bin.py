#!/usr/bin/env python3
"""
Build a multilingual BabyLM pretraining bin file using a HuggingFace tokenizer.
Modified to support HF tokenizers (e.g. Regex-Guided BBPE) instead of SentencePiece.
"""
import argparse
from pathlib import Path
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

DEFAULT_SOURCES = {
    "zh": "/root/autodl-tmp/babylm-2026-multilingual/data/zho",
    "en": "/root/autodl-tmp/babylm-2026-multilingual/data/eng",
    "nl": "/root/autodl-tmp/babylm-2026-multilingual/data/nld",
}

# Equal split for fairness (BabyLM-aligned). Chinese byte premium ≈ 0.99, almost 1.
DEFAULT_RATIOS = {
    "zh": 1,
    "en": 1,
    "nl": 1,
}


class DatasetCursor:
    def __init__(self, dataset, seed: int):
        self.dataset = dataset
        self.size = len(dataset)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.size)
        self.ptr = 0
        self.epochs = 0

    def next_text(self) -> str:
        if self.ptr >= self.size:
            self.order = self.rng.permutation(self.size)
            self.ptr = 0
            self.epochs += 1
        text = self.dataset[int(self.order[self.ptr])]["text"]
        self.ptr += 1
        return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a multilingual BabyLM pretraining bin with fixed token ratios."
    )
    parser.add_argument(
        "--output",
        default="./data/merged_multilingual_100m.bin",
        help="Output .bin path.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100_000_000,
        help="Total token budget after tokenization, including <eos>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for document order.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="/root/autodl-tmp/babylm-2026-multilingual/models/tokenizer_regex_bbpe",
        help="HuggingFace tokenizer directory path (containing tokenizer.json).",
    )
    parser.add_argument("--zh-path", default=DEFAULT_SOURCES["zh"])
    parser.add_argument("--en-path", default=DEFAULT_SOURCES["en"])
    parser.add_argument("--nl-path", default=DEFAULT_SOURCES["nl"])
    parser.add_argument(
        "--buffer-tokens",
        type=int,
        default=1_000_000,
        help="Flush token buffer to disk after this many tokens.",
    )
    return parser.parse_args()


def choose_language(token_counts, target_counts):
    remaining = {lang: target_counts[lang] - token_counts[lang] for lang in target_counts}
    candidates = [lang for lang, left in remaining.items() if left > 0]
    if not candidates:
        return max(target_counts, key=lambda lang: target_counts[lang] - token_counts[lang])
    return max(candidates, key=lambda lang: remaining[lang])


def flush_tokens(output_path: Path, token_buffer):
    if not token_buffer:
        return
    arr = np.asarray(token_buffer, dtype=np.uint16)
    with open(output_path, "ab") as f:
        f.write(arr.tobytes())
    token_buffer.clear()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()  # Reset the bin file

    # ============ Load HuggingFace tokenizer (key change!) ============
    print(f"Loading HuggingFace tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(f"  Vocab size: {len(tokenizer)}")

    # Get EOS token id (handle if not defined)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        # Fallback: use the </s> token id explicitly
        eos_id = tokenizer.convert_tokens_to_ids("</s>")
        print(f"  No eos_token_id, using </s> id: {eos_id}")
    print(f"  EOS token id: {eos_id}")

    # ============ Load datasets ============
    paths = {"zh": args.zh_path, "en": args.en_path, "nl": args.nl_path}
    datasets = {}
    for lang, path in paths.items():
        print(f"Loading {lang} dataset from {path}")
        datasets[lang] = load_from_disk(path)
        print(f"  Size: {len(datasets[lang]):,} documents")

    cursors = {lang: DatasetCursor(datasets[lang], args.seed + i) for i, lang in enumerate(paths)}

    # ============ Compute target token counts per language ============
    total_ratio = sum(DEFAULT_RATIOS.values())
    target_counts = {
        lang: int(args.budget * DEFAULT_RATIOS[lang] / total_ratio) for lang in DEFAULT_RATIOS
    }
    print(f"Target token counts per language: {target_counts}")

    # ============ Tokenize and write ============
    token_counts = {lang: 0 for lang in DEFAULT_RATIOS}
    token_buffer = []
    total_tokens = 0
    pbar = tqdm(total=args.budget, desc="Tokenizing", unit="tok")

    while total_tokens < args.budget:
        lang = choose_language(token_counts, target_counts)
        text = cursors[lang].next_text()
        if not text:
            continue

        # ============ HF tokenizer encode (key change!) ============
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        text_ids.append(eos_id)

        # Sanity check: ensure token IDs fit in uint16
        if any(tid > 65535 or tid < 0 for tid in text_ids):
            raise ValueError(f"Token id out of uint16 range encountered: max={max(text_ids)}")

        # Don't exceed budget
        remaining = args.budget - total_tokens
        if len(text_ids) > remaining:
            text_ids = text_ids[:remaining]

        token_buffer.extend(text_ids)
        token_counts[lang] += len(text_ids)
        total_tokens += len(text_ids)
        pbar.update(len(text_ids))

        if len(token_buffer) >= args.buffer_tokens:
            flush_tokens(output_path, token_buffer)

    flush_tokens(output_path, token_buffer)
    pbar.close()

    print(f"\n=== Tokenization complete ===")
    print(f"Final token counts: {token_counts}")
    print(f"Total tokens written: {total_tokens:,}")
    print(f"Output: {output_path}")
    print(f"Output size: {output_path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()