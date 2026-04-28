from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hf_remote_code.configuration_babyllama_kimi import BabyLlamaKimiConfig  # noqa: E402


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        return
    shutil.copytree(src, dst)

def _strip_prefix(state: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in state.keys()):
        return state
    out: dict = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def _add_prefix(state: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in state.items()}


def _ensure_tied_embeddings(state: dict) -> dict:
    # Ensure both keys exist in the saved HF state dict.
    tok_k = "model.tok_embeddings.weight"
    out_k = "model.output.weight"
    if tok_k in state and out_k not in state:
        state[out_k] = state[tok_k]
    elif out_k in state and tok_k not in state:
        state[tok_k] = state[out_k]
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pth", type=str, required=True, help="Path to .pth state_dict")
    ap.add_argument("--out_dir", type=str, required=True, help="Output HF directory")

    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--n_kv_heads", type=int, default=None)
    ap.add_argument("--vocab_size", type=int, default=64793)
    ap.add_argument("--multiple_of", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--rms_norm_eps", type=float, default=1e-5)

    ap.add_argument("--bos_token_id", type=int, default=1)
    ap.add_argument("--eos_token_id", type=int, default=2)
    ap.add_argument("--pad_token_id", type=int, default=None)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Copy remote-code python files
    shutil.copy2(_HERE / "configuration_babyllama_kimi.py", out_dir / "configuration_babyllama_kimi.py")
    shutil.copy2(_HERE / "modeling_babyllama_kimi.py", out_dir / "modeling_babyllama_kimi.py")

    # 2) Copy tokenizer implementation + assets (ChatGLMTokenizer)
    _copy_tree(_REPO_ROOT / "chatglm_tokenizer", out_dir / "chatglm_tokenizer")
    # Also place the SentencePiece model at the HF root dir as expected by ChatGLMTokenizer
    sp_model_src = _REPO_ROOT / "chatglm_tokenizer" / "tokenizer.model"
    sp_model_dst = out_dir / "tokenizer.model"
    if sp_model_src.exists() and not sp_model_dst.exists():
        shutil.copy2(sp_model_src, sp_model_dst)
    # AutoTokenizer dynamic module loader expects "tokenization_xxx.ClassName" (one dot),
    # so provide a root-level tokenization file.
    tok_py_src = _REPO_ROOT / "chatglm_tokenizer" / "tokenization_chatglm.py"
    tok_py_dst = out_dir / "tokenization_chatglm.py"
    if tok_py_src.exists() and not tok_py_dst.exists():
        shutil.copy2(tok_py_src, tok_py_dst)

    # also copy tokenizer_config.json if present
    if (_REPO_ROOT / "chatglm_tokenizer" / "tokenizer_config.json").exists():
        # Create a root-level tokenizer_config.json so AutoTokenizer can resolve the class.
        tok_cfg = {
            "tokenizer_class": "ChatGLMTokenizer",
            "auto_map": {
                "AutoTokenizer": ["tokenization_chatglm.ChatGLMTokenizer", None]
            },
        }
        (out_dir / "tokenizer_config.json").write_text(
            json.dumps(tok_cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # 3) Build HF config with auto_map for trust_remote_code
    config = BabyLlamaKimiConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.dim,
        num_hidden_layers=args.n_layers,
        num_attention_heads=args.n_heads,
        num_key_value_heads=args.n_kv_heads if args.n_kv_heads is not None else args.n_heads,
        multiple_of=args.multiple_of,
        rms_norm_eps=args.rms_norm_eps,
        max_position_embeddings=args.max_seq_len,
        dropout=args.dropout,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
    )

    # tell Auto classes where to import from
    config.auto_map = {
        "AutoConfig": "configuration_babyllama_kimi.BabyLlamaKimiConfig",
        "AutoModelForCausalLM": "modeling_babyllama_kimi.BabyLlamaKimiForCausalLM",
        "AutoTokenizer": "chatglm_tokenizer.tokenization_chatglm.ChatGLMTokenizer",
    }
    config.tokenizer_class = "ChatGLMTokenizer"

    # 4) Low-memory export:
    # - do NOT instantiate the HF model
    # - do NOT call save_pretrained()
    # Just write:
    # - config.json
    # - pytorch_model.bin with keys prefixed by "model."
    config.save_pretrained(out_dir)

    state = torch.load(args.pth, map_location="cpu")
    state = _strip_prefix(state, "_orig_mod.")
    state = _add_prefix(state, "model.")
    state = _ensure_tied_embeddings(state)

    torch.save(state, out_dir / "pytorch_model.bin")

    # 6) If the original tokenizer_config exists, keep it (AutoTokenizer will use it)
    # (We don't force-write special tokens here because your project already has them in tokenizer.)

    # 7) Write a small metadata file for humans
    meta = {
        "source_pth": os.path.abspath(args.pth),
        "notes": "Exported with trust_remote_code=True custom model.",
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved HF model to: {out_dir}")


if __name__ == "__main__":
    main()

