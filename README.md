# BabyLM 2026 Multilingual: LLaMA + Attention Residuals

> Compact multilingual pre-training under the BabyLM 100M-word budget,  
> with Regex-Guided BBPE tokenization and the Attention Residuals mechanism.

[![Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/kkkkkkshy/babylm-llama-ar-v1-100Mtokens)
[![Tokenizer](https://img.shields.io/badge/🤗-Tokenizer-yellow)](https://huggingface.co/kkkkkkshy/babylm-tokenizer-regex-bbpe)

---

## 📋 Repository Contents

| File | Purpose |
|------|---------|
| `model.py` | LLaMA + Attention Residuals architecture |
| `pretrain.py` | Training entry point |
| `dataset.py` | Memory-mapped binary dataset loader |
| `build_multilingual_pretrain_bin.py` | Multilingual data mixing → uint16 `.bin` |
| `data_process.py` | Data preprocessing utilities |
| `convert_to_hf.py` | Convert PyTorch checkpoint → HuggingFace format |
| `eval.py`, `eval_pretrain.py` | Evaluation utilities |
| `sft.py`, `sft_data_process.py`, `dataset_sft.py` | (Optional) Supervised fine-tuning |
| `requirements.txt` | Python dependencies |

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/to1246zh-s-star/babylm-attnres-multilingual.git
cd babylm-attnres-multilingual

pip install -r requirements.txt
```

### 2. Get tokenizer & data

```bash
# Tokenizer (from our HuggingFace release)
huggingface-cli download kkkkkkshy/babylm-tokenizer-regex-bbpe \
    --local-dir ./tokenizer

# Training data: download BabyBabelLM corpora from HuggingFace
# https://huggingface.co/babylm (English/Dutch/Chinese)
# Place under: ./data/raw/{eng,nld,zho}/
```

### 3. Generate training .bin

```bash
python build_multilingual_pretrain_bin.py \
    --output ./data/merged_100m.bin \
    --budget 100000000 \
    --tokenizer-path ./tokenizer \
    --zh-path ./data/raw/zho \
    --en-path ./data/raw/eng \
    --nl-path ./data/raw/nld
```

### 4. Train

```bash
python pretrain.py
```

Expected: ~6-8 hours on RTX 3090/4090. Checkpoints saved to `out/epoch_N.pth`.

### 5. Convert & upload to HuggingFace

```bash
python convert_to_hf.py
huggingface-cli upload YOUR_HF_USERNAME/your-model-name out/hf_model
```

---

## 🧪 Customize the Language Mix

Edit `DEFAULT_RATIOS` in `build_multilingual_pretrain_bin.py`:

```python
# Default: equal trilingual mix (33/33/33)
DEFAULT_RATIOS = {"zh": 1, "en": 1, "nl": 1}

# Chinese-heavy (60/20/20)
DEFAULT_RATIOS = {"zh": 3, "en": 1, "nl": 1}
```

Then regenerate the `.bin` (Step 3) and retrain.

---

## 📦 Released Artifacts

| Resource | HuggingFace Link |
|----------|------------------|
| Main model (LLaMA+AR, 100M tokens, 33/33/33) | [`kkkkkkshy/babylm-llama-ar-v1-100Mtokens`](https://huggingface.co/kkkkkkshy/babylm-llama-ar-v1-100Mtokens) |
| Regex-BBPE tokenizer | [`kkkkkkshy/babylm-tokenizer-regex-bbpe`](https://huggingface.co/kkkkkkshy/babylm-tokenizer-regex-bbpe) |
| SentencePiece (control) | [`kkkkkkshy/babylm-tokenizer-sentencepiece`](https://huggingface.co/kkkkkkshy/babylm-tokenizer-sentencepiece) |
| tiktoken cl100k (control) | [`kkkkkkshy/babylm-tokenizer-tiktoken`](https://huggingface.co/kkkkkkshy/babylm-tokenizer-tiktoken) |
| Baby-Qwen prototype | [`kkkkkkshy/baby-qwen-78m-multilingual`](https://huggingface.co/kkkkkkshy/baby-qwen-78m-multilingual) |

---

## 🐛 Known Issues

1. **Use `trust_remote_code=True`** when loading from HF — custom architecture.
2. **AutoDL users**: `source /etc/network_turbo` for HF/GitHub access.
3. **Do not install vLLM** — it breaks PyTorch in this setup.
4. **transformers >= 4.45** required.
5. **`include` task data missing** in finetune pipeline — patch:
```bash
   sed -i 's/arc belebele bmlama include mnli sib200 truthfulqa xnli/arc belebele bmlama mnli sib200 truthfulqa xnli/' \
       scripts/finetune_model.sh
```

---

## 📝 Citation

```bibtex
@misc{babylm2026_lund,
    title  = {Compact Multilingual Pre-training under the BabyLM 100M-Word 
              Budget with Regex-Guided Tokenization and Attention Residuals},
    author = {Liu, Yihan and Zhang, Tongjing},
    year   = {2026},
    note   = {BabyLM 2026 Multilingual Track submission, Lund University}
}
```

### Built on
- **Attention Residuals** — [arxiv:2603.15031](https://arxiv.org/abs/2603.15031) (Moonshot AI / Kimi Team)
- **BabyLM Challenge 2026** — [babylm.github.io](https://babylm.github.io/)
- **BabyBabelLM** — [Jumelet et al. 2025](https://arxiv.org/abs/2510.10159)

---


---

## 🙏 Acknowledgments

This repository builds on the original [AttentionResidualLlama](https://github.com/EEhanL/AttentionResidualLlama) 
codebase by [@EEhanL](https://github.com/EEhanL). Our contributions extend the original work to the BabyLM 2026 
Multilingual Track:
- Multilingual data pipeline (`build_multilingual_pretrain_bin.py`) supporting English / Dutch / Chinese
- Integration with the Regex-Guided BBPE tokenizer (16K vocabulary)
- HuggingFace deployment wrappers in `hf_remote_code/` (`BabyLlamaKimiForCausalLM`, `BabyLlamaKimiForSequenceClassification`)
- Trilingual training corpus calibration via Byte Premium ([Arnett et al., 2024](https://aclanthology.org/2024.sigul-1.1/))

The Attention Residuals mechanism is from [chen2026attnres](https://arxiv.org/abs/2603.15031) (Moonshot AI / Kimi Team).

## 📧 Contact

- Yihan Liu — yi7510li-s@student.lu.se
- Tongjing Zhang — tongjing.zhang.1246@student.lu.se
