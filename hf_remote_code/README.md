## HF remote-code export (custom Transformer)

This folder contains Hugging Face Transformers "remote code" wrappers for the custom model defined in `model.py`.

You can export a `.pth` checkpoint into a standard Hugging Face directory with:

```bash
python hf_remote_code/convert_pth_to_hf.py \
  --pth out/pretrain/epoch_0.pth \
  --out_dir hf_ckpt \
  --dim 512 --n_layers 8 --n_heads 8 --n_kv_heads 8 \
  --vocab_size 64793 --multiple_of 32 --max_seq_len 512 --dropout 0.0
```

Then load it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("hf_ckpt", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("hf_ckpt", trust_remote_code=True)
```

