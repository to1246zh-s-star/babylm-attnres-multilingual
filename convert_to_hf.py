import os
import torch
import json
import shutil

CHECKPOINT_PATH = "/root/autodl-tmp/AttentionResidualLlama/out/pretrain/epoch_9.pth"
OUTPUT_DIR = "/root/autodl-tmp/AttentionResidualLlama/out/hf_model_v1_100Mtokens"
TOKENIZER_SRC = "/root/autodl-tmp/babylm-2026-multilingual/models/tokenizer_regex_bbpe"

# 严格对齐 HuggingFace LlamaConfig 的参数名
HF_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 512,           # 对应原来的 dim
    "intermediate_size": 1376,    # Llama通常是 hidden_size * 8/3 向上取整到 multiple_of
    "num_hidden_layers": 8,       # 对应 n_layers
    "num_attention_heads": 8,     # 对应 n_heads
    "num_key_value_heads": 8,     # 对应 n_kv_heads
    "vocab_size": 16000,
    "max_position_embeddings": 512, # 对应 max_seq_len
    "rms_norm_eps": 1e-5,
    "initializer_range": 0.02,
    "use_cache": True,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
    "transformers_version": "4.40.0"
}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"🚀 正在转换权重...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
    
    # 这一步非常关键：
    # 如果你的原始代码中 key 的名字和 HF 标准不一致（例如层名叫做 layers 而不是 model.layers）
    # 你可能需要在这里做一个映射。但通常简单转换先尝试直接保存：
    torch.save(state_dict, os.path.join(OUTPUT_DIR, "pytorch_model.bin"))
    
    print(f"📝 正在写入对齐后的 config.json...")
    with open(os.path.join(OUTPUT_DIR, "config.json"), "w") as f:
        json.dump(HF_CONFIG, f, indent=2)
    
    print(f"📂 正在同步分词器文件...")
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src_f = os.path.join(TOKENIZER_SRC, f)
        if os.path.exists(src_f):
            shutil.copy(src_f, OUTPUT_DIR)
            
    print(f"✅ 转换完成！路径: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
