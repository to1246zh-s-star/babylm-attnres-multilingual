# Multilingual Pretrain Workflow TODO

本清单只针对当前目标：**先把 BabyLM multilingual pretrain 工作流跑通**。  
不包含后续扩展（如新增更多语言、SFT、仓库内 eval 改造）。

## 1) 需要做的代码修改（最小范围）

- [ ] 确认 `pretrain.py` 的训练数据路径为：
  - `./data/merged_multilingual_zh4_en3_nl3_100m.bin`
- [ ] 确认 `pretrain.py` 的 `vocab_size=64793`（与当前 tokenizer 保持一致）
- [ ] 为了先验证可跑通，临时调小训练规模参数（建议先做 smoke run）：
  - [ ] `max_epoch` 先设为较小值（如 `1`）
  - [ ] `batch_size` 按显存调小，避免 OOM
  - [ ] `save_interval` 调小一些，确保短时间内能看到 checkpoint 落盘

## 2) 需要执行的任务（操作流程）

- [ ] 第一步：单卡跑通
  - 命令：`python pretrain.py`
  - 验证项：
    - [ ] dataloader 正常读取 bin
    - [ ] 训练正常前进（loss 有输出）
    - [ ] `out/pretrain` 下有 checkpoint 文件（`iter_*.pth` / `epoch_*.pth`）
- [ ] 第二步：切换目标训练配置（如多卡）
  - 命令示例：`torchrun --standalone --nproc_per_node=4 pretrain.py`
  - 验证项：
    - [ ] 多卡进程启动正常
    - [ ] 训练速度和显存占用符合预期
- [ ] 第三步：完成预训练并选择最终 checkpoint
  - [ ] 记录最终采用的模型文件路径（用于 challenge eval）
- [ ] 第四步：对接 BabyLM challenge 官方 eval 项目
  - [ ] 按官方 eval 仓库说明加载你的 checkpoint 并运行评测
  - [ ] 保存评测结果（日志/分数）以便对比后续实验

## 3) 明确不做（当前阶段）

- [ ] 不改 SFT 相关代码（`sft.py`、`sft_data_process.py`）
- [ ] 不改本仓库内 eval 脚本（`eval.py`、`eval_pretrain.py`）
- [ ] 不做“未来多语言扩展”的工程化重构（如通用化语言配置）

## 4) 备注

- 当前正式训练数据：`data/merged_multilingual_zh4_en3_nl3_100m.bin`
- `_smoke.bin` 仅用于小样本测试，可忽略

### 本次临时改动记录（smoke run）

- `dataset.py`
  - memmap 分支文件打开模式：`'r'` -> `'rb'`
- `pretrain.py`
  - `log_interval`：`100` -> `1`（保留原值注释）
  - `save_interval`：`10000` -> `20`（保留原值注释）
  - `batch_size`：`32` -> `4`（保留原值注释）
  - `max_seq_len`：`512` -> `256`（保留原值注释）
  - `data_path_list`：`merged_multilingual_zh4_en3_nl3_100m.bin` -> `_smoke.bin`（保留原路径注释）
