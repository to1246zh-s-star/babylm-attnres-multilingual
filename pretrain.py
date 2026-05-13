import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from model import Transformer, ModelArgs
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset import PretrainDataset
import logging

#To run with DDP on 4 gpus on 1 node, example:
# torchrun --standalone --nproc_per_node=4 pretrain.py OR python -m torch.distributed.launch --nproc_per_node=4 pretrain.py
        
def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
# -----------------------------------------------------------------------------
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch):
    start_time=time.time()
    for step, (X, Y) in enumerate(train_loader):
        X=X.to(device)
        Y=Y.to(device)
        lr = get_lr(epoch*iter_per_epoch+step) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # and using the GradScaler if data type is float16
        #for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            model.require_backward_grad_sync = 0 == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
        #
        if (step + 1) % gradient_accumulation_steps == 0:
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)
        #打印日志
        if step % log_interval == 0:
            spend_time=time.time()-start_time
            logger.info(
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                        epoch,
                        max_epoch, 
                        step, 
                        iter_per_epoch,
                        loss.item(), 
                        optimizer.param_groups[-1]['lr'],
                        spend_time / (step+1) * iter_per_epoch // 60 - spend_time // 60))
        #
        if step % save_interval == 0:
            if ddp:
                if torch.distributed.get_rank() == 0:
                    model.eval()
                    torch.save(model.module.state_dict(),'{}/iter_{}.pth'.format(save_dir,int(step+epoch*iter_per_epoch)))
                    model.train()
            else:
                model.eval()
                torch.save(model.state_dict(),'{}/iter_{}.pth'.format(save_dir,int(step+epoch*iter_per_epoch)))
                model.train()


def init_model():
    # model init
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=16000,                 # ★ aligned with Regex-BBPE 16K
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    if init_from == "scratch":
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    return model
# I/O
if __name__=="__main__":
    out_dir = 'out'
    max_epoch = 10
    eval_interval = 1
    log_interval = 100
    # log_interval = 1  # smoke run
    save_interval = 1000
    # save_interval = 20  # smoke run
    eval_iters = 200
    eval_only = False
    always_save_checkpoint = True
    init_from = 'scratch'
    #
    gradient_accumulation_steps = 1
    batch_size = 32
    # batch_size = 4  # smoke run
    # model 
    max_seq_len = 512
    # max_seq_len = 256  # smoke run
    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    dropout = 0.0
    bias = False
    # adamw optimizer
    learning_rate = 3e-4
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    # learning rate decay settings
    decay_lr = True
    warmup_iters = 1000
    lr_decay_iters = 60000               # ★ adjusted for ~10 epochs on 100M tokens (was 10000)
    min_lr = 1e-5
    # DDP settings
    backend = 'nccl'
    # system
    device = 'cuda'
    dtype = 'bfloat16'                   # ★ changed from 'float16' for stability
    compile = False
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    # -----------------------------------------------------------------------------

    save_dir =os.path.join(out_dir , 'pretrain')
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = get_logger(os.path.join(save_dir,'log.log'))

    ddp = int(os.environ.get("RANK", -1)) != -1
    
    if ddp:
        if os.name == 'nt':
            init_process_group(backend="gloo")
        else:
            init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.cuda.amp.autocast(dtype=ptdtype)   # ★ explicitly pass dtype for bf16
    )
    #
    best_val_loss = 1e9
    #
    #-----init dataloader------
    data_path_list=[
        './data/merged_multilingual_100m.bin'        # ★ updated to new bin file
        # './data/smoke_1m.bin'                       # smoke run
    ]
    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len,memmap=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if ddp else None
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=train_sampler is None,
        num_workers=0 if os.name == 'nt' else 4,
        sampler=train_sampler
    )
    #init model
    model=init_model()
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    # wrap model into DDP container
    if ddp:
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])
        #
    raw_model = model.module if ddp else model
    # training loop
    iter_per_epoch=len(train_loader)
    for epoch in range(max_epoch):
        train_epoch(epoch)
        if ddp:
            if torch.distributed.get_rank() == 0:
                torch.save(raw_model.state_dict(),'{}/epoch_{}.pth'.format(save_dir,epoch))
        else:
            torch.save(raw_model.state_dict(),'{}/epoch_{}.pth'.format(save_dir,epoch))
    if ddp:
        destroy_process_group()