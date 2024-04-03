import torch
import numpy as np
from model import GPT, GPTConfig
from contextlib import nullcontext
import math
import wandb
import os


def get_batch(split,batch_size,block_size):
    train_arr,val_arr = load_dataset()
    arr = train_arr if split == 'train' else val_arr
    len_ds = len(arr)
    #+1 to counter range error while indexing y when len_ds-block_size is selected
    leading_indices = np.random.randint(0,len_ds-(block_size+1),(batch_size,))
    #[1,2,3,4] => [[1],[2],[3],[4]]
    leading_indices = leading_indices.reshape(batch_size,1)
    #[[1],[2],[3],[4]] => [[1,2,3],[2,3,4],[3,4,5],[4,5,6]] for block_size= 2
    batch_range_X = leading_indices + np.arange(block_size)
    #[[1,2,3],[2,3,4],[3,4,5],[4,5,6]] => [[2,3,4],[3,4,5],[4,5,6],[5,6,7]]
    batch_range_Y = batch_range_X + 1
    
    x = torch.tensor(arr[batch_range_X].astype(np.int32))
    y = torch.tensor(arr[batch_range_Y].astype(np.int32))

    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x,y



def load_dataset():
    train_arr = np.memmap('train.bin',dtype=np.uint16,mode='r')
    val_arr = np.memmap('val.bin',dtype=np.uint16,mode='r')
    return train_arr,val_arr

n_layer = 12
n_head = 12
n_embd = 768
batch_size = 12 
block_size = 1024
bias = False
dropout = 0.0
vocab_size = 50304

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
learning_rate = 6e-4
eval_iters = 2
warmup_iters = 20
lr_decay_iters = 6000
min_lr = 6e-5
eval_interval = 20
max_iters = 6000
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler


run_index = 1
out_dir = f"runs/{run_index}"
gradient_accumulation_steps = 5 * 8

grad_clip = 1.0 




torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout,vocab_size=vocab_size)

wandb.init(project="nanogpt-run", name="run " + run_index, config=model_args)


gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2))


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

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



X,y = get_batch('train',batch_size,block_size)
iter_num = 0
running_mfu = -1.0

while True:
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 :
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        wandb.log({
            "iter": iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": lr,
            "mfu": running_mfu*100, # convert to percentage
        })
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
    for micro_step in range(gradient_accumulation_steps):

        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train', batch_size, block_size)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    if iter_num > max_iters:
        break