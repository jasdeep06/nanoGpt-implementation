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
    
    x = torch.tensor(arr[batch_range_X].astype(np.int64))
    y = torch.tensor(arr[batch_range_Y].astype(np.int64))

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

# Model configuration
n_layer = 12  # Number of transformer blocks
n_head = 12  # Number of attention heads
n_embd = 768  # Embedding dimension
batch_size = 4  # Number of samples in each batch
block_size = 1024  # Length of input sequence
bias = False  # Whether to include bias in the model
dropout = 0.0  # Dropout rate
vocab_size = 50304  # Size of the vocabulary

# Optimization parameters
weight_decay = 1e-1  # Weight decay for regularization
beta1 = 0.9  # Beta1 parameter for Adam optimizer
beta2 = 0.95  # Beta2 parameter for Adam optimizer
learning_rate = 6e-4  # Learning rate
eval_iters = 2  # Number of iterations for evaluation
warmup_iters = 20  # Number of warmup iterations for learning rate
lr_decay_iters = 1000  # Number of iterations for learning rate decay
min_lr = 6e-5  # Minimum learning rate
eval_interval = 20  # Interval for evaluation
max_iters = 1000  # Maximum number of iterations
device = 'cuda'  # Device for training (cuda or cpu)
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # Data type for training

run_index = 2  # Index of the current run
out_dir = f"runs/{run_index}"  # Output directory for saving checkpoints
gradient_accumulation_steps = 5 * 8  # Number of steps for gradient accumulation

grad_clip = 1.0  # Gradient clipping threshold


if not os.path.exists(out_dir):
    os.makedirs(out_dir)



# Allow tf32 on matmul
torch.backends.cuda.matmul.allow_tf32 = True

# Allow tf32 on cudnn
torch.backends.cudnn.allow_tf32 = True

# Determine the device type (cuda or cpu) for later use in torch.autocast
device_type = 'cuda' if 'cuda' in device else 'cpu'

# Map the dtype to the corresponding torch data type
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Create a null context if the device type is cpu, otherwise use torch.amp.autocast to enable automatic mixed precision training
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


model_args = dict(num_transformer_block=n_layer, num_transformer_heads=n_head, n_embed=n_embd, block_size=block_size,
                  bias=bias, dropout=dropout,vocab_size=vocab_size)

wandb.init(project="nanogpt-run", name="run " + str(run_index), config=model_args)


gptconf = GPTConfig(**model_args)
model = GPT(gptconf)

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizer(weight_decay, learning_rate, (beta1, beta2))


@torch.no_grad()
def estimate_loss():
    """
    Estimates the loss for the model on the train and validation splits.

    Returns:
        dict: A dictionary containing the mean loss for each split.
    """
    # Create an empty dictionary to store the output
    out = {}
    
    # Set the model to evaluation mode
    model.eval()
    
    # Iterate over the train and val splits
    for split in ['train', 'val']:
        
        # Create a tensor to store the losses for each iteration
        losses = torch.zeros(eval_iters)
        
        # Perform evaluation for eval_iters number of iterations
        for k in range(eval_iters):
            
            # Get a batch of input and target sequences
            X, Y = get_batch(split, batch_size, block_size)
            
            # Enable automatic mixed precision training
            with ctx:
                
                # Forward pass through the model to get logits and loss
                logits, loss = model(X, Y)
                
            # Store the loss value in the losses tensor
            losses[k] = loss.item()
        
        # Calculate the mean loss for the current split
        out[split] = losses.mean()
    
    # Set the model back to training mode
    model.train()
    
    # Return the dictionary containing the mean losses for each split
    return out

def get_lr(it):
    """
    Calculate the learning rate based on the current iteration.

    Args:
        it (int): The current iteration.

    Returns:
        float: The calculated learning rate.

    """
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



X,Y = get_batch('train',batch_size,block_size)
iter_num = 0
running_mfu = -1.0
best_val_loss = 1e9


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

    iter_num += 1

    if iter_num > max_iters:
        break