import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k) # num_batch x seq_len x k
    last_step_v = v[:,[-1]] # num_batch x 1 x k
    mask = logits < last_step_v # num_batch x seq_len x vocab_size

    out = logits.clone()
    out[mask] = -float('inf')
    return out

@torch.no_grad()
def sample(model, x, steps: int, temp: float=1, sample: bool=False, top_k: int=None):
    """
    x: num_batch x seq_len
    """
    seq_len = model.seq_len
    model.eval()

    for _ in range(steps):
        x_cond = x if x.shape[1] <= seq_len else x[:, -seq_len] # truncate if seq_len > context window
        logits, _ = model(x_cond) # num_batch x seq_len x vocab_size
        logits = logits[:,-1,:] / temp # scale logits at the final time step by temperature

        if top_k is not None: # optionally crop prob to top k options
            logits = top_k_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1) # num_batch x seq_len x vocab_size

        # sample from distribution or take most likely, num_batch x seq_len
        idx = torch.multinomial(probs, num_samples=1) if sample else torch.topk(probs, 1)[1]

        x = torch.cat((x, idx), dim=1)
    return x
