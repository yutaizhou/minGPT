from logging import FileHandler
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """
    Multiheaded causal self attention (using mask) with proj at the end
    Transform (num_batch x seq_len x embed_dim) input into output of same dim but with contextual repr
    """
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_head == 0 
        self.num_head = config.num_head
        # get k q v for all heads
        self.to_k = nn.Linear(config.embed_dim, config.embed_dim)
        self.to_q = nn.Linear(config.embed_dim, config.embed_dim)
        self.to_v = nn.Linear(config.embed_dim, config.embed_dim)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        # output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)
        # causal attention mask to ensure entry only attend to previous entries. Entries at or below diagonal is attended to
        mask = torch.tril(torch.ones((config.seq_len, config.seq_len))).unsqueeze(0).unsqueeze(0) # 1 x 1 x seq_len x seq_len
        self.register_buffer('mask', mask)

    def forward(self, x):
        num_batch, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.num_head 

        # calculate k, q, v for all heads in batch, the appetizers
        entry_dims = (num_batch, seq_len, self.num_head, head_dim)
        k = self.to_k(x).view(*entry_dims).transpose(1,2) # num_batch x num_head x seq_len x head_dim
        q = self.to_q(x).view(*entry_dims).transpose(1,2)
        v = self.to_v(x).view(*entry_dims).transpose(1,2)

        # causal self attention: the meat n potatos
        weights = q @ k.transpose(2,3) / math.sqrt(k.shape[-1]) # num_batch x num_head x seq_len x seq_len 
        weights = weights.masked_fill(self.mask[:,:,:seq_len, :seq_len] == 0, value = float('-inf')) # fill all entries above diagonal with -inf for masking
        weights = F.softmax(weights, dim=-1) 
        weights = self.attn_dropout(weights)
        out = weights @ v # num_batch x num_head x seq_len x x head_dim 
        out = out.transpose(1,2).contiguous().view(num_batch, seq_len, embed_dim)

        # output projection, the smol dessert
        return self.residual_dropout(self.out_proj(out))


class Layer(nn.Module):
    """
    just a little decoder layer, consisting of a causal self-attention layer and a fully connnected layer
    stack multiple of these in GPT to form the entire model
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = CausalSelfAttention(config)
        self.fc = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.residual_dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_len = config.seq_len

        # input token + positional embedding
        self.tok_embed = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(config.seq_len, config.embed_dim))
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        # transformer 
        self.tf = nn.Sequential(*[Layer(config) for _ in range(config.num_layer)])

        # decoder head
        self.ln = nn.LayerNorm(config.embed_dim)
        self.out_fc = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def forward(self, idc, targets = None):
        num_batch, seq_len = idc.shape
        assert seq_len <= self.seq_len ,"Char sequence length too long"

        tok_embeddings = self.tok_embed(idc) # num_batch x seq_len x embed_dim
        pos_embeddings = self.pos_embed[:seq_len, :].unsqueeze(0) # 1 x seq_len x embed_dim

        x = self.embed_dropout(tok_embeddings + pos_embeddings) # num_batch x seq_len x embed_dim
        x = self.tf(x) 
        x = self.ln(x)
        logits = self.out_fc(x) # num_batch x seq_len x vocab_size
        
        # calculate the loss if given targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0) 

    def configure_optimizer(self, train_config):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for module_name, module in self.named_modules():
            for param_name, param, in module.named_parameters():
                full_param_name = f'{module_name}.{param_name}' if module_name else param_name 

                if param_name.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(full_param_name)
                elif param_name.endswith('weight') and isinstance(module, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(full_param_name)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_embed')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_inter = decay & no_decay
        param_union = decay | no_decay
        assert len(param_inter) == 0, f"parameters {param_inter} made it into both decay/no_decay sets!" 
        assert len(param_dict.keys() - param_union) == 0, f"parameters {param_dict.keys() - param_union} were not separated into either decay/no_decay set!" \
                                                   

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
    

