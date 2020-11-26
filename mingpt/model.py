from logging import FileHandler
import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class CausalSelfAttention(nn.Module):
    """ Multiheaded causal self attention (using mask) with proj at the end"""
    def __init__(self, config):
        super().__init__()
        assert config.embed_dim % config.num_head == 0 
        # get k q v for all heads
        self.to_k = nn.Linear(config.embed_dim, config.embed_dim)
        self.to_q = nn.Linear(config.embed_dim, config.embed_dim)
        self.to_v = nn.Linear(config.embed_dim, config.embed_dim)
        # regularization
        self.attn_dropout = config.attn_dropout
        self.residual_dropout = config.residual_dropout
        # output projection
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)


class Layer(nn.Module):
    """
    just a little decoder layer, consisting of causal self-attention and fully connnected layer
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
        self.tok_embed= nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_embed= nn.Parameter(torch.zeros(config.seq_len, config.embed_dim))
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        # transformer 
        self.tf = nn.Sequential([Layer(config) for _ in range(config.num_layer)])

        # decoder head
        self.ln = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

    def _init_weights(self, module):
        pass 

    def configure_opt(self, train_config):
        pass
    
    def forward(self, idc, targets = None):
        num_batch, seq_len = idc.shape
        assert seq_len < self.seq_len ,"Char sequence length too long"

        tok_embeddings = self.tok_embed(idc) # num_batch x seq_len x embed_dim
        pos_embeddings = self.pos_embed[:seq_len, :].unsqueeze(0) # 1 x seq_len x embed_dim

        x = self.embed_dropout(tok_embeddings + pos_embeddings) # num_batch x seq_len x embed_dim
        x = self.tf(x) 
        x = self.ln(x)
        logits = self.head(x) # num_batch x seq_len x vocab_size
        
        # calculate the loss if given targets
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss
