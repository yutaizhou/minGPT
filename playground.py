import logging

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.model import GPT
from mingpt.config import GPTCommonConfig, TrainerConfig
from mingpt.trainer import Trainer
from mingpt.dataset import CharDataset

text = open('input.txt', 'r').read()
train_dataset = CharDataset(text, seq_len=128)

# get GPT model
model_conf = GPTCommonConfig(train_dataset.vocab_size, train_dataset.seq_len,
                  num_layer=8, num_head=8, embed_dim=512)
model = GPT(model_conf)

# initialize a trainer instance and kick off training
tconf = TrainerConfig(max_epochs=2, batch_size=256, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*model.seq_len,
                      num_workers=4)
trainer = Trainer(model, train_dataset, None, tconf)
trainer.train()