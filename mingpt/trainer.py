import math
import logging
from torch.functional import split
from torch.serialization import load

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

class Trainer():
    """Boiler plate trainer. can be applied to any arbitrary NN, not just transformers"""
    
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = nn.DataParallel(self.model).to(self.device)
    
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)
    
    @staticmethod
    def _run_epoch(model, optimizer, train_dataset, test_dataset, tokens, split, epoch, config, device):

        is_train = split=='train'
        model.train(is_train)
        dataset = train_dataset if is_train else test_dataset
        loader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=config.batch_size,\
                            num_workers=config.num_workers)
        
        losses = []
        pbar = tqdm(enumerate(loader), total=len(loader) if is_train else enumerate(loader))
        
        for i, (x,y) in pbar:
            x = x.to(device)
            y = y.to(device)

            with torch.set_grad_enabled(is_train):
                logits, loss = model(x,y)
                loss = loss.mean()
                losses.append(loss.item())
            
            if is_train:
                model.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), config.grad_norm_clip)
                optimizer.step()

                if config.lr_decay:
                    tokens += (y >= 0).sum()
                    if tokens < config.warmup_tokens:
                        lr_multiplier = float(tokens) / float(max(1, config.warmup_tokens))
                    else:
                        # cosine learning rate decay
                        progress = float(tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                        lr_multiplier = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                    
                    lr = config.lr * lr_multiplier
                    
                    for param_group in optimizer.param_group:
                        param_group['lr'] = lr
                else:
                    lr = config.lr
            
                pbar.set_description(f"epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. lr {lr:e}")

        if not is_train:
            test_loss = float(np.mean(losses))
            logger.info("test loss: %f", test_loss)
            return test_loss, tokens
        
        return tokens

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        self.optimizer = raw_model.configure_optimizers(config)

        best_loss = float('inf')
        self.tokens = 0 # for lr decay

        for epoch in range(config.max_epochs):

            self.tokens = self._run_epoch(self.model, self.optimizer,
                                          self.train_dataset, self.test_dataset,
                                          self.tokens, 'test', epoch, self.config, self.device)
            if self.test_dataset is not None:
                test_loss, self.tokens = self._run_epoch(self.model, self.optimizer,
                                                         self.train_dataset, self.test_dataset,
                                                         self.tokens, 'test', epoch, self.config, self.device)
            
            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()