class GPTCommonConfig:
    """ params common across all GPT models """
    embed_dropout = 0.1
    residual_dropout = 0.1
    attn_dropout = 0.1

    def __init__(self, vocab_size, seq_len, num_layer, num_head, embed_dim):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_layer = num_layer
        self.num_head = num_head
        self.embed_dim = embed_dim

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    lr = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)