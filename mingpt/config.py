class CommonConfig:
    """ params common across all GPT models """
    embed_dropout = 0.1
    residual_dropout = 0.1
    attn_dropout = 0.1

    def __init__(self, vocab_size, seq_len, embed_dim, num_layer, num_head):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_layer = num_layer
        self.num_head = num_head