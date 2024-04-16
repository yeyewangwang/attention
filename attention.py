import torch
import torch.nn as nn


class SelfAttention(nn.module):
    def __init__(self, embedding_size: int, num_heads: int = 8):
        """
        This class implements a self attention mechanism following the transformer architecture.

        Embedding size refers to the dimension size which the neural network considers as hiddden representation
        for each token.
        """
        self.__init__()

        assert (
            embedding_size % num_heads == 0
        ), f"Num_heads ({num_heads}) does not divide embedding_size ({embedding_size})."

        self.k = embedding_size // num_heads

        self.num_heads = num_heads

        self.key_layer = nn.Linear(embedding_size, self.k)
        self.query_layer = nn.Linear(embedding_size, self.k)
        self.value_layer = nn.Linear(embedding_size, self.k)

    def forward(self, input):
        """
        Perform single-head self-attention.

        Notations:
        b: batch size, number of windows
        w: window size. Perform attend for each window on the window itself.
        d: embedding size
        k: output dimension of key, query, and value dense layers

        Input shape: (b, w, d)
        """
        # Multiplications, with all heads vectorized along the -2nd dimension,
        k = self.key_layer(input)
        q = self.query_layer(input)
        v = self.value_layer(input)

        # The scaling factor for key query dot product
        scaling_factor = torch.sqrt(self.k)

        # During the attention operation (dot, scale, weighted sum), split d into h heads,
        # place them into 0th dimension
        query_response = torch.softmax(
            torch.einsum("bwk,bkw->bww", k, q.transpose(1, 2)) / scaling_factor, dim=2
        )
        atten = torch.einsum("bww,wk->bwk", query_response, v)
        return atten

    def forward_multihead(self, input):
        """

        Notations:
        h: number of attention heads

        Input shape: (w, d)
        """
        # Reshape input into (w, h, k)
        input = torch.reshape(input, (self.num_heads, self.k))
        return
