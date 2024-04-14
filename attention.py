import torch
import torch.nn as nn


class SelfAttention(nn.module):
    def __init__(self, embedding_size: int, num_heads: int = 8):
        self.__init__()

        self.wk = torch.rand([embedding_size, 64, num_heads])
        self.wq = torch.rand([embedding_size, 64, num_heads])
        self.wv = torch.rand([embedding_size, 64, num_heads])

    def forward(self, input):
        k = torch.einsum("ldk,dhk->lhk", input, self.wk)
        q = torch.einsum("ldk,dhk->lhk", input, self.wq)
        v = torch.einsum("ldk,dhk->lhk", input, self.wv)

        query_response = torch.softmax(
            torch.einsum("lhk,hlk->llk", k, q.transpose(0, 1))
        )
        atten = torch.einsum("llk,lhk->lhk", query_response, v)
        return atten
