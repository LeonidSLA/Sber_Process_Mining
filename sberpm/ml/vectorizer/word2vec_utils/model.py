import torch
from torch import nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    """Skip-Gram model, reference to github.com/PengFoo"""
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 padding_idx: int = 0):
        super(SkipGramModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.u_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx)
        self.v_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx)
        self._init_embeddings()

    def _init_embeddings(self):
        initrange = 0.5 / self.emb_dim
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        assert pos_u.size() == pos_v.size()
        # [batch_size x emb_dim]
        emb_u = self.u_embeddings(pos_u)
        # [batch_size x emb_dim]
        emb_v = self.v_embeddings(pos_v)
        # [batch_size x neg_sample_size x emb_dim]
        emb_neg = self.v_embeddings(neg_v)

        pos_score = torch.mul(emb_u, emb_v).squeeze()
        pos_score = torch.sum(pos_score, dim=1)
        pos_score = F.logsigmoid(pos_score)

        neg_score = torch.bmm(emb_neg, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-neg_score)

        return -1 * (torch.sum(pos_score) + torch.sum(neg_score))
