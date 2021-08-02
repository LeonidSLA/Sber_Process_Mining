import torch
import torch.nn as nn


class CBOWModel(nn.Module):
    """Base CBOW model\n
    Inspired by github.com/goddoe"""
    PADDING_IDX = 0

    def __init__(self,
                 vocab_size: int,
                 emb_dim: int):
        """

        Parameters
        ----------
        vocab_size : vocabulary size
        emb_dim : embedding size
        """
        super(CBOWModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.embeddings = nn.Embedding(vocab_size,
                                       emb_dim,
                                       self.PADDING_IDX,
                                       max_norm=1)
        self.linear_layer = nn.Linear(emb_dim, vocab_size)
        self._init_embeddings()

    def _init_embeddings(self):
        """
        Initializing weights

        Returns
        -------

        """
        initrange = 0.5 / self.emb_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_):
        """

        Parameters
        ----------
        input_ : [batch_size x seq_len]

        Returns
        -------
        output_distribution : torch.Tensor
        """
        # [batch_size x 2 * window_size x emb_dim]
        inp_embeds = self.embeddings(input_)
        # [batch_size x emb_size]
        aggregated_embeds = torch.sum(inp_embeds, dim=1)
        # [batch_size x vocab_size]
        # word_dist = F.log_softmax(self.linear_layer(aggregated_embeds), dim=1)
        logits = self.linear_layer(aggregated_embeds)

        return logits
