from typing import Union

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from .word2vec_dataset import Word2VecDataset

DEFAULT_LR_FOR_ADAM = 1e-3
KARPATHY_CONSTANT = 3e-4


def train(w2v_dataset: Word2VecDataset,
          model: nn.Module,
          n_epoch: int = 10,
          batch_size: int = 32,
          shuffle: bool = True,
          verbose: Union[int, bool] = False,
          device: str = "cpu",
          optimizer_=None,
          optimizer_params=None,
          criterion=None):
    """Inspired by github.com/goddoe"""
    data_loader = DataLoader(w2v_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR_FOR_ADAM) \
        if optimizer_ is None else optimizer_(model.parameters(),
                                              **optimizer_params)

    loss_fn = nn.CrossEntropyLoss() if criterion is None else criterion
    loss_list = []

    model.to(device)

    for epoch_i in range(n_epoch):
        for batch_i, (X, Y) in enumerate(data_loader):
            Y = Y.view(-1)
            X, Y = X.to(device), Y.to(device)
            model.zero_grad()
            pred_log_prob = model(X)

            loss = loss_fn(pred_log_prob, Y)

            loss.backward()
            loss_list.append(float(loss.to('cpu').data.numpy()))

            optimizer.step()

            if verbose is not False:
                if epoch_i % verbose == 0:
                    print("loss : {:.3f}".format(loss_list[-1]))

    return loss_list
