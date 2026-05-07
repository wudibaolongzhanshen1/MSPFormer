"""
@author: S.Tahir.H.Rizvi
"""

import torch.nn as nn

from layers.RevIN import RevIN


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.in_num_features = args.enc_in
        self.Linear = nn.Linear(self.seq_len, self.seq_len)
        self.GeLU = nn.GELU()
        self.Hidden1 = nn.Linear(self.seq_len, self.pred_len)
        self.revin_layer = RevIN(self.in_num_features, args.device)

    ##############################################

    def forward(self, x, x_enc_mark=None, x_dec=None, x_dec_mark=None):
        ##############################################
        x = self.revin_layer(x, 'norm')
        x3 = x.permute(0, 2, 1)
        x3 = self.Linear(x3)
        x3 = self.GeLU(x3)
        x3 = self.Hidden1(x3)
        x3 = x3.permute(0, 2, 1)
        x3 = self.revin_layer(x3, 'denorm')
        ##############################################
        x = x3
        return x  # [Batch, Output length, Channel]
