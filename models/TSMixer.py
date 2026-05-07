import torch
import torch.nn as nn
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, args):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(args.seq_len, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.seq_len),
            nn.Dropout(args.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(args.enc_in, args.d_model),
            nn.ReLU(),
            nn.Linear(args.d_model, args.enc_in),
            nn.Dropout(args.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.task_name = args.task_name
        self.layer = args.e_layers
        self.model = nn.ModuleList([ResBlock(args)
                                    for _ in range(args.e_layers)])
        self.pred_len = args.pred_len
        self.projection = nn.Linear(args.seq_len, args.pred_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):

        # x: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)

        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'cross_sequence_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        else:
            raise ValueError('Only forecast tasks implemented yet')