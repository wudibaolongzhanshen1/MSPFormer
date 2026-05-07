
import torch
import torch.nn as nn
from einops import rearrange, repeat

from layers.Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer

from math import ceil


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.enc_in = args.enc_in
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.seg_len = args.patch_size
        self.win_size = 2  # 1:不用segmerge
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * args.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * args.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (args.e_layers - 1)))
        self.head_nf = args.d_model * self.out_seg_num
        # Embedding
        self.enc_value_embedding = PatchEmbedding(args.d_model, self.seg_len, self.seg_len,
                                                  self.pad_in_len - args.seq_len, 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, args.enc_in, self.in_seg_num, args.d_model))
        self.pre_norm = nn.LayerNorm(args.d_model)
        # Encoder
        self.encoder = Encoder(
            [
                scale_block(args, 1 if l == 0 else self.win_size, args.d_model, args.n_heads,
                            args.d_ff,
                            1, args.dropout,
                            self.in_seg_num if l == 0 else ceil(self.in_seg_num / self.win_size ** l), args.factor
                            ) for l in range(args.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, args.enc_in, (self.pad_out_len // self.seg_len), args.d_model))
        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(args, (self.pad_out_len // self.seg_len), args.factor,
                                           args.d_model, args.n_heads, args.d_ff, args.dropout),
                    AttentionLayer(
                        FullAttention(False, args.factor, attention_dropout=args.dropout,
                                      output_attention=False),
                        args.d_model, args.n_heads),
                    self.seg_len,
                    args.d_model,
                    args.d_ff,
                    dropout=args.dropout,
                    # activation=args.activation,
                )
                for l in range(args.e_layers + 1)
            ],
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]