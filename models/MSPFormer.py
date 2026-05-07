import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.Crossformer_EncDec import Encoder, Decoder
from layers.Embed import PyraPatchEmbedding
from layers.MSPFormer_EncDec import MultiScaleBlock, MultiscaleDecoderLayer, MultiscaleDecoderLayerRmBiCAF
from layers.SelfAttention_Family import AttentionLayer, FullAttention, TwoStageAttentionLayer
from math import ceil


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.enc_in = config['enc_in']
        self.seq_len = config['seq_len']
        self.pred_len = config['pred_len']
        self.seg_len = config.get('patch_size', 12)
        self.win_size = 2  # 1:不用segmerge
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * config['seq_len'] / (2 * self.seg_len)) * (2 * self.seg_len)
        self.pad_out_len = ceil(1.0 * config['seq_len'] / (2 * self.seg_len)) * (2 * self.seg_len)
        self.in_seg_num = (3 * self.pad_in_len) // (2 * self.seg_len)
        self.out_seg_num = ceil(self.in_seg_num / (self.win_size ** (config['e_layers'] - 1)))
        self.head_nf = config['d_model'] * self.out_seg_num
        # Embedding
        self.enc_value_embedding = PyraPatchEmbedding(config['d_model'], self.seg_len,
                                                      self.pad_in_len - config['seq_len'], 0)
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, config['enc_in'], self.in_seg_num, config['d_model']))
        self.pre_norm = nn.LayerNorm(config['d_model'])
        # Encoder
        self.encoder = Encoder(
            [
                MultiScaleBlock(config, 1 if l == 0 else self.win_size, config['d_model'], config['n_heads'],
                                config['d_ff'],
                                1, config['dropout'],
                                self.in_seg_num,
                                self.pad_in_len - config['seq_len'],
                                config['factor']
                                ) for l in range(config['e_layers'])
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(1, config['enc_in'], (self.pad_out_len // self.seg_len), config['d_model']))
        self.decoder = Decoder(
            [
                MultiscaleDecoderLayer(
                    config,
                    TwoStageAttentionLayer(config, (self.pad_out_len // self.seg_len), config['factor'],
                                           config['d_model'], config['n_heads'], config['d_ff'], config['dropout']),
                    self.seg_len,
                    config['d_model'],
                    config['d_ff'],
                    dropout=config['dropout'],
                    # activation=config['activation'],
                )
                for l in range(config['e_layers'] + 1)
            ],
        )

    def forward(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(x_enc, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc.shape[0])
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out[:, -self.pred_len:, :]


class MSPFormerPlusAMT(nn.Module):
    def __init__(self, args):
        super(MSPFormerPlusAMT, self).__init__()
        self.args = args
        self.predict_model = Model(args)
        self.upsample = nn.Linear(args.pred_len, args.mt_dims)
        self.modal_transform = nn.MultiheadAttention(args.mt_dims, args.n_heads)
        init_out = torch.randn(args.c_out, args.batch_size, args.mt_dims).to(
            torch.device(args.device))
        self.register_buffer('init_out', init_out)
        self.fc = nn.Linear(args.mt_dims, args.pred_len)

    def get_parameter(self, target: str) -> "Parameter":
        if target == "predict_model":
            return list(self.predict_model.parameters())
        elif target == "modal_transform":
            return list(self.modal_transform.parameters()) + list(self.fc.parameters()) + list(
                self.upsample.parameters())
        else:
            raise ValueError(f"target {target} is not valid")

    def forward(self, x_enc, x_enc_mark=None, x_dec=None, x_dec_mark=None):
        B, L, D = x_enc.shape
        dec_out = self.predict_model(x_enc, x_enc_mark, x_dec, x_dec_mark)  # (batch_size, pred_len, d_model)
        ori_out = self.upsample(dec_out.transpose(1, 2))  # (batch_size, enc_in, mt_dims)
        ori_out = rearrange(ori_out, 'batch_size enc_in mt_dims ->enc_in batch_size mt_dims')
        init = self.init_out[:, :B, :]  # (c_out, batch_size, mt_dims)
        # (channels, batch_size, )
        final_out, attn = self.modal_transform(init, ori_out, ori_out)  # (c_out, batch_size, mt_dims)
        final_out = self.fc(final_out)  # (c_out, batch_size, pred_len)
        return final_out.permute(1, 2, 0), dec_out

