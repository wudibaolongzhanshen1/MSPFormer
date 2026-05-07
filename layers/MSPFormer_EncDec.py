import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.SelfAttention_Family import TwoStageAttentionLayer, AttentionLayer, FullAttention


class SegMerging(nn.Module):
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0:
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    def __init__(self, args, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, factor=10):
        super(scale_block, self).__init__()
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(args, seg_num, factor, d_model, n_heads, d_ff, dropout))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        _, ts_dim, _, _ = x.shape
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        for layer in self.encode_layers:
            x = layer(x)
        return x, None


class MultiScaleBlock(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, seg_num=10, padding=0, factor=10):
        super(MultiScaleBlock, self).__init__()
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        self.encode_layers = nn.ModuleList()
        self.config = configs
        self.padding = padding
        self.get_mask()
        for i in range(depth):
            self.encode_layers.append(
                TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, d_ff, dropout))

    def get_mask(self):
        patch_size = self.config['patch_size']
        seq_len = self.config['seq_len'] + self.padding
        mask = torch.ones(((3 * seq_len) // (2 * patch_size), (3 * seq_len) // (2 * patch_size)),
                          dtype=torch.bool)
        mask[:seq_len // patch_size, :seq_len // patch_size] = False
        mask[seq_len // patch_size:, seq_len // patch_size:] = False
        for i in range(seq_len // patch_size, (3 * seq_len) // (2 * patch_size)):
            mask[i, 2 * (i - seq_len // patch_size)] = False
            mask[2 * (i - seq_len // patch_size), i] = False
            mask[i, 2 * (i - seq_len // patch_size) + 1] = False
            mask[2 * (i - seq_len // patch_size) + 1, i] = False
        self.mask = mask.to(torch.device(self.config['device']))

    def forward(self, x, channel_mask=None):
        _, ts_dim, _, _ = x.shape
        for layer in self.encode_layers:
            x = layer(x, self.mask)
        return x, None


class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x, channel_mask=None):
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x, attns = block(x, channel_mask)
            encode_x.append(x)

        return encode_x, None


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None, )
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        return dec_output, layer_predict


class MultiscaleDecoderLayer(nn.Module):
    def __init__(self, config, self_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super(MultiscaleDecoderLayer, self).__init__()
        self.config = config
        self.self_attention = self_attention
        self.cross_attention = AttentionLayer(
            FullAttention(False, config['factor'], attention_dropout=config['dropout'],
                          output_attention=False),
            config['d_model'], config['n_heads'])
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        pad_in_len = math.ceil(1.0 * self.config['seq_len'] / (2 * self.config['patch_size'])) * (
                2 * self.config['patch_size'])
        cross1 = cross[:, :, :pad_in_len // self.config['patch_size'], :]  # (b, ts_d, in_seg_num, d_model)
        cross2 = cross[:, :, pad_in_len // self.config['patch_size']:, :]
        cross1 = rearrange(cross1, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        cross2 = rearrange(cross2, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        cross2 = torch.repeat_interleave(cross2, repeats=2, dim=1)  # (b, ts_d, in_seg_num, d_model)
        cross1 = cross1 + self.cross_attention(cross1, cross2, cross2, None, None, None)[0]
        cross1 = self.norm1(cross1 + self.MLP1(cross1))
        cross2 = cross2 + self.cross_attention(cross2, cross1, cross1, None, None, None)[0]
        cross2 = self.norm2(cross2 + self.MLP1(cross2))
        cross = cross1 * cross2
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None)
        y = x = self.norm3(self.dropout(tmp))
        y = self.MLP2(y)
        dec_output = self.norm4(x + y)
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        return dec_output, layer_predict


class MultiscaleDecoderLayerRmBiCAF(nn.Module):
    def __init__(self, args, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super(MultiscaleDecoderLayerRmBiCAF, self).__init__()
        self.args = args
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None)
        y = x = self.norm1(self.dropout(tmp))
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        return dec_output, layer_predict


class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList(layers)

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        return final_predict