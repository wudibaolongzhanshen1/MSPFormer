import numpy as np
import torch
from einops import rearrange
from timm.layers import DropPath, trunc_normal_
from torch import nn


class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    # x shape: (batch_size, seq_len, in_features)
    def forward(self, x):
        x = x.transpose(1, 2)
        x1 = self.conv1(x)
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)
        x2 = self.conv2(x)
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)
        out1 = x1 * x2_2
        out2 = x2 * x1_2
        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x


class Pyramid_ICB(nn.Module):
    def __init__(self, in_features=1, hidden_features=64, drop=0.):
        super(Pyramid_ICB, self).__init__()
        # 卷积层，用不同的卷积核提取不同尺度的特征
        self.conv3 = nn.Conv1d(in_features, hidden_features, kernel_size=3, padding=1)  # 3x卷积核
        self.conv5 = nn.Conv1d(in_features, hidden_features, kernel_size=5, padding=2)  # 5x卷积核
        self.conv7 = nn.Conv1d(in_features, hidden_features, kernel_size=7, padding=3)  # 7x卷积核
        # 使用一个 1x1 的卷积融合不同尺度的特征
        self.conv_fusion = nn.Conv1d(hidden_features * 3, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):
        # 输入形状: (batch_size, seq_len, 1)
        # 转换为 (batch_size, 1, seq_len) 以适配卷积操作
        x = x.transpose(1, 2)
        # 提取不同尺度的特征
        feature3 = self.conv3(x)
        feature5 = self.conv5(x)
        feature7 = self.conv7(x)
        feature3, feature5, feature7 = self.drop(self.act(feature3)), self.drop(self.act(feature5)), self.drop(
            self.act(feature7))
        # 拼接不同尺度的特征
        fused_features = torch.cat([feature3, feature5, feature7], dim=1)  # 拼接在通道维度
        # 使用 1x1 卷积进行特征融合
        fused_features = self.conv_fusion(fused_features)
        fused_features = fused_features.transpose(1, 2)
        # 输出形状: (batch_size, seq_len, 1)
        return fused_features


class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, args, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.ones(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.ones(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1))  # * 0.5)
        self.adaptive_filter = args.adaptive_filter

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape
        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions
        # Normalize energy
        normalized_energy = energy / (median_energy + 1e-6)
        adaptive_mask = (((normalized_energy > self.threshold_param).float() - self.threshold_param).detach()
                         + self.threshold_param)
        adaptive_mask = adaptive_mask.unsqueeze(-1)
        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape
        dtype = x_in.dtype
        x = x_in.to(torch.float32)
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x, dim=1, norm='ortho')  # shape:(B,N/2,C)
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)
            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2
        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape
        return x


class TSLANet_layer(nn.Module):
    def __init__(self, args, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(args, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.ICB = args.ICB
        self.ASB = args.ASB
        self.args = args

    def forward(self, x):
        # Check if both ASB and ICB are true
        if self.ICB and self.ASB:
            x1 = self.norm1(x)
            x1 = self.asb(x1)
            x1 = self.norm2(x1)
            x1 = self.icb(x1)
            x = x + self.drop_path(x1)
        # If only ICB is true
        elif self.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif self.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.patch_size = args.patch_size
        self.stride = self.patch_size // 2
        num_patches = int((args.seq_len - self.patch_size) / self.stride + 1)
        # Layers/Networks
        self.input_layer = nn.Linear(self.patch_size, args.emb_dim)
        dpr = [x.item() for x in torch.linspace(0, args.dropout, args.depth)]  # stochastic depth decay rule
        self.tsla_blocks = nn.ModuleList([
            TSLANet_layer(args, dim=args.emb_dim, drop=args.dropout, drop_path=dpr[i])
            for i in range(args.depth)]
        )
        self.args = args
        self.out_layer = nn.Linear(args.emb_dim * num_patches, args.label_len + args.pred_len)

    def forward(self, x):
        B, L, M = x.shape
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x /= stdev
        x = rearrange(x, 'b l m -> b m l')
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        x = self.input_layer(x)
        for tsla_blk in self.tsla_blocks:
            x = tsla_blk(x)
        outputs = self.out_layer(x.reshape(B * M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = outputs * stdev
        outputs = outputs + means
        return outputs


def get_frequency_modes(seq_len, modes=64, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len // 2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index


# ########## fourier layer #############
class FourierBlock(nn.Module):
    def __init__(self, d_model, seq_len, pred_len, modes=16, mode_select_method='low'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.index = get_frequency_modes(seq_len, modes, mode_select_method)
        self.scale = (1 / (d_model ** 0.5))  # 更合理的缩放因子
        self.weights = nn.Parameter(
            self.scale * torch.rand(d_model, d_model, len(self.index), dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # 调整矩阵乘法维度 [batch, dim] x [dim, dim] -> [batch, dim]
        return torch.einsum("bd, do->bo", input, weights)

    def forward(self, x):
        # 输入q形状: [B, L, D]
        B, L, D = x.shape
        x = x.permute(0, 2, 1)
        x_ft = torch.fft.rfft(x, dim=-1)  # 输出形状 [B, D, Freq]
        # 频域操作
        out_ft = torch.zeros(B, D, x_ft.size(-1), device=x.device, dtype=torch.cfloat)
        out_ft2 = torch.zeros_like(x_ft)
        # 修改点4：单头处理循环
        for wi, freq_idx in enumerate(self.index):
            # 对每个选中的频率分量进行全维度变换
            out_ft[:, :, wi] = self.compl_mul1d(x_ft[:, :, freq_idx], self.weights[:, :, wi])
        for wi, freq_idx in enumerate(self.index):
            # 保持原始频率位置
            out_ft2[:, :, freq_idx] = out_ft[:, :, wi]
        res_ft = torch.zeros(
            [out_ft2.size(0), out_ft2.size(1), int((self.seq_len + self.pred_len) / 2 + 1)],
            dtype=out_ft2.dtype).to(out_ft2.device)
        res_ft[:, :, :out_ft2.size(2)] = out_ft2  # zero padding
        # 逆变换恢复时域信号
        x = torch.fft.irfft(res_ft)
        # 恢复维度顺序 [B, L, D]
        return x.permute(0, 2, 1)