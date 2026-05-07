import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, device: str, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.device = device
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features, device=torch.device(self.device)))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features, device=torch.device(self.device)))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()  # 沿时间维度求平均，得到各通道的时间均值
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class TimeMixerPPRevIN(nn.Module):
    def __init__(self, n_features: int, eps: float = 1e-9, affine: bool = True):
        super(TimeMixerPPRevIN, self).__init__()
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, missing_mask=None, mode: str = "norm"):
        if mode == "norm":
            x = self._normalize(x, missing_mask)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.n_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.n_features))

    def _normalize(self, x, missing_mask=None):
        dim2reduce = tuple(range(1, x.ndim - 1))

        # calculate mean and stdev
        if missing_mask is None:
            # original implementation
            mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps)
        else:
            # pypots implementation for POTS data
            missing_sum = torch.sum(missing_mask == 1, dim=dim2reduce, keepdim=True) + self.eps
            mean = torch.sum(x, dim=dim2reduce, keepdim=True) / missing_sum
            x_enc = x.masked_fill(missing_mask == 0, 0)
            variance = torch.sum(x_enc * x_enc, dim=dim2reduce, keepdim=True) + self.eps
            stdev = torch.sqrt(variance / missing_sum)

        # detach mean and stdev to avoid backpropagation
        self.mean = mean.detach()
        self.stdev = stdev.detach()
        # normalize the input
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            # apply affine transformation
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        # reverse affine transformation
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        # denormalize the input
        x = x * self.stdev
        x = x + self.mean
        return x


class Leddam_RevIN(nn.Module):
    def __init__(self, channel, output_dim):
        super(Leddam_RevIN, self).__init__()
        self.output_dim = output_dim

    def forward(self, x):
        # Calculate mean and std along dim=1
        self.means = x.mean(1, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + 1e-5)

        # Normalize using learned parameters
        x_normalized = (x - self.means) / self.stdev
        return x_normalized

    def inverse_normalize(self, x_normalized):
        x_normalized = x_normalized * \
                       (self.stdev[:, 0, :].unsqueeze(1).repeat(
                           1, self.output_dim, 1))
        x_normalized = x_normalized + \
                       (self.means[:, 0, :].unsqueeze(1).repeat(
                           1, self.output_dim, 1))
        return x_normalized
