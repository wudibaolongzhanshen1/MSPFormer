import torch
from einops import rearrange
from torch import nn


class AMTWrapper(nn.Module):
    def __init__(self, model, args):
        super(AMTWrapper, self).__init__()
        self.predict_model = model
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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, L, D = x_enc.shape
        dec_out = self.predict_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        ori_out = self.upsample(dec_out.transpose(1, 2))  # (batch_size, enc_in, mt_dims)
        ori_out = rearrange(ori_out, 'batch_size enc_in mt_dims ->enc_in batch_size mt_dims')
        init = self.init_out[:, :B, :]  # (c_out, batch_size, mt_dims)
        # (channels, batch_size, )
        final_out, attn = self.modal_transform(init, ori_out, ori_out)  # (c_out, batch_size, mt_dims)
        final_out = self.fc(final_out)  # (c_out, batch_size, pred_len)
        return final_out.permute(1, 2, 0), dec_out


class MlpWrapperLevel3(nn.Module):
    def __init__(self, model, args):
        super(MlpWrapperLevel3, self).__init__()
        self.predict_model = model
        self.modal_transform = nn.Sequential(
            nn.Linear(args.enc_in, args.mt_dims),
            nn.Mish(),
            nn.Linear(args.mt_dims, args.mt_dims),
            nn.Mish(),
            nn.Linear(args.mt_dims, args.c_out)
        )

    def get_parameter(self, target: str) -> "Parameter":
        if target == "predict_model":
            return list(self.predict_model.parameters())
        elif target == "modal_transform":
            return list(self.modal_transform.parameters())
        else:
            raise ValueError(f"target {target} is not valid")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.predict_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        y = self.modal_transform(dec_out)
        return y, dec_out


class MlpWrapperLevel2(nn.Module):
    def __init__(self, model, args):
        super(MlpWrapperLevel2, self).__init__()
        self.predict_model = model
        self.modal_transform = nn.Sequential(
            nn.Linear(args.enc_in, args.mt_dims),
            nn.Mish(),
            nn.Linear(args.mt_dims, args.c_out)
        )

    def get_parameter(self, target: str) -> "Parameter":
        if target == "predict_model":
            return list(self.predict_model.parameters())
        elif target == "modal_transform":
            return list(self.modal_transform.parameters())
        else:
            raise ValueError(f"target {target} is not valid")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.predict_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        y = self.modal_transform(dec_out)
        return y, dec_out


class MlpWrapperLevel1(nn.Module):
    def __init__(self, model, args):
        super(MlpWrapperLevel1, self).__init__()
        self.predict_model = model
        self.modal_transform = nn.Linear(args.enc_in, args.c_out)

    def get_parameter(self, target: str) -> "Parameter":
        if target == "predict_model":
            return list(self.predict_model.parameters())
        elif target == "modal_transform":
            return list(self.modal_transform.parameters())
        else:
            raise ValueError(f"target {target} is not valid")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.predict_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        y = self.modal_transform(dec_out)
        return y, dec_out