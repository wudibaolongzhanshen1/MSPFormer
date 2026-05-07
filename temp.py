import argparse

import torch
from torch import nn

from models import MSPFormer
from models.MSPFormer import MSPFormerPlusAMT
from models.ModelWrapper import AMTWrapper


class ModelWrapper(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        print(f'using model {args.model}')
        self.criterion = nn.MSELoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        self.mae = nn.L1Loss(reduction='mean')
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.initialize_he_extended()

    def initialize_he_extended(self):
        for module in self.model.modules():
            # 初始化线性层
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            # 初始化卷积层
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def calculate_loss(self, batch):
        """合并_calculate_loss逻辑"""
        batch_x, batch_y, x_mark, y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        if self.args.features == 'MS':
            batch_x = batch_x[:, -self.args.seq_len:, :-1]  # 去掉最后一列OT
            future_x = batch_y[:, -self.args.pred_len:, :-1]
            batch_y = batch_y[:, -self.args.pred_len:, -1:]  # 最后一列OT的未来数据
        else:
            batch_x = batch_x[:, -self.args.seq_len:, :]
            future_x = batch_y[:, -self.args.pred_len:, :]
            batch_y = batch_y[:, -self.args.pred_len:, :]
        outputs, ori_out = self(batch_x)
        outputs = outputs[:, -self.args.pred_len:, :]
        mse_loss = self.criterion(outputs, batch_y)
        total_loss = mse_loss
        if self.args.use_ori_seq_loss:
            ori_loss = self.args.ori_seq_loss_weight * self.criterion(ori_out, future_x)
            total_loss += ori_loss
        mse = self.mse(outputs, batch_y)
        mae = self.mae(outputs, batch_y)
        # 损失计算逻辑保持不变...
        loss_dict = {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'mse': mse,
            'mae': mae,
            # 其他定制化损失项...
        }
        if self.args.use_ori_seq_loss:
            loss_dict['ori_loss'] = ori_loss
        return loss_dict


parser = argparse.ArgumentParser(description='MSPFormer')

# basic config
parser.add_argument('--task_name', type=str, default='cross_sequence_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='train', help='model id')
parser.add_argument('--model', type=str, default='DLinear.sh',
                    help='model name, options: [Autoformer, Transformer, TimesNet]')
parser.add_argument('--wrapper', type=str, default='AMT', help='wrapper name, options: [AMT, Mlp3, Mlp2, Mlp1]')

# data loader
parser.add_argument('--data', type=str, default='traffic', help='dataset type')
parser.add_argument('--root_path', type=str, default='E:\MyDataset\\traffic', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='traffic.csv', help='data file')
parser.add_argument('--features', type=str, default='MS',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--scale', type=int, default=1)
parser.add_argument('--data_size', type=float, default=1)
parser.add_argument('--in_dataset_augmentation', type=bool, default=False)
parser.add_argument('--closer_data_aug_more', type=bool, default=True)
parser.add_argument('--aug_data_size', type=int, default=1)
parser.add_argument('--aug_method', type=str, default='f_mix')
parser.add_argument('--aug_rate', type=float, default=0.5, help='augmentation rate for training data')

# forecasting task
parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=6, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

# model define
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--channel_attn', type=bool, default=True)
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=861, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=512, help='dimension of fcn')
parser.add_argument('--patch_size', type=int, default=12, help='patch size')
parser.add_argument('--mt_dims', type=int, default=128)
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=0, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='whether to use future_temporal_feature; True 1 False 0')
parser.add_argument('--use_ori_seq_loss', type=bool, default=True)
parser.add_argument('--ori_seq_loss_weight', type=float, default=0.2)

# imputation task
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

# anomaly detection task
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=4e-5, help='optimizer learning rate')
parser.add_argument('--test_time_train', type=bool, default=True)
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='TST', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=True)
parser.add_argument('--comment', type=str, default='none', help='com')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--device', type=str, default='cuda:0')

# de-stationary projector params
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

args = parser.parse_args()
model = ModelWrapper(args, MSPFormerPlusAMT(args))
model_state_dict = torch.load('checkpoints/cross_sequence_forecast_train_none_MSPFormer_traffic_sl336_pl6_dm128_nh8_el2_dl1_df512_fc3_ebtimeF_dtTrue_test_0/best_model.pth',
                              map_location=torch.device("cuda"))
unexpected, missing = model.load_state_dict(model_state_dict, strict=False)
if unexpected:
    print(f"Unexpected keys in state_dict: {unexpected}")
if missing:
    print(f"Missing keys in state_dict: {missing}")
print(model.parameters())
