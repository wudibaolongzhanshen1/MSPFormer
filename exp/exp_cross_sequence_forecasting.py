from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import os
import warnings

warnings.filterwarnings('ignore')

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


class Exp_Cross_Sequence_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Cross_Sequence_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        wrapper = self.wrapper_dict[self.args.wrapper](model, self.args).float()
        wrapper = ModelWrapper(self.args, wrapper)
        if self.args.use_multi_gpu and self.args.use_gpu:
            wrapper = nn.DataParallel(wrapper, device_ids=self.args.device_ids)
        return wrapper

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        WEIGHT_DECAY = 2e-5
        predict_optimizer = optim.AdamW(self.model.model.get_parameter('predict_model'), lr=self.args.learning_rate,
                                        weight_decay=WEIGHT_DECAY)
        modal_transform_optimizer = optim.AdamW(self.model.model.get_parameter('modal_transform'),
                                                lr=self.args.learning_rate, weight_decay=WEIGHT_DECAY)
        return predict_optimizer, modal_transform_optimizer

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        writer = SummaryWriter(log_dir=path)
        predict_optimizer, modal_transform_optimizer = self._select_optimizer()

        # 学习率调度器
        predict_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            predict_optimizer, mode='min', factor=0.1, patience=self.args.patience)
        modal_transform_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            modal_transform_optimizer, mode='min', factor=0.1, patience=self.args.patience)
        min_interval = 50
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_mse = 0
            train_mae = 0
            train_ori_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]", miniters=min_interval)
            batch_counter = 0
            for batch in progress_bar:
                predict_optimizer.zero_grad()
                modal_transform_optimizer.zero_grad()
                if self.args.use_amp:
                    with autocast():
                        loss_dict = self.model.calculate_loss(batch)
                    loss = loss_dict['total_loss']
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    scaler.step(predict_optimizer)
                    scaler.step(modal_transform_optimizer)
                    scaler.update()
                    predict_optimizer.zero_grad()
                    modal_transform_optimizer.zero_grad()
                else:
                    loss_dict = self.model.calculate_loss(batch)
                    loss = loss_dict['total_loss']
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                    predict_optimizer.step()
                    modal_transform_optimizer.step()
                    predict_optimizer.zero_grad()
                    modal_transform_optimizer.zero_grad()
                train_loss += loss.item()
                train_mse += loss_dict['mse'].item()
                train_mae += loss_dict['mae'].item()
                train_ori_loss = train_ori_loss + (loss_dict['ori_loss'].item() if self.args.use_ori_seq_loss else 0)
                batch_counter += 1
                if batch_counter % min_interval == 0:
                    postfix = {
                        'loss': f"{loss.item():.4f}",
                        'mse': f"{loss_dict['mse'].item():.4f}",
                        'mae': f"{loss_dict['mae'].item():.4f}",
                    }
                    if self.args.use_ori_seq_loss:
                        postfix['ori_loss'] = f"{loss_dict['ori_loss'].item():.4f}"
                    progress_bar.set_postfix(postfix)

            train_loss /= len(train_loader)
            train_mse /= len(train_loader)
            train_mae /= len(train_loader)
            train_ori_loss = train_ori_loss / len(train_loader) if self.args.use_ori_seq_loss else 0
            writer.add_scalar("train/mae", train_mae, epoch)
            writer.add_scalar("train/mse", train_mse, epoch)
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("train/ori_loss", train_ori_loss, epoch)
            print('====================================================')
            print_msg = f"Epoch {epoch + 1} Train Loss: {train_loss:.4f} | MSE: {train_mse:.4f} | MAE: {train_mae:.4f}"
            if self.args.use_ori_seq_loss:
                print_msg += f" | Ori Loss: {loss_dict['ori_loss'].item():.4f}"
            print(print_msg)
            print('====================================================')
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_mae = 0
            val_mse = 0
            val_ori_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validating"):
                    loss_dict = self.model.calculate_loss(batch)
                    val_loss += loss_dict['total_loss'].item()
                    val_mse += loss_dict['mse'].item()
                    val_mae += loss_dict['mae'].item()
                    val_ori_loss = val_ori_loss + (loss_dict['ori_loss'].item() if self.args.use_ori_seq_loss else 0)
            val_loss = val_loss / len(val_loader)
            val_mae /= len(val_loader)
            val_mse = val_mse / len(val_loader)
            val_ori_loss = val_ori_loss / len(val_loader) if self.args.use_ori_seq_loss else 0
            writer.add_scalar("val/mse", val_mse, epoch)
            writer.add_scalar("val/mae", val_mae, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/ori_loss", val_ori_loss, epoch)
            print('====================================================')
            print_msg = f"Epoch {epoch + 1} Validation LOSS: {val_loss:.4f} | MSE: {val_mse:.4f} | MAE: {val_mae:.4f}"
            if self.args.use_ori_seq_loss:
                print_msg += f" | Ori Loss: {loss_dict['ori_loss'].item():.4f}"
            print(print_msg)
            print('====================================================')
            predict_scheduler.step(val_loss)  # 更新学习率
            modal_transform_scheduler.step(val_loss)
            # 早停和模型保存
            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
        print(f'Loading best model from {path}')
        self.model.load_state_dict(torch.load(path, map_location=self.model.device))
        self.model.eval()
        self.model.to(device=self.model.device)
        test_loss = 0
        test_mse = 0
        test_mae = 0
        test_ori_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                loss_dict = self.model.calculate_loss(batch)
                test_loss += loss_dict['total_loss'].item()
                test_mae += loss_dict['mae'].item()
                test_mse += loss_dict['mse'].item()
                test_ori_loss = test_ori_loss + (loss_dict['ori_loss'].item() if self.args.use_ori_seq_loss else 0)
        test_mse = test_mse / len(test_loader)
        test_mae = test_mae / len(test_loader)
        test_ori_loss = test_ori_loss / len(test_loader) if self.args.use_ori_seq_loss else 0
        test_loss = test_loss / len(test_loader)
        print('====================================================')
        print_msg = f"Final Test LOSS: {test_loss:.4f} | MSE: {test_mse:.4f} | MAE: {test_mae:.4f}"
        if self.args.use_ori_seq_loss:
            print_msg += f" | Ori Loss: {test_ori_loss:.4f}"
        print(print_msg)
        print('====================================================')
        return
