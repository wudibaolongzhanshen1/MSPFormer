import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from sktime.datasets import load_from_tsfile_to_dataframe
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

from utils.tools import augmentation

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.args = args
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        if args.test_time_train:
            type_map = {'train': 0, 'val': 1, 'test': 1}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.data_path = data_path
        self.global_indices = []
        self.__read_data__()
        self.collect_all_data()
        if self.args.in_dataset_augmentation and self.set_type == 0:
            self.data_augmentation()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        if self.args.test_time_train:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 20 * 30 * 24]
            border2s = [18 * 30 * 24, 20 * 30 * 24, 20 * 30 * 24]

        self.border1 = border1 = border1s[self.set_type]
        self.border2 = border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if self.features == 'MS':
            self.data_x = self.data_x[:, :-1]
            self.data_y = self.data_y[:, -1:]
        self.data_stamp = data_stamp

    def regenerate_augmentation_data(self):
        self.collect_all_data()
        self.data_augmentation()

    def reload_data(self, x_data, y_data, x_time, y_time):
        self.x_data = x_data
        self.y_data = y_data
        self.x_time = x_time
        self.y_time = y_time

    def collect_all_data(self):
        self.x_data = []
        self.y_data = []
        self.x_time = []
        self.y_time = []
        self.global_indices = []
        data_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        mask_data_len = int((1 - self.args.data_size) * data_len) if self.args.data_size < 1 else 0
        for i in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
            if (self.set_type == 0 and i >= mask_data_len) or self.set_type != 0:
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                global_indices = np.arange(self.border1 + s_begin, self.border1 + s_begin + self.seq_len)
                self.global_indices.append(global_indices)
                self.x_data.append(self.data_x[s_begin:s_end])
                self.y_data.append(self.data_y[r_begin:r_end])
                self.x_time.append(self.data_stamp[s_begin:s_end])
                self.y_time.append(self.data_stamp[r_begin:r_end])

    def data_augmentation(self):
        origin_len = len(self.x_data)
        if not self.args.closer_data_aug_more:
            aug_size = [self.args.aug_data_size for i in range(origin_len)]
        else:
            aug_size = [int(self.args.aug_data_size * i / origin_len) + 1 for i in range(origin_len)]

        for i in range(origin_len):
            for _ in range(aug_size[i]):
                self.global_indices.append(self.global_indices[i].copy())
                aug = augmentation('dataset')
                if self.args.aug_method == 'f_mask':
                    x, y = aug.freq_dropout(self.x_data[i], self.y_data[i], dropout_rate=self.args.aug_rate)
                elif self.args.aug_method == 'f_mix':
                    rand = float(np.random.random(1))
                    i2 = int(rand * len(self.x_data))
                    x, y = aug.freq_mix(self.x_data[i], self.y_data[i], self.x_data[i2], self.y_data[i2],
                                        dropout_rate=self.args.aug_rate)
                else:
                    raise ValueError
                self.x_data.append(x)
                self.y_data.append(y)
                self.x_time.append(self.x_time[i])
                self.y_time.append(self.y_time[i])

    def __getitem__(self, index):
        seq_x = self.x_data[index]
        seq_y = self.y_data[index]
        indices = self.global_indices[index]
        return seq_x, seq_y, self.x_time[index], self.y_time[index], indices

    def __len__(self):
        return len(self.x_data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', test_time_train=False):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        self.test_time_train = test_time_train
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        if self.test_time_train:
            type_map = {'train': 0, 'val': 1, 'test': 1}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.global_indices = []
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        self.border1 = border1 = border1s[self.set_type]
        self.border2 = border2 = border2s[self.set_type]
        if self.test_time_train:
            border1s = [0, 18 * 30 * 24 - self.seq_len, 20 * 30 * 24]
            border2s = [18 * 30 * 24, 20 * 30 * 24, 20 * 30 * 24]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        if self.features == 'MS':
            self.data_x = self.data_x[:, :-1]
            self.data_y = self.data_y[:, -1:]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        global_indices = np.arange(self.border1 + s_begin, self.border1 + s_begin + self.seq_len)
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, global_indices

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        if args.test_time_train:
            type_map = {'train': 0, 'val': 1, 'test': 1}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.collect_all_data()
        if self.args.in_dataset_augmentation and self.set_type == 0:
            self.data_augmentation()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]

        if self.args.test_time_train:
            num_train = int(len(df_raw) * 0.9)
            border1s = [0, num_train - self.seq_len, len(df_raw)]
            border2s = [num_train, len(df_raw), len(df_raw)]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def regenerate_augmentation_data(self):
        self.collect_all_data()
        self.data_augmentation()

    def reload_data(self, x_data, y_data, x_time, y_time):
        self.x_data = x_data
        self.y_data = y_data
        self.x_time = x_time
        self.y_time = y_time

    def collect_all_data(self):
        self.x_data = []
        self.y_data = []
        self.x_time = []
        self.y_time = []
        data_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        mask_data_len = int((1 - self.args.data_size) * data_len) if self.args.data_size < 1 else 0
        for i in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
            if (self.set_type == 0 and i >= mask_data_len) or self.set_type != 0:
                s_begin = i
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                self.x_data.append(self.data_x[s_begin:s_end])
                self.y_data.append(self.data_y[r_begin:r_end])
                self.x_time.append(self.data_stamp[s_begin:s_end])
                self.y_time.append(self.data_stamp[r_begin:r_end])

    def data_augmentation(self):
        origin_len = len(self.x_data)
        if not self.args.closer_data_aug_more:
            aug_size = [self.args.aug_data_size for i in range(origin_len)]
        else:
            aug_size = [int(self.args.aug_data_size * i / origin_len) + 1 for i in range(origin_len)]

        for i in range(origin_len):
            for _ in range(aug_size[i]):
                aug = augmentation('dataset')
                if self.args.aug_method == 'f_mask':
                    x, y = aug.freq_dropout(self.x_data[i], self.y_data[i], dropout_rate=self.args.aug_rate)
                elif self.args.aug_method == 'f_mix':
                    rand = float(np.random.random(1))
                    i2 = int(rand * len(self.x_data))
                    x, y = aug.freq_mix(self.x_data[i], self.y_data[i], self.x_data[i2], self.y_data[i2],
                                        dropout_rate=self.args.aug_rate)
                else:
                    raise ValueError
                self.x_data.append(x)
                self.y_data.append(y)
                self.x_time.append(self.x_time[i])
                self.y_time.append(self.y_time[i])

    def __getitem__(self, index):
        seq_x = self.x_data[index]
        seq_y = self.y_data[index]
        return seq_x, seq_y, self.x_time[index], self.y_time[index]

    def __len__(self):
        return len(self.x_data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)