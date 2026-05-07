from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
}
def data_provider(args, flag):
    flag = args.data + '_' + flag
    print('data_provider flag:', flag)
    if flag == 'ETTH1_train':
        data_set = Dataset_ETT_hour(
            args=args,
            root_path=args.root_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTh1.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTH1_val':
        data_set = Dataset_ETT_hour(
            args=args,
            root_path=args.root_path,
            flag="val",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTh1.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTH1_test':
        data_set = Dataset_ETT_hour(
            args=args,
            root_path=args.root_path,
            flag="test",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTh1.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTH2_train':
        data_set = Dataset_ETT_hour(
            args=args,
            root_path=args.root_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTh2.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTH2_val':
        data_set = Dataset_ETT_hour(
            args=args,
            root_path=args.root_path,
            flag="val",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTh2.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTH2_test':
        data_set = Dataset_ETT_hour(
            args=args,
            root_path=args.root_path,
            flag="test",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTh2.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTM1_train':
        data_set = Dataset_ETT_minute(
            root_path=args.root_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTm1.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='t',
            test_time_train=args.test_time_train
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTM1_val':
        data_set = Dataset_ETT_minute(
            root_path=args.root_path,
            flag="val",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTm1.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='t',
            test_time_train=args.test_time_train
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTM1_test':
        data_set = Dataset_ETT_minute(
            root_path=args.root_path,
            flag="test",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTm1.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='t',
            test_time_train=args.test_time_train
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTM2_train':
        data_set = Dataset_ETT_minute(
            root_path=args.root_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTm2.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='t',
            test_time_train=args.test_time_train
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTM2_val':
        data_set = Dataset_ETT_minute(
            root_path=args.root_path,
            flag="val",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTm2.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='t',
            test_time_train=args.test_time_train
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'ETTM2_test':
        data_set = Dataset_ETT_minute(
            root_path=args.root_path,
            flag="test",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="ETTm2.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='t',
            test_time_train=args.test_time_train
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'traffic_train':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="traffic.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'traffic_val':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="val",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="traffic.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'traffic_test':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="test",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="traffic.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'weather_train':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="weather.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'weather_val':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="val",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="weather.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'weather_test':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="test",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="weather.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'electricity_train':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="electricity.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'electricity_val':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="val",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="electricity.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    elif flag == 'electricity_test':
        data_set = Dataset_Custom(
            args=args,
            root_path=args.root_path,
            flag="test",
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            data_path="electricity.csv",
            target="OT",
            scale=args.scale,
            timeenc=0,
            freq='h'
        )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers)
    else:
        raise ValueError('datafactory Invalid flag')
    return data_set, data_loader
