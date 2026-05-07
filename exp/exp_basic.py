import os
import torch
from models import TimeMixer, Crossformer, PatchTST, iTransformer, MSPFormer, DLinear, GLinear, FITS, TiDE, TSMixer, \
    PAttn, TSLANet
from models.ModelWrapper import AMTWrapper, MlpWrapperLevel3, MlpWrapperLevel1, MlpWrapperLevel2


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimeMixer': TimeMixer,
            'Crossformer': Crossformer,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'MSPFormer': MSPFormer,
            'DLinear': DLinear,
            'GLinear': GLinear,
            'TSLANet': TSLANet,
            'FITS': FITS,
            'TiDE': TiDE,
            'TSMixer': TSMixer,
            'PAttn': PAttn
        }
        self.wrapper_dict = {
            'AMT': AMTWrapper,
            'Mlp3': MlpWrapperLevel3,
            'Mlp2': MlpWrapperLevel2,
            'Mlp1': MlpWrapperLevel1,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            import platform
            if platform.system() == 'Darwin':
                device = torch.device('mps')
                print('Use MPS')
                return device
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(self.args.device)
            if self.args.use_multi_gpu:
                print('Use GPU: cuda{}'.format(self.args.device_ids))
            else:
                print('Use GPU: {}'.format(self.args.device))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
