import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from parameter import *
from trainer import Trainer
from tester import Tester
from data_loader import Data_Loader
import torch
from torch.backends import cudnn
from utils import *


def main(config):
    if config.mode=='train':
    # Create directories if not exist
        make_folder(config.model_save_path, config.version)
        make_folder(config.sample_path, config.version)
        make_folder(config.log_path, config.version)

        if config.over_sampler:
            rare_list=get_rare_list(config.img_path, config.label_path, target_classes=[3, 14, 15, 16])
        else:
            rare_list = None

        data_loader = Data_Loader(config.img_path, config.label_path, config.sample_path, config.imsize, config.batch_size, rare_list ,config.mode)
        trainer = Trainer(data_loader.loader(), config)
        trainer.train()

    else:
        tester = Tester(config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    print(config)
    main(config)
