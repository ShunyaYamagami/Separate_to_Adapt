import os
from os.path import join as ospjoin
import pandas as pd
import yaml
import torch
import random
import numpy as np
from datetime import datetime
import logging

def set_determinism(seed: int=1, benchmark: bool=False, determinism: bool=True):
    assert sum([benchmark, determinism]) <= 1, "You can only set one of benchmark or determinism to True"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = benchmark

    if determinism:
        if hasattr(torch, "set_deterministic"):
            torch.set_deterministic(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def set_logger(save_dir):
    LOG_DEST = "log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, LOG_DEST)),
            logging.StreamHandler()
        ])


class Config:
    def __init__(self):
        self.cuda_visible_devices = list(map(int, os.environ.get('CUDA_VISIBLE_DEVICES').split(",")))
        self.exec_num = int(os.environ.get('exec_num'))
        
        # set path
        self.parent = 'Office31'
        self.task = 'true_domains'
        self.dset = 'dslr_webcam'
        self.nas_root = f'/nas/data/syamagami/GDA/data/{self.parent}/imgs'
        self.txt_root = f'data/{self.parent}/{self.task}/{self.dset}'
        self.labeled_txt_path = os.path.join(self.txt_root, 'labeled.txt')
        self.unlabeled_txt_path = os.path.join(self.txt_root, 'unlabeled.txt')
        self.test_txt_path = os.path.join(self.txt_root, 'test.txt')
        self.resnet_pretrained_path = 'pretrained_models/resnet50_pretrained.pth'

        self.num_known_classes = pd.read_csv(self.labeled_txt_path, sep=' ', header=None)[1].nunique()
        self.num_classes = self.num_known_classes + 1
        self.num_all_classes = 31
        
        # set params
        self.batch_size = 32

        self.log_dir = self._set_log_dir()
        self._save_config()
        set_logger(self.log_dir)
        set_determinism()

    def _set_log_dir(self):
        dirname = datetime.now().strftime("%y%m%d_%H:%M:%S") + f'--c{self.cuda_visible_devices[0]}n{self.exec_num}' + f'--{self.task}' + f'--{self.dset}'
        dirname += f'--bs{self.batch_size}'
        dirname += f'--{self.task}'
        log_dir = os.path.join("log", self.parent, dirname)
        os.makedirs(log_dir, exist_ok=True)

        return log_dir
    
    def _save_config(self):
        instance_attributes = {k: v for k, v in self.__dict__.items() if not k.startswith('__') and not callable(v)}
        with open(ospjoin(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(instance_attributes, f, default_flow_style=False)


if __name__ == '__main__':
    config = Config()
    print(config.num_classes)