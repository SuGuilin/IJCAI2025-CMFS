import os
import argparse
import time
import cv2
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, parse_devices
from engine.evaluator_vision import Evaluator
from kornia.metrics import AverageMeter
from engine.logger import get_logger
from dataloader.RGBXDataset import RGBXDataset
from models.model import CMFS
from dataloader.dataloader import ValPre
from PIL import Image

logger = get_logger()
Fuse_result = [AverageMeter() for _ in range(8)]

class VisionEvaluator(Evaluator):

    def func_per_iteration_vision(self, data, device):
        img = data['data']
        modal_x = data['modal_x']
        name = data['fn']
        Fuse = self.sliding_eval_rgbX_vision(img, modal_x, self.eval_crop_size, self.eval_stride_rate, device)

        Fuse = (Fuse - Fuse.min()) / (Fuse.max() - Fuse.min()) * 255.0
        
        if self.save_path is not None:
            ensure_dir(self.save_path)
            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(Fuse.astype(np.uint8), mode='RGB')
            result_img.save(os.path.join(self.save_path, fn))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='MRFS', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('--dataset_name', '-n', default='mfnet', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--save_path', '-p', default=f"./results/")

    args = parser.parse_args()
    all_dev = parse_devices(args.devices)
    dataset_name = args.dataset_name
    args.save_path = f"./results_fusion/{dataset_name}"
    args.log_dir = f"./experiment/{dataset_name}/exp1"
    args.checkpoint_dir = os.path.join(args.log_dir, "checkpoint")

    import yaml
    from easydict import EasyDict as edict

    def load_config(yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return edict(config)
    
    if dataset_name == 'MFNet':
        config_path = './configs/config_mfnet.yaml'
    elif dataset_name == 'FMB':
        config_path = './configs/config_fmb.yaml'
    elif dataset_name == 'PST900':
        config_path = './configs/config_pst900.yaml'
    else:
        raise ValueError('Not a valid dataset name')

    config = load_config(config_path)

    network = CMFS(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'rgb_root': os.path.join(config.dataset_path, config.rgb_folder),
                    'rgb_format': config.rgb_format,
                    'x_root': os.path.join(config.dataset_path, config.x_folder),
                    'x_format': config.x_format,
                    'x_single_channel': config.x_is_single_channel,
                    'gt_root': os.path.join(config.dataset_path, config.label_folder),
                    'gt_format': config.label_format,
                    'transform_gt': config.gt_transform,
                    'class_names': config.class_names,
                    'train_source': os.path.join(config.dataset_path, "train.txt"),
                    'eval_source': os.path.join(config.dataset_path, "test.txt")}
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, 'val', val_pre)

    with torch.no_grad():
        Fuser = VisionEvaluator(config, dataset, network, all_dev, config.verbose, args.save_path)
        Fuser.run(args.checkpoint_dir, args.epochs)