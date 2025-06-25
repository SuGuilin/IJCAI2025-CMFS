import os
import cv2
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

from utils.pyt_utils import ensure_dir, parse_devices
from utils.visualize import print_iou
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.model import CMFS
from dataloader.dataloader import ValPre

logger = get_logger()

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        pred = self.sliding_eval_rgbX(img, modal_x, self.eval_crop_size, self.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.class_num, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)

            fn = name + '.png'
            fn_real = name + '_real.png'

            # save colored result
            class_colors = self.config.pattale
            temp = np.zeros((pred.shape[0], pred.shape[1], 3))
            ground_truth = np.zeros((pred.shape[0], pred.shape[1], 3))

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # 白色文本
            thickness = 2
            line_type = cv2.LINE_AA

            for i in range(self.class_num):
                temp[pred == i] = class_colors[i]
                ground_truth[label == i] = class_colors[i]
            
            cv2.imwrite(os.path.join(self.save_path, fn_real), temp)
            cv2.putText(temp, 'Prediction', (10, 30), font, font_scale, font_color, thickness, line_type)
            cv2.putText(ground_truth, 'Ground Truth', (10, 30), font, font_scale, font_color, thickness, line_type)

            result = np.hstack((temp, ground_truth))
            cv2.imwrite(os.path.join(self.save_path, fn), result)
            

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.class_num, self.class_num))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc, class_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_acc,
                                dataset.class_names, show_no_back=False)
        return result_line

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
    args.save_path = f"./results/{dataset_name}"
    args.log_dir = f"./experiment/{dataset_name}/exp1"
    args.checkpoint_dir = os.path.join(args.log_dir, "checkpoint")
    exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    args.log_file = os.path.join(args.log_dir, f"val_{exp_time}.log")
    args.link_log_file = os.path.join(args.log_dir, "val_last.log")

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
        segmentor = SegEvaluator(config, dataset, network, all_dev, config.verbose, args.save_path)
        segmentor.run(args.checkpoint_dir, args.epochs, args.log_file, args.link_log_file)