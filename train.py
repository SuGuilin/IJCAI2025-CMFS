import argparse
import os
import sys
import time
import warnings

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from dataloader.dataloader import get_train_loader, get_val_loader
from dataloader.RGBXDataset import RGBXDataset
from engine.engine import Engine
from engine.logger import get_logger
from models.model import CMFS, group_weight
from utils.loss_utils import MultiTaskLoss
from utils.lr_policy import WarmUpPolyLR
from utils.metric import compute_metric, hist_info
from utils.pyt_utils import all_reduce_tensor, reduce_value

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
logger = get_logger()
    
with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    dataset_name = args.dataset_name

    if dataset_name == 'mfnet':
        config_path = './configs/config_mfnet.yaml'
    elif dataset_name == 'fmb':
        config_path = './configs/config_fmb.yaml'
    elif dataset_name == 'pst900':
        config_path = './configs/config_pst900.yaml'
    else:
        config_path = './configs/base_config.yaml'
        # raise ValueError('Not a valid dataset name')

    config = engine.load_config(config_path)
    cudnn.benchmark = True
    seed = config.seed
    niters_per_epoch = config.num_train_imgs // config.batch_size + 1

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(config, engine, RGBXDataset)
    val_loader, val_sampler = get_val_loader(config, engine, RGBXDataset)

    if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
        tb_dir = config.tb_dir + '/{}'.format(exp_time)
        tb = SummaryWriter(log_dir=tb_dir)

    # config network and criterion
    criterion = MultiTaskLoss(config.background, cfg=config)

    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d
    
    model = CMFS(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    
    # group weight and config optimizer
    base_lr = config.lr
    
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
    
    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                            output_device=engine.local_rank, find_unused_parameters=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin trainning:')
    best_mIoU = 0.
    for epoch in range(engine.state.epoch, config.nepochs+1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss = 0
        sum_loss_seg = 0
        sum_loss_fus = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs = minibatch['data']
            gts = minibatch['label']
            modal_xs = minibatch['modal_x']
            Mask = minibatch['Mask']

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)
            Mask = Mask.cuda(non_blocking=True)

            loss, seg_loss, fus_loss = model(imgs, modal_xs, Mask, gts)
            # reduce the whole loss over multi-gpu
            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                semantic_loss = all_reduce_tensor(seg_loss, world_size=engine.world_size)
                fusion_loss = all_reduce_tensor(fus_loss, world_size=engine.world_size)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch- 1) * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)

            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            if engine.distributed:
                sum_loss += reduce_loss.item()
                sum_loss_seg += semantic_loss.item()
                sum_loss_fus += fusion_loss.item()
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1))) \
                            + ' loss_seg=%.4f total_loss_seg=%.4f' % (semantic_loss.item(), (sum_loss_seg / (idx + 1))) \
                            + ' loss_fusion=%.4f total_loss_fusion=%.4f' % (fusion_loss.item(), (sum_loss_fus / (idx + 1)))
            else:
                sum_loss += loss
                sum_loss_seg += seg_loss
                sum_loss_fus += fus_loss
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Iter {}/{}:'.format(idx + 1, niters_per_epoch) \
                            + ' lr=%.4e' % lr \
                            + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1))) \
                            + ' loss_seg=%.4f total_loss_seg=%.4f' % (seg_loss, (sum_loss_seg / (idx + 1))) \
                            + ' loss_fusion=%.4f total_loss_fusion=%.4f' % (fus_loss, (sum_loss_fus / (idx + 1)))

            del loss, seg_loss, fus_loss
            pbar.set_description(print_str, refresh=False)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)
            tb.add_scalar('fusion_loss', sum_loss_fus / len(pbar), epoch)
            tb.add_scalar('semantic_loss', sum_loss_seg / len(pbar), epoch)

        if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.checkpoint_dir)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir)

            ### val
            if engine.distributed and (engine.local_rank == 0) or (not engine.distributed):
                logger.info('########Validation########')
            all_result = []
            if engine.distributed:
                val_sampler.set_epoch(epoch)
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(len(val_loader)), file=sys.stdout, bar_format=bar_format)
            dataloader = iter(val_loader)
            model.eval()
            for idx in pbar:
                print_str = 'validation {}/{}'.format(idx + 1, len(val_loader))
                pbar.set_description(print_str, refresh=False)

                data = next(dataloader)
                img = data['data']
                label = data['label']
                modal_x = data['modal_x']
                name = data['fn']

                img = img.cuda(non_blocking=True)
                modal_x = modal_x.cuda(non_blocking=True)

                
                with torch.no_grad():
                    score, Fus_img = model(img, modal_x)
                    score = torch.exp(score[0])
                    score = score.permute(1, 2, 0)
                    processed_pred = cv2.resize(score.cpu().numpy(), (config.eval_crop_size[1], config.eval_crop_size[0]), interpolation=cv2.INTER_LINEAR)
                    pred = processed_pred.argmax(2)
                
                result = hist_info(config.num_classes, pred, label.squeeze().numpy())[0]
                all_result.append(result)
            model.train()
            iou, mIoU = compute_metric(all_result, config.num_classes)
            iou, mIoU = iou if not engine.distributed else reduce_value(iou, average=True),\
                mIoU if not engine.distributed else reduce_value(mIoU, average=True)

            if engine.distributed and (engine.local_rank == 0) or (not engine.distributed):
                result_line = [f"{config.class_names[i]:8s}: \t {iou[i]*100:.3f}% \n" for i in range(config.num_classes)]
                result_line.append(f"mean IoU: \t {mIoU[0]*100:.3f}% \n")

                results = open(config.log_file, 'a')
                results.write(f"##epoch:{epoch:4d} " + "#" * 67 + "\n")
                print("#" * 80 + "\n")
                for line in result_line:
                    print(line)
                    results.write(line)
                results.write("#" * 80 + "\n")
                results.flush()
                results.close()
                if mIoU[0] > best_mIoU:
                    best_mIoU = mIoU[0]
                    engine.save_checkpoint(os.path.join(config.checkpoint_dir, "best.pth"))



