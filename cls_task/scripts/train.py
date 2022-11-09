import os
import sys
sys.path.append(os.path.abspath('your cls_task absolute path'))
import yaml
import time
import shutil
import random
import argparse
import datetime
import itertools
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from trainers.training_utils.visualizer import Visualizer
from trainers.training_utils.logger import get_logger
from datasets import create_dataset
from trainers.cls_trainer import NetworkTrainer as HematomaExpansionTrainer


def train(cfg, writer, logger, visual, logdir):
    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(cfg.get('random_seed', 88))
    np.random.seed(cfg.get('random_seed', 88))
    torch.manual_seed(cfg.get('random_seed', 88))
    torch.random.manual_seed(cfg.get('random_seed', 88))
    torch.cuda.manual_seed(cfg.get('random_seed', 88))
    torch.cuda.manual_seed_all(cfg.get('random_seed', 88))
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    datasets = create_dataset(cfg)
    model = HematomaExpansionTrainer(cfg, writer, logger, visual, logdir)

    torch.multiprocessing.set_sharing_strategy('file_system')
    random.seed(cfg.get('random_seed', 88)) 
    np.random.seed(cfg.get('random_seed', 88))
    torch.manual_seed(cfg.get('random_seed', 88))
    torch.random.manual_seed(cfg.get('random_seed', 88))
    torch.cuda.manual_seed(cfg.get('random_seed', 88))
    torch.cuda.manual_seed_all(cfg.get('random_seed', 88))

    train_loader = datasets.train_loader
    valid_loader = datasets.valid_loader
    epoch_batches = len(train_loader)
    model.init_lr_schedulers(epoch_batches)
    logger.info('train batchsize is {}'.format(train_loader.batch_size))
    logger.info('train loader len is {}'.format(len(train_loader)))
    print('train batchsize is {}'.format(train_loader.batch_size))
    print('train loader len is {}'.format(len(train_loader)))
    logger.info('valid batchsize is {}'.format(valid_loader.batch_size))
    logger.info('valid loader len is {}'.format(len(valid_loader)))
    print('valid batchsize is {}'.format(valid_loader.batch_size))
    print('valid loader len is {}'.format(len(valid_loader)))
    cfg['training']['train_iters'] = cfg['training']['n_epochs'] * epoch_batches

    # begin training
    model.iter = 0
    prev_time = time.time()
    best_valid_auc = 0
    best_valid_loss = 99
    best_step = 0

    for epoch in range(cfg['training']['n_epochs']):
        if model.iter > cfg['training']['train_iters']:
            break

        # ===================
        # === Training ===
        # ===================

        model.reset_epoch_records()
        for train_batch in train_loader:
            i = model.iter
            if i > cfg['training']['train_iters']:
                break

            inputs = train_batch['inputs']
            cls_labels = train_batch['cls_labels']

            if (inputs != inputs).sum() > 0:
                continue

            model.set_input(inputs, cls_labels)
            model.train()
            loss_cls = model.train_step()

            batches_left = cfg['training']['n_epochs'] * epoch_batches - model.iter
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [loss: %4f] ETA: %s"
                % (
                    epoch+1,
                    cfg['training']['n_epochs'],
                    model.iter,
                    cfg['training']['n_epochs'] * epoch_batches,
                    loss_cls,
                    time_left,
                )
            )

            model.iter += 1
        
        # record train metrics
        model.train_metric_record(epoch, cfg['training']['n_epochs'])

        # ===================
        # === Evalutation ===
        # ===================

        # reset model.epoch_records for evaluation iteration
        model.reset_epoch_records()

        # evaluation
        torch.cuda.empty_cache()
        for valid_batch in valid_loader:
            inputs = valid_batch['inputs']
            cls_labels = valid_batch['cls_labels']

            model.set_input(inputs, cls_labels)
            model.eval()
            valid_loss = model.eval_step()
        
        # record valid metrics
        valid_auc = model.valid_metric_record(epoch, cfg['training']['n_epochs'])

        if best_valid_auc < valid_auc:
            best_valid_auc = valid_auc
            best_step = model.iter

            model_dir = os.path.join(logdir, "best_auc_model")
            os.makedirs(model_dir, exist_ok=True)
            model.save_valid_best_model(model_dir)

            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hematoma expansion prediction setting options")
    parser.add_argument(
        "--config", nargs="?", type=str, 
        default="configs/hematoma.yml", 
        help="Configuration for hematoma expansion prediction"
    )
    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    
    cfg['local_rank'] = args.local_rank
    torch.cuda.set_device(cfg['local_rank'])

    run_id = random.randint(1, 100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)
    visual = Visualizer(cfg, logdir, writer)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Let the games begin')

    train(cfg, writer, logger, visual, logdir)