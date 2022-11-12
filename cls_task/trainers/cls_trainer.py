import os
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

from models.resnet import get_resnet_model
from trainers.training_utils.lr_shcedules import get_lr_schedule
from losses import get_loss
from utils import metrics


class NetworkTrainer:
    def __init__(self, cfg, writer, logger, visual, log_save_dir):
        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.visual = visual
        self.log_save_dir = log_save_dir

        self.init_nets()
        self.init_device()
        self.init_optimizers()

        self.epoch_outputs = None
        self.epoch_labels = None
        self.epoch_loss = None

        self.iter = 0
        self.loss = get_loss(loss_name=self.cfg['loss']['name'], alpha=self.cfg['loss']['alpha'], gamma=self.cfg['loss']['gamma'])
        # self.ce_loss = nn.CrossEntropyLoss()   # for multi-class classification problems
        # self.bce = nn.BCEWithLogitsLoss()      # for logistic regression problems
        # self.focal_loss = FocalLossV2(alpha=self.cfg['loss']['alpha'], gamma=self.cfg['loss']['gamma'])

    def reset_epoch_records(self):
        self.epoch_outputs = []
        self.epoch_labels = []
        self.epoch_loss = []

    def set_input(self, inputs, cls_labels, train=False):
        self.inputs = inputs.cuda()
        self.cls_labels = cls_labels.cuda()

    def cls_net_forward(self, inputs):
        outputs = self.cls_net(inputs)
        return {
            "outputs": outputs,
        }
    
    @torch.no_grad()
    def cls_evaluation(self):     # evaluate a mini-batch 
        out_dict = self.cls_net_forward(self.inputs)
        loss = self.loss(out_dict["outputs"], self.cls_labels.float())

        self.epoch_loss.append(loss.item())
        output_g = out_dict["outputs"]
        target_g = self.cls_labels
        self.epoch_outputs.append(torch.sigmoid(output_g).detach().data.cpu())
        self.epoch_labels.append(target_g.data.cpu())

        return loss

    def cal_cls_gradients(self):  # calculate classification gradients of a mini-batch 
        log_losses = dict()
        
        out_dict = self.cls_net_forward(self.inputs)
        loss = self.loss(out_dict["outputs"], self.cls_labels.float())  # if labels not float, then BCEWithLogits will raise Exception

        log_losses['cls_loss/loss_ce'] = loss.detach()
        self.visual.plot_current_errors(log_losses, self.iter)

        self.epoch_loss.append(loss.item())
        output_g = out_dict["outputs"]
        target_g = self.cls_labels
        self.epoch_outputs.append(torch.sigmoid(output_g).detach().data.cpu())
        self.epoch_labels.append(target_g.data.cpu())

        return loss
    
    def train_step(self):
        self.optimizer_cls.zero_grad()
        loss = self.cal_cls_gradients()
        loss.backward()
        self.optimizer_cls.step()
        self.lr_scheduler_cls.step()

        return loss.detach()
    
    def eval_step(self):
        loss = self.cls_evaluation()

        return loss.detach()

    def init_nets(self):
        self.cls_net = get_resnet_model(self.cfg["model"]["arch"], in_channels=self.cfg["in_channels"], n_classes=self.cfg["n_classes"])

    def init_device(self):
        self.cls_net = self.cls_net.cuda()
        self.nets = [self.cls_net]

    def init_optimizers(self):
        optimizer_name = self.cfg['model']['optimizer']['name']
        momentum = self.cfg['model']['optimizer']['momentum']
        betas = self.cfg['model']['optimizer']['betas']
        weight_decay = self.cfg['model']['optimizer']['weight_decay']
        lr = self.cfg['model']['optimizer']['lr']

        bias_list = (param for name, param in self.cls_net.named_parameters() if name[-4:] == 'bias')
        other_list = (param for name, param in self.cls_net.named_parameters() if name[-4:] != 'bias')
        parameters = [{'params': bias_list, 'weight_decay': 0}, {'params': other_list}]

        if optimizer_name == 'adam':
            self.optimizer_cls = torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer_cls = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.optimizer_cls = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.optimizers = [self.optimizer_cls]

    def init_lr_schedulers(self, epoch_batches):
        n_epochs = self.cfg['training']['n_epochs']
        
        lr_lambda_name = self.cfg['model']['lr_scheduler']['name']
        lr_lambda_start_from = self.cfg['model']['lr_scheduler']['start_from']
        lr_schedule_lambda = get_lr_schedule(lr_lambda_name)
        
        self.lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR(self.optimizer_cls, 
                                                                  lr_lambda=lr_schedule_lambda(n_epochs * epoch_batches, lr_lambda_start_from, ).step)
        
        self.lr_schedulers = [self.lr_scheduler_cls]

    def load_pretrained_weights(self, pth):
        self.cls_net.load_state_dict(torch.load(pth))

    def train(self):
        for net in self.nets:
            net.train()

    def eval(self):
        for net in self.nets:
            net.eval()

    def visualization(self, tag='train'):
        visual_list = []
        # you should leverage a  proper window width to clip the ct_brain, [0, 100]
        visual_list += {('ct_brain', self.inputs[:, 0, 15, :, :].cpu().unsqueeze(1))}
        self.visual.display_current_results(OrderedDict(visual_list), tag, self.iter)

    def train_metric_record(self, epoch, epoches):
        self.visualization('train')
        
        epoch_outputs = torch.cat(self.epoch_outputs).numpy()
        epoch_labels = torch.cat(self.epoch_labels).numpy()
        if np.isnan(epoch_outputs).sum() > 0 or np.isinf(epoch_outputs).sum() > 0:
            self.logger.info('Epoch [{}/{}], train auc: NAN/INF, acc: NAN/INF, loss: NAN/INF'.format(epoch, epoches))
        else:
            train_acc = metrics.get_accuracy(epoch_outputs, epoch_labels)
            # train_spe = metrics.get_specifity(epoch_outputs, epoch_labels)
            # train_sen = metrics.get_sensitivity(epoch_outputs, epoch_labels)
            train_auc = metrics.get_auc(epoch_outputs, epoch_labels)

            self.logger.info('Epoch [{}/{}], Train AUC: {:.3f}, ACC: {:.3f}, Loss: {:.3f}'.format(
                epoch, epoches, train_auc, train_acc, np.mean(self.epoch_loss)))

            self.writer.add_scalar('train_avg/auc', train_auc, epoch)
            self.writer.add_scalar('train_avg/acc', train_acc, epoch)
            self.writer.add_scalar('train_avg/loss', np.mean(self.epoch_loss), epoch)
    
    def valid_metric_record(self, epoch, epoches):
        self.visualization("valid")

        epoch_outputs = torch.cat(self.epoch_outputs).numpy()
        epoch_labels = torch.cat(self.epoch_labels).numpy()

        if np.isnan(epoch_outputs).sum() > 0 or np.isinf(epoch_outputs).sum() > 0:
            self.logger.info('Epoch [{}/{}], Valid AUC: NAN/INF, ACC: NAN/INF, Loss: NAN/INF'.format(epoch, epoches))
            return 0
        
        valid_acc = metrics.get_accuracy(epoch_outputs, epoch_labels)
        valid_spe = metrics.get_specifity(epoch_outputs, epoch_labels)
        valid_sen = metrics.get_sensitivity(epoch_outputs, epoch_labels)
        valid_auc = metrics.get_auc(epoch_outputs, epoch_labels)

        self.writer.add_scalar('valid_avg/auc', valid_auc, epoch)
        self.writer.add_scalar('valid_avg/acc', valid_acc, epoch)
        self.writer.add_scalar('valid_avg/specifity', valid_spe, epoch)
        self.writer.add_scalar('valid_avg/sensitivity', valid_sen, epoch)
        self.writer.add_scalar('valid_avg/loss', np.mean(self.epoch_loss), epoch)

        self.logger.info('Epoch [{}/{}], Valid AUC: {:.3f}, ACC: {:.3f}, Specifity: {:.3f}, Sensitivity: {:.3f}, Loss: {:.3f}'.format(
                epoch, epoches, valid_auc, valid_acc, valid_spe, valid_sen, np.mean(self.epoch_loss)))
        
        return valid_auc

    def save_valid_best_model(self, model_dir):
        torch.save(self.cls_net.state_dict(), os.path.join(model_dir, "best_model.pt"))