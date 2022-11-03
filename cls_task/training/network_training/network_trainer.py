import torch
import numpy as np
from collections import OrderedDict

from cls_task.network_architecture.resnet import *
from cls_task.training.learning_rate.poly_lr import PolyLR
from cls_task.training.learning_rate.linear_lr import LinearLR
from cls_task.evaluation.metrics import metric, multi_cls_roc_auc_score


class NetworkTrainer:
    def __init__(self, cfg, writer, logger, visual, log_save_dir):
        self.optimizers = None
        self.optimizers_cls = None
        self.lr_scheduler_cls = None
        self.lr_schedulers = None
        self.epoch_outputs = None
        self.epoch_labels = None
        self.epoch_loss = None
        self.inputs = None
        self.cls_labels = None
        self.cls_net = None
        self.nets = None

        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.visual = visual
        self.log_save_dir = log_save_dir

        self.iter = 0
        self.ce_loss = nn.CrossEntropyLoss()

    def reset_epoch_params(self):
        self.epoch_outputs = []
        self.epoch_labels = []
        self.epoch_loss = []

    def set_input(self, inputs, cls_labels, train=False):
        self.inputs = inputs.cuda()
        self.cls_labels = cls_labels.cuda()

    def cls_forward(self, inputs):
        out_dict = self.cls_net(inputs)
        return out_dict

    def cls_model_backward(self, train=False):
        log_losses = dict()
        flatten_feats, outputs = self.cls_forward(self.inputs)
        loss = self.ce_loss(outputs, self.cls_labels)
        if train:
            log_losses['cls_loss/loss_ce'] = loss.detach()
            self.visual.plot_current_errors(log_losses, self.iter)
        self.epoch_loss.append(loss.item())
        output_g = outputs
        target_g = self.cls_labels
        self.epoch_outputs.append(torch.softmax(output_g, dim=1).detach().data.cpu())
        self.epoch_labels.append(target_g.data.cpu())

        return loss

    def lr_scheduler_step(self):
        for lr_s in self.lr_schedulers:
            lr_s.step()

    def optimizers_zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()

    def optimizers_step(self):
        for optim in self.optimizers:
            optim.step()

    def init_nets(self):
        self.cls_net = generate_model(18)

    def init_device(self):
        self.cls_net = self.cls_net.to('cuda')
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
            self.optimizers_cls = torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizers_cls = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.optimizers_cls = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.optimizers = [self.optimzier_cls]

    def init_lr_schedulers(self, epoch_batches):
        n_epochs = self.cfg['training']['n_epochs']

        if self.cfg['model']['lr_scheduler'] == 'poly':
            self.lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR(self.optimizers_cls,
                                                                      lr_lambda=PolyLR(n_epochs * epoch_batches,
                                                                                       0, ).step)
        elif self.cfg['model']['lr_scheduler'] == 'linear':
            self.lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR(self.optimzier_cls,
                                                                      lr_lambda=LinearLR(n_epochs * epoch_batches,
                                                                                         0).step)
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
        visual_list += {('b1', self.inputs[:, 0, 10, :, :].cpu().unsqueeze(1))}
        self.visual.display_current_results(OrderedDict(visual_list), tag, self.iter)

    def train_step(self):
        self.optimizers_zero_grad()
        loss = self.cls_model_backward(train=True)
        loss.backward()
        self.optimizers_step()
        self.lr_scheduler_step()

        return loss.detach()

    def train_metric_record(self, epoch, epochs):
        self.visualization('train')
        epoch_outputs = torch.cat(self.epoch_outputs).numpy()
        epoch_labels = torch.cat(self.epoch_labels).numpy()
        if np.isnan(epoch_outputs).sum() > 0 or np.isinf(epoch_outputs).sum() > 0:
            self.logger.info('Epoch [{}/{}], train auc: NAN/INF, acc: NAN/INF, loss: NAN/INF'.format(epoch, epochs))
        else:
            train_metric = metric(epoch_outputs, epoch_labels)
            train_auc_list = multi_cls_roc_auc_score(epoch_outputs, epoch_labels)
            self.logger.info('Epoch [{}/{}], train auc: {:.3f}, acc: {:.3f}, loss: {:.3f}'.format(epoch, epochs, np.mean(train_auc_list), train_metric['acc'], np.mean(self.epoch_loss)))
            self.writer.add_scalar('train_avg/auc', np.mean(train_auc_list), epoch)
            self.writer.add_scalar('train_avg/acc', train_metric['acc'], epoch)
            self.writer.add_scalar('train_avg/auc', np.mean(self.epoch_loss), epoch)

