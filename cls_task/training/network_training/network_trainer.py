from cls_task.network_architecture.resnet import *
import torch
from cls_task.training.learning_rate.poly_lr import PolyLR
from cls_task.training.learning_rate.linear_lr import LinearLR


class NetworkTrainer:
    def __init__(self, cfg, writer, logger, visual, log_save_dir):
        self.cfg = cfg
        self.writer = writer
        self.logger = logger
        self.visual = visual
        self.log_save_dir = log_save_dir

        self.optimzier_cls = None
        self.epoch_outputs = None
        self.epoch_labels = None
        self.epoch_loss = None

        self.inputs = None
        self.cls_labels = None

        self.cls_net = None

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
            self.optimzier_cls = torch.optim.Adam(parameters, lr=lr, betas=betas, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimzier_cls = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            self.optimzier_cls = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

        self.optimizers = [self.optimzier_cls]

    def init_lr_schedulers(self, epoch_batches):
        n_epochs = self.config['training']['n_epochs']

        if self.cfg['model']['lr_scheduler'] == 'poly':
            self.lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR(self.optimizers_cls,
                                                                      lr_lambda=PolyLR(n_epochs * epoch_batches,
                                                                                       0, ).step)
        elif self.cfg['model']['lr_scheduler'] == 'linear':
            self.lr_scheduler_cls = torch.optim.lr_scheduler.LambdaLR(self.optimzier_cls,
                                                                      lr_lambda=LinearLR(n_epochs * epoch_batches,
                                                                                         0).step)
        self.lr_schedulers = [self.lr_scheduler_cls]
