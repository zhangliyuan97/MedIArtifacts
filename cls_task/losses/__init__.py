from .focal_loss import FocalLossV2
from torch.nn import BCEWithLogitsLoss

def get_loss(loss_name, **kwargs):
    if loss_name == "bce":
        return BCEWithLogitsLoss()
    elif loss_name == "focal":
        return FocalLossV2(**kwargs)
    else:
        raise Exception("Loss name of {} not implemented!!".format(loss_name))