# get resnet models specified in cfg.yml

from .resnet_batchnorm import get_resnet_bn_model
from .resnet_instancenorm import get_resnet_in_model

def get_resnet_model(arch, **kwargs):
    arch_name = arch[:-3]  # res10_bn or res10_in
    is_instancenorm = True if arch[-2:] == "in" else False
    if is_instancenorm:
        return get_resnet_in_model(arch_name, **kwargs)
    else:
        return get_resnet_bn_model(arch_name, **kwargs)