import torch
import config as cfg

def build_optimizer(model):
    """构建优化器, 默认的学习率分配规则为
        1, 对于fpn和head部分的bias参数，学习率为基本学习率*2
        2, 对于其他参数，学习率为基本学习率
    """
    if(hasattr(model,"module")):
        model=model.module

    double_lr_param=[]
    non_double_lr_param=[]
    params=[]
    for name, param in model.named_parameters():
        if("resNet" in name):
            non_double_lr_param.append(param)
            continue
        if("bias" in name):
            double_lr_param.append(param)
        else:
            non_double_lr_param.append(param)
    params.append({"params":non_double_lr_param})
    params.append({"params":double_lr_param,"lr":cfg.base_lr*cfg.bias_lr_factor,"weight_decay":cfg.weight_decay*cfg.bias_wd_factor})
    optimizer=torch.optim.SGD(params,lr=cfg.base_lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay)
    return optimizer

class scheduler():
    def __init__(self, optimizer):
        self.init_lr=[group["lr"] for group in optimizer.param_groups]
        self.optimizer=optimizer
        self.warmup_factor=cfg.warmup_factor
        self.lr_decay_factor = cfg.lr_decay_factor

    def start_warmup(self):
        for groups, init_lr in zip(self.optimizer.param_groups, self.init_lr):
            groups["lr"] = init_lr * self.warmup_factor

    def end_warmup(self):
        for groups, init_lr in zip(self.optimizer.param_groups, self.init_lr):
            groups["lr"] = init_lr

    def lr_decay(self):
        for groups in self.optimizer.param_groups:
            groups["lr"] = groups["lr"] * self.lr_decay_factor
