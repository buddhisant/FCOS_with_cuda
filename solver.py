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
        self.lr_decay_time=cfg.lr_decay_time
        self.num_warmup_iters=cfg.num_warmup_iters

    def lr_decay(self, epoch):
        new_lr = [self.compute_lr_by_epoch(lr,epoch) for lr in self.init_lr]
        self.set_lr(new_lr)

    def constant_warmup(self,epoch,iteration):
        if(iteration-1 < self.num_warmup_iters) and epoch==1:
            new_lr = [lr* self.warmup_factor for lr in self.init_lr]
            self.set_lr(new_lr)
        elif(iteration-1 == self.num_warmup_iters) and epoch==1:
            self.set_lr(self.init_lr)

    def compute_lr_by_epoch(self, lr, epoch):
        lr_decay_time=np.array(self.lr_decay_time,dtype=np.int)
        index=np.nonzero(lr_decay_time<=epoch)[0]
        if(index.size==0):
            return lr
        num=index[-1].item()+1
        return lr*(self.lr_decay_factor**num)

    def set_lr(self, lrs):
        for params_group,new_lr in zip(self.optimizer.param_groups,lrs):
            params_group["lr"] = new_lr
