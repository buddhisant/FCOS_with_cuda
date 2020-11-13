import os
import time
import utils
import torch
import argparse
import dataset
import transform
import config as cfg
import torch.distributed as dist

import solver

from sampler import groupSampler
from sampler import distributedGroupSampler
from fcos import FCOS
from dataloader import build_dataloader

pretrained_path={
    50:"./pretrained/resnet50_caffe.pth",
    101:"./pretrained/resnet101_caffe.pth"
}

def train(is_dist, local_rank):
    transforms=transform.build_transforms()
    coco_dataset = dataset.COCODataset(is_train=True, transforms=transforms)
    if(is_dist):
        sampler = distributedGroupSampler(coco_dataset)
    else:
        sampler = groupSampler(coco_dataset)
    dataloader = build_dataloader(coco_dataset, sampler)

    batch_time_meter = utils.AverageMeter()
    cls_loss_meter = utils.AverageMeter()
    reg_loss_meter = utils.AverageMeter()
    cen_loss_meter = utils.AverageMeter()
    losses_meter = utils.AverageMeter()

    model = FCOS(is_train=True)
    model.resNet.load_pretrained(pretrained_path[cfg.resnet_depth])
    model=model.cuda()

    if is_dist:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank,],output_device=local_rank,broadcast_buffers=False)
    optimizer=solver.build_optimizer(model)
    scheduler=solver.scheduler(optimizer)

    model.train()

    for epoch in range(1, cfg.max_epochs + 1):
        if is_dist:
            dataloader.sampler.set_epoch(epoch)

        lr_decay_time = cfg.lr_decay_time
        if(epoch in lr_decay_time):
            scheduler.lr_decay()

        end_time = time.time()
        for iteration, datas in enumerate(dataloader, 1):
            images = datas["images"]
            bboxes = datas["bboxes"]
            labels = datas["labels"]

            images = images.cuda()
            bboxes = [bbox.cuda() for bbox in bboxes]
            labels = [label.cuda() for label in labels]

            if(epoch==1) and (iteration < cfg.num_warmup_iters):
                scheduler.start_warmup()
            elif(epoch==1) and (iteration == cfg.num_warmup_iters):
                scheduler.end_warmup()
            loss_dict = model([images, bboxes, labels])
            cls_loss = loss_dict["cls_loss"]
            reg_loss = loss_dict["reg_loss"]
            cen_loss = loss_dict["cen_loss"]

            losses = cls_loss + reg_loss + cen_loss
            optimizer.zero_grad()
            losses.backward()
            utils.clip_grads(model.parameters())
            optimizer.step()

            batch_time_meter.update(time.time()-end_time)
            end_time = time.time()

            cls_loss_meter.update(cls_loss.item())
            reg_loss_meter.update(reg_loss.item())
            cen_loss_meter.update(cen_loss.item())
            losses_meter.update(losses.item())

            if(iteration % 50 == 0):
                if(local_rank == 0):
                    res = "\t".join([
                        "Epoch: [%d/%d]" % (epoch,cfg.max_epochs),
                        "Iter: [%d/%d]" % (iteration, len(dataloader)),
                        "Time: %.3f (%.3f)" % (batch_time_meter.val, batch_time_meter.avg),
                        "Cls_loss: %.4f (%.4f)" % (cls_loss_meter.val, cls_loss_meter.avg),
                        "Reg_loss: %.4f (%.4f)" % (reg_loss_meter.val, reg_loss_meter.avg),
                        "Cen_loss: %.4f (%.4f)" % (cen_loss_meter.val, cen_loss_meter.avg),
                        "Loss: %.4f (%.4f)" % (losses_meter.val, losses_meter.avg),
                        "lr: %.6f" % (optimizer.param_groups[0]["lr"])
                    ])
                    print(res)

                batch_time_meter.reset()
                cls_loss_meter.reset()
                reg_loss_meter.reset()
                cen_loss_meter.reset()
                losses_meter.reset()

        utils.save_model(model, epoch)

def main():
    # os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser=argparse.ArgumentParser(description="FCOS")
    parser.add_argument("--local_rank", type=int, default=0)
    gpu_nums=torch.cuda.device_count()
    is_dist = gpu_nums>1

    args=parser.parse_args()
    if(is_dist):
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        utils.synchronize()

    train(is_dist, args.local_rank)

if __name__=="__main__":
    main()