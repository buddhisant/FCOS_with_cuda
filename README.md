# FCOS_with_cuda
基于pytorch的标准版FCOS，支持多卡分布式训练，nms采用了基于pytorch的extension(该部分代码来源于mmdetection)，需编译安装cuda extension。

**代码中参考了[maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)和[mmdetection](https://github.com/open-mmlab/mmdetection)**

**更多目标检测代码请参见[友好型的object detection代码实现和论文解读](https://blog.csdn.net/gongyi_yf/article/details/109660890)**

**backbone网络基于resnet50**

**请确保已经安装pycocotools以及1.1.0版本以上的pytorch**

# 性能

# 使用方法：
- git clone https://github.com/buddhisant/FCOS_with_cuda.git
- cd FCOS_with_cuda
- mkdir pretrained
- cd pretrained
- wget https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth -O resnet50_caffe.pth
- wget https://download.openmmlab.com/pretrain/third_party/resnet101_caffe-3ad79236.pth -O resnet101_caffe.pth
- cd ..
- mkdir data
- cd data
- mkdir coco
- cd ../..
- python setup.py build_ext develop

将coco数据集放在./data/coco下面

# 训练
如果你的设备有多个显卡，请使用分布式训练，例如你的设备有2个显卡，请采用命令
- python -m torch.distributed.launch --nproc_per_node=2 --master_port=$((RANDOM+10000)) train.py --dist

这里的dist参数表示开启分布式训练

如果你的设备只有一个显卡，请采用普通的训练，请采用命令
- python train.py

# 测试
测试只支持单卡,请采用命令
- python test.py

# 当你不再使用该程序时
可以使用如下命令，恢复你的python环境
- pip uninstall fcos-cuda -y

如有技术问题可联系qq 1401997998
