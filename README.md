# smartPytorch

smartPytorch是一个通用化的基于Pytorch的训练平台，希望整合目前辅助驾驶中的一些常用的检测，分割等一些任务，未来还将整合压缩，量化，追踪等一些功能。

目前完成了:
    1）数据输入，
    2）多层不同的学习率（1.freeze bn的mean和var固定，2.finetune 固定前向的特征提取层，3.diffRate 特征提取层和分类层使用不同的学习率），
    3）多gpu学习，
    4）不同的optimizer方法，
    5）不同的学习策略（1.step，2.multistep，3.ploy，4.warm-up），
    6）不同的特征提取层以及预训练模型（vgg，resnet，densenet，inception-v3，googlenet，mobilenet，shufflenet，mnasnet），
    7）检测方法（yolov3和ssd），分割方法（pspnet和mtsd）。
