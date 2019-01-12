import torchvision.transforms as transforms

# task selection
task = "classify"

# select architecture
cfg = "arch/test.cfg"
data_config_path = "cfg/wissenperson.data"

# data_transform
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# pretrained model
#pretrainedModel = "./weights/darknet53.pth"#"./snapshot/yolov3SPP_best_model_0.6709.pkl"#

# resume from checkpoint
resume = None#"/home/wfw/HASCO/LaneDet_2018.12.12-/smartPytotch/classify/test/checkpoint.pth.tar"#"./weights/best.pt"#"./snapshot/yolov3-SPP_best_model.pkl"#None#

max_epoch =300
display = 20

# L2 regularizer
optimizer = {0: {'optimizer': 'SGD',
             'lr': 1e-3,
             'momentum': 0.9,
             'weight_decay': 5e-4} }
             #'diffRate': [0.1, 1]} }

# learning rate policy
learning_policy = {0: {'policy': 'multistep',
                       'steps': [120, 160] }}
                   #0: {'policy': 'burn-in',
                   #    'burn-in': 100},
                   #1: {'policy': 'poly',
                   #    'power': 0.9} }

# dataSet
imgSize = [640, 352]
trainList = "/home/wfw/data/VOCdevkit/cifar10/train_data.txt"
valList = "/home/wfw/data/VOCdevkit/cifar10/test_data.txt"
valLabelPath = "/home/wfw/data/VOCdevkit/BerkeleyDet/Annotations/"
train_batch_size = 256
test_batch_size = 100

# yolo param
anchors = [[[44,76], [65,116], [117,164]],
           [[15,18], [22,38], [29,56]],
           [[7,10], [9,18], [14,30]]]
#className = ['truck', 'person', 'bicycle', 'car', 'motorbike', 'bus']
className = ['person']
numClasses = 1
iouThresh = 0.45
confThresh=0.24

jitter = 0.3
#hue = 0.1
hue = 0
saturation = 1.5
exposure = 1.5

# GPU SETTINGS
CUDA_DEVICE = [0] # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 1 # set to 1 if want to run on gpu.

# SETTINGS FOR DISPLAYING ON TENSORBOARD
visdomTrain = False
visdomVal = False
