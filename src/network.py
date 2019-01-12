from src.data import *
from src.parseConfig import *
from src.utils import *
from src.loss import *
from src.logger import Logger
import arch.config as config
from task import *

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader

cuda = torch.cuda.is_available()

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if cuda:
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

def run_task():
    save_path = config.task + "/" + config.cfg.split("/")[1].split(".")[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))

    # init params
    params = parse_data_config()

    dstrain = load_images_and_labels_cls(params["trainList"], transform=config.transform_train)
    dsval = load_images_and_labels_cls(params["valList"], transform=config.transform_train)
    trainloader = DataLoader(dstrain, batch_size=params["train_batch_size"], shuffle=True, num_workers=1)
    valloader = DataLoader(dsval, batch_size=params["test_batch_size"], shuffle=False, num_workers=1)

    numTrainData = len(dstrain)
    numTestData = len(dsval)
    logging.info("Train data num: {}".format(numTrainData))
    logging.info("Test data num: {}".format(numTestData))

    # creat model
    model = Net(config.cfg)
    hyperparams = model.hyperparams

    # multi-gpu
    if len(config.CUDA_DEVICE) > 1:
        logging.info('Using {} GPUS'.format(len(config.CUDA_DEVICE)))
        model = nn.DataParallel(model, device_ids=config.CUDA_DEVICE)
    model.cuda()

    # set gradient
    model, optimizer = multiLrMethod(model, hyperparams)

    # summary the model
    summary(model, hyperparams, save_path)

    # write logger
    logger = Logger(os.path.join(save_path, "logs"))

    # bn_param_visual
    viz = vis_bn(model)

    if config.task == "classify":
        run_cls = run_classify(model, params, hyperparams, optimizer, trainloader, valloader, save_path, logger, viz)
        run_cls.run()

if __name__ == "__main__":
    run_task()
