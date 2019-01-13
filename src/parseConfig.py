import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.basic_module import *
from modules import *
from src.utils import *
from src.loss import *
import arch.config as config

def parse_data_config():
    params = {}

    params["trainList"] = config.trainList
    params["valList"] = config.valList
    params["train_batch_size"] = config.train_batch_size
    params["test_batch_size"] = config.test_batch_size
    params["maxEpoches"] = config.max_epoch

    logging.info("created model with configuration : {}".format(params))
    return params

def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs

def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'basicConv':
            bn = int(module_def['batch_normalize']) if 'batch_normalize' in module_def else 0
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = int(module_def['pad']) if 'pad' in module_def else 0
            dilation = int(module_def['dilation']) if 'dilation' in module_def else 1
            groups = int(module_def['groups']) if 'groups' in module_def else 1
            activation = module_def['activation']
            stopbackward = str(module_def['stopbackward']) if 'stopbackward' in module_def else 'None'
            hyperparams['conv_%d' % i] = stopbackward
            modules.add_module('conv_%d' % i, basicConv(in_channels=output_filters[-1],
                                                        n_filters=filters,
                                                        k_size=kernel_size,
                                                        stride=int(module_def['stride']),
                                                        bias=not bn,
                                                        padding=pad,
                                                        dilation=dilation,
                                                        groups=groups,
                                                        with_bn=bn,
                                                        activation=activation))
        elif module_def['type'] == 'pool':
            pooltype = str(module_def['pooltype'])
            kernel_size = int(module_def['size']) if 'size' in module_def else 1
            stride = int(module_def['stride']) if 'stride' in module_def else 1
            if pooltype=="max":
                modules.add_module('pool_%d' % i, nn.MaxPool2d(kernel_size, stride=stride))
            elif pooltype=="avg":
                modules.add_module('pool_%d' % i, nn.AvgPool2d(kernel_size, stride=stride))
            elif pooltype=="global_avg":
                modules.add_module('pool_%d' % i, nn.AdaptiveAvgPool2d(1))

        elif module_def['type'] == 'innerProduct':
            nClass = int(module_def['nClass'])
            stopbackward = str(module_def['stopbackward']) if 'stopbackward' in module_def else 'None'
            hyperparams['innerProduct_%d' % i] = stopbackward
            modules.add_module('innerProduct_%d' % i, innerProductLayer(output_filters[-1], nClass))

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'skip':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('skip_%d' % i, EmptyLayer())

        elif module_def['type'] == 'basemodel':
            modeltype = str(module_def['modeltype'])
            basemodel = eval(modeltype)()
            for m in basemodel.modules():
                if isinstance(m, nn.Conv2d):
                    filters = m.out_channels
            modules.add_module('basemodel_%d' % i, basemodel)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return hyperparams, module_list

class Net(nn.Module):
    """parse and generate model"""

    def __init__(self, cfg_path):
        super(Net, self).__init__()
        self.module_defs = parse_model_config(cfg_path)
        self.hyperparams, self.module_list = create_modules(self.module_defs)

    def forward(self, x):
        layer_outputs = []
        loss_outputs = {}

        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['basicConv', 'pool', 'innerProduct', 'basemodel']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'skip':
                layer_i = int(module_def['from'])
                x = layer_outputs[layer_i]
            elif module_def['type'] == 'loss':
                self.hyperparams['losstype'] = module_def['losstype']
                loss_outputs[module_def['losstype']] = x

            layer_outputs.append(x)

        return loss_outputs

def test():
    import numpy as np
    import hiddenlayer as hl
    #params = parse_data_config()
    #print(params)

    model = Net("arch/vgg16.cfg")
    if len(config.CUDA_DEVICE) > 1:
        logging.info('Using {} GPUS'.format(len(config.CUDA_DEVICE)))
        model = nn.DataParallel(model, device_ids=config.CUDA_DEVICE)
    model.cuda().train()

    x = torch.from_numpy(np.random.rand(4, 3, 32, 32))
    x = Variable(x.type(torch.FloatTensor).cuda())
    ys = model(x)
    print(ys)

if __name__ == "__main__":
    test()

#def main():
#    if
'''
    model = Net("arch/test.cfg")
    model.cuda()
    input = torch.zeros((1, 3, 224, 224))
    summary(model, Variable(input.type(torch.FloatTensor).cuda()))

    x = torch.ones(1,3,352,640)
    y = torch.ones(8,640,352)
    x = Variable(x.type(torch.FloatTensor).cuda())
    y = Variable(y.type(torch.LongTensor).cuda())
    ys = model(x)
    print(ys)
'''

