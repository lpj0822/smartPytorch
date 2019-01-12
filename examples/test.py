from src.data import *
from src.parseConfig import *
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from visdom import Visdom
import numpy as np

# init params
params = parse_data_config()

dstrain = load_images_and_labels_cls(params["trainList"], transform=config.transform_train)
dsval = load_images_and_labels_cls(params["valList"], transform=config.transform_train)
trainloader = DataLoader(dstrain, batch_size=params["train_batch_size"], shuffle=True, num_workers=1)
valloader = DataLoader(dsval, batch_size=params["test_batch_size"], shuffle=False, num_workers=1)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        # an affine operation: y = Wx + b
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.bn1(self.conv1(x)))
        # If the size is a square you can only specify a single number
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class vis_bn():
    def __init__(self, model):
        self.axss = []
        self.bn_layers = []
        self.bn_layer_name = []
        bn_layer_index = 0

        for name, p in model.named_parameters():
            if 'bn' in name and 'weight' in name:
                self.bn_layer_name.append('.'.join(name.split(".")[:-1]))

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                f, axs = plt.subplots(4, 1, figsize=(6, 6))
                f.suptitle(self.bn_layer_name[bn_layer_index])
                bn_layer_index += 1
                self.axss.append(axs)
                self.bn_layers.append(m)

    def plot(self):
        for i, axs in enumerate(self.axss):
            m = self.bn_layers[i]
            self.plot_hist(axs, m.weight.data, m.bias.data, m.running_mean.data, m.running_var.data)

    def plot_hist(self, axs, weight, bias, running_mean, running_var):
        [a.clear() for a in [axs[0], axs[1], axs[2], axs[3]]]
        axs[0].bar(range(len(running_mean.cpu().numpy())), weight.cpu().numpy(), color='#FF9359')
        axs[1].bar(range(len(running_var.cpu().numpy())), bias.cpu().numpy(), color='g')
        axs[2].bar(range(len(running_mean.cpu().numpy())), running_mean.cpu().numpy(), color='#74BCFF')
        axs[3].bar(range(len(running_var.cpu().numpy())), running_var.cpu().numpy(), color='y')
        axs[0].set_ylabel('weight')
        axs[1].set_ylabel('bias')
        axs[2].set_ylabel('running_mean')
        axs[3].set_ylabel('running_var')
        plt.pause(0.01)

criterion = nn.CrossEntropyLoss()

net = Net()
net.cuda()
net.train()
print(net)

# define visdom
#viz = Visdom(env='bn_param')
viz = vis_bn(net)

# define optimizer
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

start_epoch=0
if os.path.exists("./examples/parmas.pkl"):
    print("Loading checkpoint......")
    checkpoint = torch.load("./examples/parmas.pkl")
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])

for epoch in range(start_epoch ,100):
    if epoch == 20:
        net.bn1.momentum = 0
    for i, (inputs, target) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        target = Variable(target.cuda())

        output = net(inputs)
        loss = criterion(output, target)
        viz.plot()
        #plot_hist(weight, bias, running_mean, running_var)
        #viz.matplot(f)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print("Epoch: {}\t loss: {}\t".format(epoch, loss))

    if epoch % 5 ==0:
        print("saving checkpoint......")
        torch.save({'epoch': epoch, 'state_dict': net.state_dict()}, "./examples/parmas.pkl")