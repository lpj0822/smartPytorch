import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class cross_entropy2dCls(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(cross_entropy2dCls, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        loss_fn = nn.CrossEntropyLoss(weight=self.weight, size_average=self.size_average)
        loss = loss_fn(input, target)

        return loss

class cross_entropy2dDet(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(cross_entropy2dDet, self).__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h > ht and w > wt:  # upsample labels
            target = target.unsequeeze(1)
            target = F.upsample(target, size=(h, w), mode='nearest')
            target = target.sequeeze(1)
        elif h < ht and w < wt:  # upsample images
            input = F.upsample(input, size=(ht, wt), mode='bilinear')
        elif h != ht and w != wt:
            raise Exception("Only support upsampling")

        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss_fn = nn.CrossEntropyLoss(ignore_index=250,
                                      weight=self.weight, size_average=self.size_average)
        loss = loss_fn(input, target)
        # if size_average:
        #    loss = loss / mask.sum().item()

        return loss

def bootstrapped_cross_entropy2d(input, target, K, weight=None, size_average=True):
    
    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, target, K, weight=None, size_average=True):
        n, c, h, w = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, weight=weight, ignore_index=250,
                          reduce=False, size_average=False)
        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(input=torch.unsqueeze(input[i], 0),
                                           target=torch.unsqueeze(target[i], 0),
                                           K=K,
                                           weight=weight,
                                           size_average=size_average)
    return loss / float(batch_size)


def multi_scale_cross_entropy2d(input, target, weight=None, size_average=True, scale_weight=None):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None: # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(input=inp, target=target, weight=weight, size_average=size_average)

    return loss

class focalLoss(nn.Module):
    def __init__(self, class_num, ignoreIndex=None, alpha=None, gamma=2, size_average=True):
        super(focalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)

        self.class_num = class_num
        self.ignoreIndex = ignoreIndex
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        P = F.softmax(input, dim = 1)
        P = P.transpose(1, 2).transpose(2, 3).contiguous().view(-1, self.class_num)

        ids = target.view(-1, 1)
        if self.ignoreIndex != None:
            P = P[(ids != self.ignoreIndex).expand_as(P)].view(-1, self.class_num)
            ids = ids[ids!=self.ignoreIndex].view(-1, 1)

        class_mask = Variable(torch.zeros(P.shape))
        class_mask.scatter_(1, ids.cpu(), 1.)

        if input.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
            class_mask = class_mask.cuda()
        self.alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)
        log_p = probs.log()
        batch_loss = -log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    CE = nn.CrossEntropyLoss(ignore_index=2)
    N = 4
    C = 5
    inputs = torch.rand(N, C, 5, 5)
    targets = torch.LongTensor(N, 5, 5).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())

    inputs_ce = Variable(inputs.clone(), requires_grad=True)
    targets_ce = Variable(targets.clone())
    print('----inputs----')
    print(inputs)
    print('---target-----')
    print(targets)

    fl_loss1 = focalLoss(inputs_fl, targets_fl, 5, ignoreIndex=0)  
    ce_loss = CE(inputs_ce, targets_ce)
    print('ce = {}, fl1 ={}'.format(ce_loss.data[0], fl_loss1.data[0]))
    #fl_loss.backward()
    #ce_loss.backward()
    #print(inputs_fl.grad.data)
    #print(inputs_ce.grad.data)


