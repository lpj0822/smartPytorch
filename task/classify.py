from src.utils import *
from src.loss import *
from src.bn_fusion import fuse_bn_recursively
import arch.config as config

import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

class run_classify():
    def __init__(self, model, params, hyperparams, optimizer, trainloader, valloader, save_path, logger):#, viz):
        self.model = model
        self.params = params
        self.hyperparams = hyperparams
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.valloader = valloader
        self.savePath = save_path
        self.logger = logger
        #self.viz = viz

        self.lossFunc = {}

    def run(self):
        global best_prec1
        best_prec1 = 0

        # optionally resume from a checkpoint
        if config.resume is not None:
            if os.path.exists(os.path.join(self.savePath, config.resume)):
                checkpoint_file = os.path.join(
                    self.savePath, config.resume)
            if os.path.isfile(checkpoint_file):
                checkpoint = torch.load(checkpoint_file)
                start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                logging.info("loaded checkpoint '%s' (epoch %s)",
                             checkpoint_file, checkpoint['epoch'])
            else:
                logging.error("no checkpoint found '%s'", os.path.join(self.savePath, config.resume))
        else:
            start_epoch = 0

        # define loss function (criterion) and optimizer
        for key, val in self.hyperparams.items():
            if key == 'losstype' and val not in self.lossFunc:
                self.lossFunc[val] = eval(val)()
                #lossFunc[val]
        criterion = getattr(self.model, 'criterion', nn.CrossEntropyLoss)()

        for epoch in range(start_epoch, self.params["maxEpoches"]):
            adjust_optimizer(self.optimizer, epoch, config.optimizer)

            # train for one epoch
            train_loss, train_prec1, train_prec5 = self.train(
                self.trainloader, self.model, criterion, epoch, self.optimizer)

            # evaluate on validation set
            val_loss, val_prec1, val_prec5 = self.validate(
                self.valloader, self.model, criterion, epoch, self.optimizer)

            # remember best prec@1 and save checkpoint
            is_best = val_prec1 > best_prec1
            best_prec1 = max(val_prec1, best_prec1)

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, path=self.savePath)
            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec@1 {train_prec1:.3f} \t'
                         'Training Prec@5 {train_prec5:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec@1 {val_prec1:.3f} \t'
                         'Validation Prec@5 {val_prec5:.3f} \n'
                         .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1,
                                 train_prec5=train_prec5, val_prec5=val_prec5))

    def forward(self, data_loader, model, criterion, epoch=0, training=True, optimizer=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for i, (inputs, target) in enumerate(data_loader):
            if training:
                adjust_learning_rate(optimizer, config.learning_policy,
                                    i, len(data_loader), epoch)

            # measure data loading time
            data_time.update(time.time() - end)
            target = target.cuda(async=True)
            input_var = Variable(inputs.cuda(), volatile=not training)
            target_var = Variable(target)
            loss = 0

            # compute output
            outputs = model(input_var)
            for lossType, output in outputs.items():
                loss += self.lossFunc[lossType](output, target_var)

            # bn_visual
            #self.viz.plot()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))

            if training:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # tensorboard logger
            infotrain = {
                'losstrain': loss.data
            }
            infoval = {
                'lossval': loss.data,
                'prec1': prec1,
                'prec5': prec5
            }

            if i % config.display == 0:
                logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
                             'LR: [{3}/{4}]\t'\
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                             'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(data_loader), '%.7f' % optimizer.param_groups[0]['lr'] if training else 'NULL',
                    '%.7f' % optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 and training else 'NULL',
                    phase='TRAINING' if training else 'EVALUATING',
                    batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5))

                if training:
                    for tag, value in infotrain.items():
                        self.logger.scalar_summary(tag, value, (epoch * (len(data_loader) - 1) + i))
                else:
                    for tag, value in infoval.items():
                        self.logger.scalar_summary(tag, value, (epoch * (len(data_loader) - 1) + i))

        return losses.avg, top1.avg, top5.avg

    def train(self, data_loader, model, criterion, epoch, optimizer):
        # switch to train mode
        model.train()
        return self.forward(data_loader, model, criterion, epoch,
                       training=True, optimizer=optimizer)


    def validate(self, data_loader, model, criterion, epoch, optimizer):
        # switch to evaluate mode
        #model = fuse_bn_recursively(model)
        model.eval()
        return self.forward(data_loader, model, criterion, epoch,
                       training=False, optimizer=optimizer)