import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from src.utils import *

class basicConv(nn.Module):

    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, groups=1, with_bn=True, activation="linear"):
        super(basicConv, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)

        self.cb_unit = nn.Sequential(conv_mod)


        if with_bn:
            self.cb_unit.add_module('bn', nn.BatchNorm2d(int(n_filters)))

        if activation=="relu":
            self.cb_unit.add_module('relu', nn.ReLU(inplace=True))

        elif activation=="leaky":
            self.cb_unit.add_module('leaky', nn.LeakyReLU(0.1, inplace=True))

        elif activation=="logistic":
            self.cb_unit.add_module('logistic', nn.Sigmoid())

        elif activation=="tanh":
            self.cb_unit.add_module('tanh', nn.Tanh())

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class innerProductLayer(nn.Module):
    """innerPooduct layers"""

    def __init__(self, in_channels, nClass):
        super(innerProductLayer, self).__init__()
        self.fc = nn.Linear(in_channels, nClass)

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        return self.fc(inputs)

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""

    def __init__(self):
        super(EmptyLayer, self).__init__()

class YOLOLayer(nn.Module):

    def __init__(self, anchors, nC, img_width, img_height, anchor_idxs):
        super(YOLOLayer, self).__init__()

        anchors = [(a_w, a_h) for a_w, a_h in anchors]  # (pixels)
        nA = len(anchors)

        self.anchors = anchors
        self.nA = nA  # number of anchors (3)
        self.nC = nC  # number of classes (80)
        self.bbox_attrs = 5 + nC
        self.img_width = img_width  # from hyperparams in cfg file, NOT from parser
        self.img_height = img_height

        if anchor_idxs[0] == (nA * 2):  # 6
            stride = 32
        elif anchor_idxs[0] == nA:  # 3
            stride = 16
        else:
            stride = 8

        # Build anchor grids
        nGw = int(self.img_width / stride)  # number grid points
        nGh = int(self.img_height / stride)  # number grid points
        self.grid_x = torch.arange(nGw).repeat(nGh, 1).view([1, 1, nGh, nGw]).float()
        self.grid_y = torch.arange(nGh).repeat(nGw, 1).t().view([1, 1, nGh, nGw]).float()
        self.scaled_anchors = torch.FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, nA, 1, 1))
        self.weights = class_weights()

        self.loss_means = torch.ones(6)

    def forward(self, p, targets=None, batch_report=False):
        FT = torch.cuda.FloatTensor if p.is_cuda else torch.FloatTensor

        bs = p.shape[0]  # batch size
        nGh = p.shape[2]
        nGw = p.shape[3]  # number of grid points
        stride = self.img_width / nGw

        if p.is_cuda and not self.grid_x.is_cuda:
            self.grid_x, self.grid_y = self.grid_x.cuda(), self.grid_y.cuda()
            self.anchor_w, self.anchor_h = self.anchor_w.cuda(), self.anchor_h.cuda()
            self.weights, self.loss_means = self.weights.cuda(), self.loss_means.cuda()

        # p.view(12, 255, 13, 13) -- > (12, 3, 13, 13, 80)  # (bs, anchors, grid, grid, classes + xywh)
        p = p.view(bs, self.nA, self.bbox_attrs, nGh, nGw).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        # Get outputs
        x = torch.sigmoid(p[..., 0])  # Center x
        y = torch.sigmoid(p[..., 1])  # Center y

        # Width and height (yolo method)
        w = p[..., 2]  # Width
        h = p[..., 3]  # Height
        width = torch.exp(w.data) * self.anchor_w
        height = torch.exp(h.data) * self.anchor_h

        # Width and height (power method)
        # w = torch.sigmoid(p[..., 2])  # Width
        # h = torch.sigmoid(p[..., 3])  # Height
        # width = ((w.data * 2) ** 2) * self.anchor_w
        # height = ((h.data * 2) ** 2) * self.anchor_h

        # Add offset and scale with anchors (in grid space, i.e. 0-13)
        pred_boxes = FT(bs, self.nA, nGh, nGw, 4)
        pred_conf = p[..., 4]  # Conf
        pred_cls = p[..., 5:]  # Class

        # Training
        if targets is not None:
            MSELoss = nn.MSELoss()
            BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            CrossEntropyLoss = nn.CrossEntropyLoss()

            if batch_report:
                gx = self.grid_x[:, :, :nGh, :nGw]
                gy = self.grid_y[:, :, :nGh, :nGw]
                pred_boxes[..., 0] = x.data + gx - width / 2
                pred_boxes[..., 1] = y.data + gy - height / 2
                pred_boxes[..., 2] = x.data + gx + width / 2
                pred_boxes[..., 3] = y.data + gy + height / 2

            tx, ty, tw, th, mask, tcls, TP, FP, FN, TC = \
                build_targets(pred_boxes, pred_conf, pred_cls, targets, self.scaled_anchors, self.nA, self.nC, nGw, nGh,
                               batch_report)
            tcls = tcls[mask]
            if x.is_cuda:
                tx, ty, tw, th, mask, tcls = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda(), mask.cuda(), tcls.cuda()

            # Compute losses
            nT = sum([len(x) for x in targets])  # number of targets
            nM = mask.sum().float()  # number of anchors (assigned to targets)
            nB = len(targets)  # batch size
            k = nM / nB
            if nM > 0:
                lx = k * MSELoss(x[mask], tx[mask])
                ly = k * MSELoss(y[mask], ty[mask])
                lw = k * MSELoss(w[mask], tw[mask])
                lh = k * MSELoss(h[mask], th[mask])

                # lconf = k * BCEWithLogitsLoss(pred_conf[mask], mask[mask].float())
                lconf = (k * 10) * BCEWithLogitsLoss(pred_conf, mask.float())

                lcls = (k / 10) * CrossEntropyLoss(pred_cls[mask], torch.argmax(tcls, 1))
                # lcls = (k * 10) * BCEWithLogitsLoss(pred_cls[mask], tcls.float())
            else:
                lx, ly, lw, lh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0]), FT([0]), FT([0])

            # Add confidence loss for background anchors (noobj)
            # lconf += k * BCEWithLogitsLoss(pred_conf[~mask], mask[~mask].float())

            # Sum loss components
            balance_losses_flag = False
            if balance_losses_flag:
                k = 1 / self.loss_means.clone()
                loss = (lx * k[0] + ly * k[1] + lw * k[2] + lh * k[3] + lconf * k[4] + lcls * k[5]) / k.mean()

                self.loss_means = self.loss_means * 0.99 + \
                                  FT([lx.data, ly.data, lw.data, lh.data, lconf.data, lcls.data]) * 0.01
            else:
                loss = lx + ly + lw + lh + lconf + lcls

            # Sum False Positives from unassigned anchors
            FPe = torch.zeros(self.nC)
            if batch_report:
                i = torch.sigmoid(pred_conf[~mask]) > 0.5
                if i.sum() > 0:
                    FP_classes = torch.argmax(pred_cls[~mask][i], 1)
                    FPe = torch.bincount(FP_classes, minlength=self.nC).float().cpu()  # extra FPs

            return loss, loss.item(), lx.item(), ly.item(), lw.item(), lh.item(), lconf.item(), lcls.item(), \
                   nT, TP, FP, FPe, FN, TC

        else:
            pred_boxes[..., 0] = x.data + self.grid_x
            pred_boxes[..., 1] = y.data + self.grid_y
            pred_boxes[..., 2] = width
            pred_boxes[..., 3] = height

            # If not in training phase return predictions
            output = torch.cat((pred_boxes.view(bs, -1, 4) * stride,
                                torch.sigmoid(pred_conf.view(bs, -1, 1)), pred_cls.view(bs, -1, self.nC)), -1)
            return output.data

class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True):
        super(conv2DBatchNorm, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)


        if with_bn:
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(n_filters)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs

class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True, dilation=1, with_bn=True, leaky_relu=False):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, 
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        if with_bn and leaky_relu:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.LeakyReLU(0.1, inplace=True),)

        elif with_bn == True and leaky_relu == False:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(n_filters)),
                                          nn.ReLU(inplace=True),)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0),
                                       nn.BatchNorm2d(out_size),
                                       nn.ReLU(),)
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 0),
                                       nn.ReLU(),)
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 0),
                                       nn.ReLU(),)
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetDown3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown3, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs

class segnetUp3(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp3, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(in_channels,  n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(n_filters,  n_filters, k_size=3, padding=1, stride=stride, bias=False)
        self.convbn3 = nn.Conv2DBatchNorm(n_filters,  n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class residualBlockYolo(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(residualBlockYolo, self).__init__()
        self.convbnleakyrelu1 = conv2DBatchNormRelu(in_channels, n_filters, k_size=3, stride=2, padding=1, bias=False, leaky_relu=True)
        self.convbnleakyrelu2 = conv2DBatchNormRelu(n_filters, n_filters/2, k_size=1, stride=1, padding=0, bias=False, leaky_relu=True)
        self.convbnleakyrelu3 = conv2DBatchNormRelu(n_filters/2, n_filters, k_size=3, stride=1, padding=1, bias=False, leaky_relu=True)

    def forward(self ,x):
        x = self.convbnleakyrelu1(x)

        residual = x
        out = self.convbnleakyrelu2(x)
        out = self.convbnleakyrelu3(out)

        return out + residual

class residualBlockYoloExp(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(residualBlockYoloExp, self).__init__()
        self.convbnleakyrelu1 = conv2DBatchNormRelu(in_channels, n_filters, k_size=1, stride=1, padding=0, bias=False, leaky_relu=True)
        self.convbnleakyrelu2 = conv2DBatchNormRelu(n_filters, n_filters*2, k_size=3, stride=1, padding=1, bias=False, leaky_relu=True)

    def forward(self ,x):
        residual = x

        out = self.convbnleakyrelu1(x)
        out = self.convbnleakyrelu2(out)

        return out + residual

class BlockYoloExp(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(BlockYoloExp, self).__init__()
        self.convbnleakyrelu1 = conv2DBatchNormRelu(in_channels, n_filters, k_size=1, stride=1, padding=0, bias=False, leaky_relu=True)
        self.convbnleakyrelu2 = conv2DBatchNormRelu(n_filters, n_filters*2, k_size=3, stride=1, padding=1, bias=False, leaky_relu=True)

    def forward(self, x):
        out = self.convbnleakyrelu1(x)
        out = self.convbnleakyrelu2(out)

        return out

class YOLOLoss(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLOLoss, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = self.build_targets(
                pred_boxes=pred_boxes.cpu().data,
                pred_conf=pred_conf.cpu().data,
                pred_cls=pred_cls.cpu().data,
                target=targets.cpu().data,
                anchors=scaled_anchors.cpu().data,
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            nProposals = int((pred_conf > 0.5).sum().item())
            recall = float(nCorrect / nGT) if nGT else 1
            precision = float(nCorrect / nProposals) if nProposals else 0

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

            return (
                loss,
                loss_x.item(),
                loss_y.item(),
                loss_w.item(),
                loss_h.item(),
                loss_conf.item(),
                loss_cls.item(),
                recall,
                precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output

    def build_targets(self, pred_boxes, pred_conf, pred_cls, target, anchors, num_anchors, num_classes, grid_size, ignore_thres, img_dim):
        nB = target.size(0)
        nA = num_anchors
        nC = num_classes
        nG = grid_size
        mask = torch.zeros(nB, nA, nG, nG)
        conf_mask = torch.ones(nB, nA, nG, nG)
        tx = torch.zeros(nB, nA, nG, nG)
        ty = torch.zeros(nB, nA, nG, nG)
        tw = torch.zeros(nB, nA, nG, nG)
        th = torch.zeros(nB, nA, nG, nG)
        tconf = torch.ByteTensor(nB, nA, nG, nG).fill_(0)
        tcls = torch.ByteTensor(nB, nA, nG, nG, nC).fill_(0)

        nGT = 0
        nCorrect = 0
        for b in range(nB):
            for t in range(target.shape[1]):
                if target[b, t].sum() == 0:
                    continue
                nGT += 1
                # Convert to position relative to box
                gx = target[b, t, 1] * nG
                gy = target[b, t, 2] * nG
                gw = target[b, t, 3] * nG
                gh = target[b, t, 4] * nG
                # Get grid box indices
                gi = int(gx)
                gj = int(gy)
                # Get shape of gt box
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                # Get shape of anchor box
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
                # Calculate iou between gt and anchor shapes
                anch_ious = bbox_iou(gt_box, anchor_shapes)
                # Where the overlap is larger than threshold set mask to zero (ignore)
                conf_mask[b, anch_ious > ignore_thres, gj, gi] = 0
                # Find the best matching anchor box
                best_n = np.argmax(anch_ious)
                # Get ground truth box
                gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
                # Get the best prediction
                pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)
                # Masks
                mask[b, best_n, gj, gi] = 1
                conf_mask[b, best_n, gj, gi] = 1
                # Coordinates
                tx[b, best_n, gj, gi] = gx - gi
                ty[b, best_n, gj, gi] = gy - gj
                # Width and height
                tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
                th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
                # One-hot encoding of label
                target_label = int(target[b, t, 0])
                tcls[b, best_n, gj, gi, target_label] = 1
                tconf[b, best_n, gj, gi] = 1

                # Calculate iou between ground truth and best matching prediction
                iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
                pred_label = torch.argmax(pred_cls[b, best_n, gj, gi])
                score = pred_conf[b, best_n, gj, gi]
                if iou > 0.5 and pred_label == target_label and score > 0.5:
                    nCorrect += 1

        return nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls

class pyramidDownsampling(nn.Module):

    def __init__(self, in_channels, pool_sizes, fusion_mode='cat', with_bn=True):
        super(pyramidDownsampling, self).__init__()

        self.pool_sizes = pool_sizes
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]

        k_sizes = []
        strides = []
        for pool_size in self.pool_sizes:
            k_sizes.append((int(h/pool_size), int(w/pool_size)))
            strides.append((int(h/pool_size), int(w/pool_size)))
        if self.fusion_mode == 'cat': # pspnet: concat (including x)
            output_slices = [x]

            for i, pool_size in enumerate(zip(self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                out = F.upsample(out, size=(h,w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else: # icnet: element-wise sum (including x)
            pp_sum = x

            for i, pool_size in enumerate(zip(self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                out = F.upsample(out, size=(h,w), mode='bilinear')
                pp_sum = pp_sum + out

            return pp_sum

class linknetUp(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(linknetUp, self).__init__()

        # B, 2C, H, W -> B, C/2, H, W
        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters/2, k_size=1, stride=1, padding=1)

        # B, C/2, H, W -> B, C/2, H, W
        self.deconvbnrelu2 = nn.deconv2DBatchNormRelu(n_filters/2, n_filters/2, k_size=3,  stride=2, padding=0)

        # B, C/2, H, W -> B, C, H, W
        self.convbnrelu3 = conv2DBatchNormRelu(n_filters/2, n_filters, k_size=1, stride=1, padding=1)

    def forward(self, x):
        x = self.convbnrelu1(x)
        x = self.deconvbnrelu2(x)
        x = self.convbnrelu3(x)
        return x


class FRRU(nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """
    def __init__(self, prev_channels, out_channels, scale):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels

        self.conv1 = conv2DBatchNormRelu(prev_channels + 32, out_channels, k_size=3, stride=1, padding=1)
        self.conv2 = conv2DBatchNormRelu(out_channels, out_channels, k_size=3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s*self.scale for _s in y_prime.shape[-2:]])
        x = F.upsample(x, size=upsample_size, mode='nearest')
        z_prime = z + x

        return y_prime, z_prime


class RU(nn.Module):
    """
    Residual Unit for FRRN
    """
    def __init__(self, channels, kernel_size=3, strides=1):
        super(RU, self).__init__()

        self.conv1 = conv2DBatchNormRelu(channels, channels, k_size=kernel_size, stride=strides, padding=1)
        self.conv2 = conv2DBatchNorm(channels, channels, k_size=kernel_size, stride=strides, padding=1)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + incoming


class residualConvUnit(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(residualConvUnit, self).__init__()
        
        self.residual_conv_unit = nn.Sequential(nn.ReLU(inplace=True),
                                                nn.Conv2d(channels, channels, kernel_size=kernel_size),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(channels, channels, kernel_size=kernel_size),)
    def forward(self, x):
        input = x
        x = self.residual_conv_unit(x)
        return x + input

class multiResolutionFusion(nn.Module):
    def __init__(self, channels, up_scale_high, up_scale_low, high_shape, low_shape):
        super(multiResolutionFusion, self).__init__()
        
        self.up_scale_high = up_scale_high
        self.up_scale_low = up_scale_low

        self.conv_high = nn.Conv2d(high_shape[1], channels, kernel_size=3)

        if low_shape is not None:
            self.conv_low = nn.Conv2d(low_shape[1], channels, kernel_size=3)

    def forward(self, x_high, x_low):
        high_upsampled = F.upsample(self.conv_high(x_high), 
                                    scale_factor=self.up_scale_high, 
                                    mode='bilinear')

        if x_low is None:
            return high_upsampled

        low_upsampled = F.upsample(self.conv_low(x_low),
                                   scale_factor=self.up_scale_low, 
                                   mode='bilinear')

        return low_upsampled + high_upsampled

class chainedResidualPooling(nn.Module):
    def __init__(self, channels, input_shape):
        super(chainedResidualPooling, self).__init__()
        
        self.chained_residual_pooling = nn.Sequential(nn.ReLU(inplace=True),
                                                      nn.MaxPool2d(5, 1, 2),
                                                      nn.Conv2d(input_shape[1], channels, kernel_size=3),)

    def forward(self, x):
        input = x
        x = self.chained_residual_pooling(x)
        return x + input


class pyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, model_name='pspnet', fusion_mode='cat', with_bn=True):
        super(pyramidPooling, self).__init__()

        bias = not with_bn

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, with_bn=with_bn))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes
        self.model_name = model_name
        self.fusion_mode = fusion_mode

    def forward(self, x):
        h, w = x.shape[2:]

        if self.training or self.model_name != 'icnet': # general settings or pspnet
            k_sizes = []
            strides = []
            for pool_size in self.pool_sizes:
                k_sizes.append((int(h/pool_size), int(w/pool_size)))
                strides.append((int(h/pool_size), int(w/pool_size)))
        else: # eval mode and icnet: pre-trained for 1025 x 2049
            k_sizes = [(8, 15), (13, 25), (17, 33), (33, 65)]
            strides = [(5, 10), (10, 20), (16, 32), (33, 65)]

        if self.fusion_mode == 'cat': # pspnet: concat (including x)
            output_slices = [x]

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                output_slices.append(out)

            return torch.cat(output_slices, dim=1)
        else: # icnet: element-wise sum (including x)
            pp_sum = x

            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, k_sizes[i], stride=strides[i], padding=0)
                #out = F.adaptive_avg_pool2d(x, output_size=(pool_size, pool_size))
                if self.model_name != 'icnet':
                    out = module(out)
                out = F.upsample(out, size=(h,w), mode='bilinear')
                pp_sum = pp_sum + out

            return pp_sum


class bottleNeckPSP(nn.Module):
    
    def __init__(self, in_channels, mid_channels, out_channels, 
                 stride, dilation=1, with_bn=True):
        super(bottleNeckPSP, self).__init__()

        bias = not with_bn
            
        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, with_bn=with_bn) 
        if dilation > 1: 
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                            stride=stride, padding=dilation,
                                            bias=bias, dilation=dilation, with_bn=with_bn) 
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                            stride=stride, padding=1,
                                            bias=bias, dilation=1, with_bn=with_bn)
        self.cb3 = conv2DBatchNorm(mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, with_bn=with_bn)
        self.cb4 = conv2DBatchNorm(in_channels, out_channels, 1, stride=stride, padding=0, bias=bias, with_bn=with_bn)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x)
        return F.relu(conv+residual, inplace=True)


class bottleNeckIdentifyPSP(nn.Module):
    
    def __init__(self, in_channels, mid_channels, stride, dilation=1, with_bn=True):
        super(bottleNeckIdentifyPSP, self).__init__()

        bias = not with_bn

        self.cbr1 = conv2DBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, with_bn=with_bn) 
        if dilation > 1: 
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                            stride=1, padding=dilation,
                                            bias=bias, dilation=dilation, with_bn=with_bn) 
        else:
            self.cbr2 = conv2DBatchNormRelu(mid_channels, mid_channels, 3,
                                            stride=1, padding=1,
                                            bias=bias, dilation=1, with_bn=with_bn)
        self.cb3 = conv2DBatchNorm(mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, with_bn=with_bn)
        
    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x+residual, inplace=True)


class residualBlockPSP(nn.Module):
    
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1, include_range='all', with_bn=True):
        super(residualBlockPSP, self).__init__()

        if dilation > 1:
            stride = 1

        # residualBlockPSP = convBlockPSP + identityBlockPSPs
        layers = []
        if include_range in ['all', 'conv']:
            layers.append(bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation, with_bn=with_bn))
        if include_range in ['all', 'identity']:
            for i in range(n_blocks-1):
                layers.append(bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation, with_bn=with_bn))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class cascadeFeatureFusion(nn.Module):
    def __init__(self, n_classes, low_in_channels, high_in_channels, out_channels, with_bn=True):
        super(cascadeFeatureFusion, self).__init__()

        bias = not with_bn

        self.low_dilated_conv_bn = conv2DBatchNorm(low_in_channels, out_channels, 3, stride=1, padding=2, bias=bias, dilation=2, with_bn=with_bn)
        self.low_classifier_conv = nn.Conv2d(int(low_in_channels), int(n_classes), kernel_size=1, padding=0, stride=1, bias=True, dilation=1) # Train only
        self.high_proj_conv_bn = conv2DBatchNorm(high_in_channels, out_channels, 1, stride=1, padding=0, bias=bias, with_bn=with_bn)

    def forward(self, x_low, x_high):
        x_low_upsampled = F.upsample(x_low, size=get_interp_size(x_low, z_factor=2), mode='bilinear')

        low_cls = self.low_classifier_conv(x_low_upsampled)

        low_fm = self.low_dilated_conv_bn(x_low_upsampled)
        high_fm = self.high_proj_conv_bn(x_high)
        high_fused_fm = F.relu(low_fm+high_fm, inplace=True)

        return high_fused_fm, low_cls



def get_interp_size(input, s_factor=1, z_factor=1): # for caffe
    ori_h, ori_w = input.shape[2:]

    # shrink (s_factor >= 1)
    ori_h = (ori_h - 1) / s_factor + 1
    ori_w = (ori_w - 1) / s_factor + 1

    # zoom (z_factor >= 1)
    ori_h = ori_h + (ori_h - 1) * (z_factor - 1)
    ori_w = ori_w + (ori_w - 1) * (z_factor - 1)

    resize_shape = (int(ori_h), int(ori_w))
    return resize_shape


def interp(input, output_size, mode='bilinear'):
    n, c, ih, iw = input.shape
    oh, ow = output_size

    # normalize to [-1, 1]
    h = torch.arange(0, oh) / (oh-1) * 2 - 1
    w = torch.arange(0, ow) / (ow-1) * 2 - 1

    grid = torch.zeros(oh, ow, 2)
    grid[:, :, 0] = w.unsqueeze(0).repeat(oh, 1)
    grid[:, :, 1] = h.unsqueeze(0).repeat(ow, 1).transpose(0, 1)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1) # grid.shape: [n, oh, ow, 2]
    grid = Variable(grid)
    if input.is_cuda:
        grid = grid.cuda()

    return F.grid_sample(input, grid, mode=mode)
