# https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/resnet.py#L124


import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import pdb
from utils import load_state_dict_from_url


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class ResNet(nn.Module):



    def __init__(self, block, num_blocks, num_classes=10):
        

        super(ResNet, self).__init__()
        #cameron
        self.num_classes=10
        self.construct_SUFB = False
        self.phase_two = False
        self.features = []
        self.expand = block.expansion
        self.first = True


        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        # self.linear2 = nn.Linear(512*block.expansion, 512*block.expansion)




    def setFeatureBankSize(self, featureBankSize):
        self.features = [torch.zeros(512*self.expand) for _ in range(featureBankSize)]


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _forward_impl(self, y):
        normal = True
        subBatchSize = 26
        batchSize = 0
        idx1, idx2, x, x2 = None, None, None, None
        # See note [TorchScript super()]

        if self.construct_SUFB: #cameron
            # print("should not be reached")
            (x,idx) = y
            batchSize = x.shape[0]
        elif self.phase_two:
            # print("should not be reached")
            # figure out which indices will be re made and not re made
            # move all the indices together for the ones that will be remade
            # move all the indices together for the ones that won't be remade

            # move all the target images together
            # move all the non target images together

            # whole batch
            # (x,idx) = y


            # partial batch
            (mat,idx) = y
            batchSize = mat.shape[0]

            idx1,idx2 = idx[0:subBatchSize], idx[subBatchSize:batchSize]
            x,x2 = mat[0:subBatchSize, :], mat[subBatchSize:batchSize, :]
            normal = True

            # whole batch
            # if random.random() < 0.2:
            #     normal = True
            # else:
            #     normal = False
        
        else:
            (x,idx)=y
            batchSize = x.shape[0]
            # print(idx.shape)
        
        if normal:
            out = F.relu(self.bn1(self.conv1(x)))
            if (torch.isnan(out).any()):
                print("before layer 1")
                pdb.set_trace()
            out = self.layer1(out)
            if (torch.isnan(out).any()):
                print("after layer 1")
                pdb.set_trace()
            # print(out.shape)
            out = self.layer2(out)
            if (torch.isnan(out).any()):
                print("after layer 2")
                pdb.set_trace()
            # print(out.shape)
            out = self.layer3(out)
            if (torch.isnan(out).any()):
                print("after layer 3")
                pdb.set_trace()
            # print(out.shape)
            # out = self.layer4(out)
            # out = F.avg_pool2d(out, 4)
            # out = out.view(out.size(0), -1)

        if self.construct_SUFB: # cameron
            x = out
            # print("should not be reached")
            for i in range(batchSize):
                f = x[i,:]
                self.features[idx[i]] = f
            return torch.zeros(self.num_classes)
        
        if self.phase_two:
            # print("should not be reached")
            batchFeatures = []
            # use the indices to update the list
            if normal:
                x = out
                # partial batch, for whole batch change to range(batchSize)
                for i in range(subBatchSize):
                    f = x[i,:]
                    self.features[idx[i]] = f.clone().detach()
                
            
            # use the non-target indices to get all of the non-target features
            else:
                pass
                # whole batch code
                # for index in idx:
                #     batchFeatures.append(self.features[index])
                # batchFeatures = torch.stack(batchFeatures)
                # out = batchFeatures

            # partial batch code
            for index in idx2:
                batchFeatures.append(self.features[index])
            batchFeatures = torch.stack(batchFeatures)
            # merge x with the non target features
            # still partial batch code
            out = torch.cat((x,batchFeatures))
            # print('p2:', x.shape)


        out = self.layer4(out)
        if (torch.isnan(out).any()):
            print("after layer 4")
            pdb.set_trace()
        # print(out.shape)
        out = F.avg_pool2d(out, 4)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)

        out = self.linear(out)
        # out = F.relu(out)
        # out = self.linear(out)
        # print(out.shape)
        # print('12:', x.shape)
        # print("End forward pass")
        #quit
        # quit()

        return out

    def forward(self, x):
        return self._forward_impl(x)
    
    def deactivateBN(self):
        pass
        # for m in self.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
    
    def preparePhaseTwo(self): #cameron
        self.linear.reset_parameters()
        # self.linear2.reset_parameters()

        # partial batch code, uncomment for whole batch
        # self.deactivateBN()
        
        for m in self.layer4.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()
        
        self.phase_two = True
    
    def setPhaseTwo(self, onOrOff):
        self.phase_two = onOrOff
    
    def deactivatePhaseTwo(self):
        self.phase_two = False
    
    def setConstruction(self, newVal): #cameron
        self.construct_SUFB = newVal


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

