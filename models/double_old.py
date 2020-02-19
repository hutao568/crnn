import torch.nn as nn
import torch
from models.cbam import *
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, nclass, leakyRelu=False):
        super(CRNN, self).__init__()

        nc = 1
        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        #nm = [64, 128, 256, 512, 1024, 2048, 4096]
        #nm = [16, 32, 64, 64, 128, 128, 128]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # batchsizex64x16x50
        convRelu(1)
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2) )  # batchsizex256x4x25

        convRelu(4, True)
        convRelu(5)
        convRelu(6, True)

        self.cbam1 = CBAM(nm[-1])
        self.cbam2 = CBAM(nm[-1])

        self.embedding = nn.Linear(512, nclass)
        #self.w1 = nn.Linear(26, 128)
        #self.w2 = nn.Linear(128, 26)
        self.cnn = cnn

    def forward(self, input):

        # conv features
        conv = self.cnn(input)

        x_compress = torch.cat( (torch.max(conv,1, keepdim=True)[0], torch.mean(conv,1, keepdim = True)), dim=1 )
        conv_up = self.cbam1(conv, x_compress)
        conv_down = self.cbam2(conv, x_compress)

        conv = torch.cat((conv_up, conv_down), dim = 3)

        y = nn.MaxPool2d((12, 2), (12, 2))(conv).permute(0,2,3,1).contiguous()
        y = y.view((-1, 512))
        output = self.embedding(y)
        output = output.view((-1, 25, 69))

        output = F.softmax(output, dim = 2)
        return output
    
if __name__ == '__main__':
    x = torch.Tensor(8, 1, 32, 100)
    y = CRNN(nclass = 69)(x)
