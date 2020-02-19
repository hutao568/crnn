import torch.nn as nn
from models.deform_conv import DeformConv2d
from models.attention import Attention
from models.transformlation import TPS_SpatialTransformerNetwork

class CRNN(nn.Module):
    def __init__(self,nclass,imgH=32, inp=1, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        
        num_fiducial=20
        self.Transformation=TPS_SpatialTransformerNetwork(F=num_fiducial,I_size=(32,100),I_r_size=(32,100),I_channel_num=inp)
        #conv0
        index=0
        self.conv0=nn.Sequential(nn.Conv2d(inp,nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.pool0=nn.MaxPool2d(2)
        #conv1
        index+=1
        # self.conv1=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
        #                         nn.BatchNorm2d(nm[index]),
        #                         nn.ReLU(True))
        self.conv1=nn.Sequential(DeformConv2d(nm[index-1],nm[index],3,1,1,modulation=True),
                                nn.ReLU(True))
        self.pool1=nn.MaxPool2d(2)
        #conv2
        index+=1
        self.conv2=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        #conv3
        index+=1
        self.conv3=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.pool2=nn.MaxPool2d((2,2),(2,1),(0,1))
        
        #conv4
        index+=1
        self.conv4=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))

        # self.conv4=nn.Sequential(DeformConv2d(nm[index-1],nm[index],3,1,1,modulation=True),
        #                         nn.BatchNorm2d(nm[index]),
        #                         nn.ReLU(True))

        #conv5
        index+=1
        self.conv5=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.pool3=nn.MaxPool2d((2,2),(2,1),(0,1))

        #conv6
        index+=1
        self.conv6=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],2,1,0),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))

        self.embedding = nn.Linear(nm[index], nclass)
        #hidden_size
        self.att=Attention(nm[index],256,nclass)


    def forward(self, x):
        # x=self.Transformation(x)
        x=self.conv0(x)
        x=self.pool0(x)
        x=self.conv1(x)
        x=self.pool1(x)
        x=self.conv2(x)
        x=self.conv3(x)+x
        x=self.pool2(x)
        x=self.conv4(x)
        x=self.conv5(x)+x
        x=self.pool3(x)
        x=self.conv6(x)

        x = x.squeeze(2) # batchsize x 512 x 25
        x = x.permute(0, 2, 1)  # batchsize x 25 x 512
        # conv = conv.view((-1, 512))
        output = self.embedding(x) # batchsize x 25 x class_num
        # output = output.view((-1, 26, 68))

        return output

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

if __name__ == '__main__':
    x = torch.Tensor(8, 1, 32, 100)
    y= CRNN(nclass = 69)(x)
    print(y.shape)