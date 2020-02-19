import torch.nn as nn
import torch
from models.cbam import *
import torch.nn.functional as F
from models.deform_conv import DeformConv2d

class CRNN(nn.Module):
    def __init__(self, nclass, leakyRelu=False):
        super(CRNN, self).__init__()

        self.nclass=nclass
        nc = 1
        ks = [3, 3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        inp=1

        #conv0
        index=0
        self.conv0=nn.Sequential(nn.Conv2d(inp,nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.pool0=nn.MaxPool2d(2)
        #conv1
        index+=1
        self.conv1=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.conv1=nn.Sequential(DeformConv2d(nm[index-1],nm[index],3,1,1,modulation=True),
                                 nn.ReLU(True))
        self.pool1=nn.MaxPool2d(2)
        
        self.cbam1 = CBAM(nm[-1])
        self.cbam2 = CBAM(nm[-1])
        
        #conv2
        index+=1
        self.conv2_1=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.conv2_2=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        #conv3
        index+=1
        self.conv3_1=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.conv3_2=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.pool2=nn.MaxPool2d((2,2),(2,1),(0,1))
        
        #conv4
        index+=1
        self.conv4_1=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.conv4_2=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))

        # self.conv4=nn.Sequential(DeformConv2d(nm[index-1],nm[index],3,1,1,modulation=True),
        #                         nn.BatchNorm2d(nm[index]),
        #                         nn.ReLU(True))

        #conv5
        index+=1
        self.conv5_1=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.conv5_2=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],3,1,1),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.pool3=nn.MaxPool2d((2,2),(2,1),(0,1))

        #conv6
        index+=1
        self.conv6_1=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],2,1,0),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))
        self.conv6_2=nn.Sequential(nn.Conv2d(nm[index-1],nm[index],2,1,0),
                                nn.BatchNorm2d(nm[index]),
                                nn.ReLU(True))

        self.embedding_1 = nn.Linear(nm[index], nclass)
        self.embedding_2 = nn.Linear(nm[index], nclass)
        

    def forward(self, x):
        
        x=self.conv0(x)
        x=self.pool0(x)
        x=self.conv1(x)
        x=self.pool1(x)
        x_compress = torch.cat( (torch.max(x,1, keepdim=True)[0], torch.mean(x,1, keepdim = True)), dim=1 )
        x1=self.cbam1(x,x_compress)
        x2=self.cbam2(x,x_compress)

        x1=self.conv2_1(x1)
        x1=self.conv3_1(x1)+x1
        x1=self.pool2(x1)
        x1=self.conv4_1(x1)
        x1=self.conv5_1(x1)+x1
        x1=self.pool3(x1)
        x1=self.conv6_1(x1)

        x2=self.conv2_2(x2)
        x2=self.conv3_2(x2)+x2
        x2=self.pool2(x2)
        x2=self.conv4_2(x2)
        x2=self.conv5_2(x2)+x2
        x2=self.pool3(x2)
        x2=self.conv6_2(x2)
        
        x1 = x1.squeeze(2) # batchsize x 512 x 25
        x1 = x1.permute(0, 2, 1)  # batchsize x 25 x 512
        output1 = self.embedding_1(x1) # batchsize x 25 x class_num

        x2 = x2.squeeze(2) # batchsize x 512 x 25
        x2 = x2.permute(0, 2, 1)  # batchsize x 25 x 512
        output2 = self.embedding_2(x2) # batchsize x 25 x class_num
        output=torch.cat((output1,output2),dim=1)
        return output
    
if __name__ == '__main__':
    x = torch.Tensor(8, 1, 32, 100)
    y = CRNN(nclass = 69)(x)
