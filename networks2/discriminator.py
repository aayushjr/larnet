import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size=3, stride=1, padd=1):
        super(ConvBlock,self).__init__()
        self.add_module('conv',nn.Conv3d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        self.add_module('norm',nn.BatchNorm3d(out_channel)),
        self.add_module('LeakyRelu',nn.LeakyReLU(0.2, inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#singan discriminator
class WDiscriminator(nn.Module):
    def __init__(self,inchannels=3,N=64):
        super(WDiscriminator, self).__init__()
        ker_size=3
        padd_size=1
        stride=1

        self.head = ConvBlock(inchannels,N,ker_size,padd_size,stride)
        self.body = nn.Sequential()
        for i in range(3):
            N = int(N/2)
            block = ConvBlock(2*N,N,ker_size,padd_size,stride)
            self.body.add_module('block%d'%(i+1),block)
        if(inchannels == 3):
            self.tail = ConvBlock(N,3,ker_size,padd_size,stride)
        else:
            self.tail = ConvBlock(N,N,ker_size,padd_size,stride)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)        
        return x

#to be used if there is classification loss
class WDiscriminator_video(nn.Module):
    def __init__(self,inchannels=3):
        super(WDiscriminator_video, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        N = 32
        self.head = ConvBlock(inchannels,N)
        ker_size=3
        padd_size=1
        
        self.classify1 = ConvBlock(N,3,4,2,1)
        self.classify2 = ConvBlock(N,3,4,2,1)
        self.classify3 = ConvBlock(int(N/2),3,4,2,1)
        self.fc = nn.Linear(int(N/2)*98,49)
        self.sigmoid = nn.Sigmoid()
        
        self.body = nn.Sequential()
        for i in range(2):
            N = int(N/2)
            block = ConvBlock(2*N,N,ker_size,padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        # self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)
        self.tail = ConvBlock(N,3,4,2,1)
        
        
        
        #32x2x7x7
    def forward(self,x):
        head = self.head(x)
        x = self.body(head)
        x = self.tail(x)
        
        label = self.classify1(head)
        label = self.classify2(label)
        label = self.classify3(label)
        bsz = label.size()[0]
        label = label.reshape(bsz,-1)
        label = self.sigmoid(self.fc(label))
        
        return x,label

#to be used if there is classification loss
class WDiscriminator_action(nn.Module):
    def __init__(self,inchannels=256):
        super(WDiscriminator_action, self).__init__()
        # self.is_cuda = torch.cuda.is_available()
        N = 64
        self.head = ConvBlock(inchannels,N)
        ker_size=3
        padd_size=1
        self.classify1 = ConvBlock(N,3,3,1,1)
        self.classify2 = ConvBlock(int(N/2),3,4,2,1)
        #32x2x7x7
        self.fc = nn.Linear(int(N/2)*98,49)
        self.sigmoid = nn.Sigmoid()
        
        self.body = nn.Sequential()
        for i in range(2):
            N = int(N/2)
            block = ConvBlock(2*N,N,ker_size,padd_size,1)
            self.body.add_module('block%d'%(i+1),block)
        # self.tail = nn.Conv2d(max(N,opt.min_nfc),1,kernel_size=opt.ker_size,stride=1,padding=opt.padd_size)
        self.tail = ConvBlock(N,N,4,2,1)
        
       
        
    def forward(self,x):
        head = self.head(x)
        x = self.body(head)
        x = self.tail(x)
        
        label = self.classify1(head)
        label = self.classify2(label)
        bsz = label.size()[0]
        label = label.reshape(bsz,-1)
        label = self.sigmoid(self.fc(label))
        
        return x, label



def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    #print real_data.size()
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device)#cuda() #gpu) #if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)


    interpolates = interpolates.to(device)#.cuda()
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


if __name__ == "__main__":
    print_summary = True

    # gen = Generator(in_channels= 768, out_frames=16).to('cpu')
    # bsz=4
    x = torch.randn(1,256,4,14,14)
    disc = WDiscriminator(inchannels=256)
    y = disc(x)
    # y = x.resize(3,4,14,14)
    # y = F.interpolate(x, size=(4,14,14), mode='trilinear')
    print(y.size())
    # if print_summary:
        # summary(gen, input_size=(1, 4, 14, 14))
