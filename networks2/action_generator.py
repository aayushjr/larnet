import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import re
import os
import numpy as np
#from gensim.models import word2vec, KeyedVectors
import codecs
import math
from six.moves import cPickle as pickle

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class WordVec(object):
  ''' generate word vectors
  '''
  def __init__(self, wv_type='glove'):
    '''
      argument
       wv_type: one of {'self', 'google', 'glove'}
         self:   use movie review dataset to train word embedding
         google: google pretrained 300 dims wordvec
         glove:  glove 100 dims wordvec trained by Wikipedia
    '''
    self.wv_type = wv_type
    self.wv, self.dims = self.get_wv()

  def get_wv(self):
    
    if self.wv_type == 'self':
      word_vector = KeyedVectors.load_word2vec_format('/home/nbiyani/Desktop/UCF_work/wordvec/word_vector.bin', binary=True).wv
      dims = 100
    elif self.wv_type == 'google':
      # load Google's pre-trained Word2Vec model.
      word_vector = KeyedVectors.load_word2vec_format('/home/nbiyani/Desktop/UCF_work/wordvec/GoogleNews-vectors-negative300.bin', binary=True).wv
      dims = 300
    elif self.wv_type == 'glove':
      if not os.path.exists('/home/namanb/wordvec/glove-6B.300d.pkl'):
        glove = open("/data/namanb/wordvec/glove.6B/glove.6B.300d.txt", "r").read().split("\n")[:-1]
        word_vector = {line.split()[0]: np.array(line.split()[1:]).astype(np.float32) for line in glove}
        pickle.dump(word_vector, open('/data/namanb/wordvec/glove-6B.300d.pkl', "wb", True))
      word_vector = pickle.load(open('/home/namanb/wordvec/glove-6B.300d.pkl', "rb", True))
      dims = 300
    return word_vector, dims

  def get_dim(self):
    return self.dims

class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, dim=2, use_noise=True, sigma=0.2):
        super(dcgan_upconv, self).__init__()
        if(dim==2):
          self.main = nn.Sequential(
                # Noise(use_noise, sigma),
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.ReLU(inplace=True),
                )
        else:
          self.main = nn.Sequential(
                nn.ConvTranspose3d(nin, nout, 4, 2, 1),
                nn.BatchNorm3d(nout),
                nn.ReLU(inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class ActionGenerator(nn.Module):
    """
    Class representing the Generator network to be used.
    """

    def __init__(self, in_channels, out_channels, gen_name="ActionGenerator"):
        super(ActionGenerator, self).__init__()
        
        self.gen_name = gen_name
        self.in_channels = in_channels
        self.out_channels = out_channels
        layer_out_channels = {'upconv1': 256,
                              'upconv2': 256,
                              'upconv3': 256,
                              'upconv4': 256,
                              'upconv5': out_channels
                            }
        layer_in_channels = {'upconv1': in_channels,
                              'upconv2': layer_out_channels['upconv1'],
                              'upconv3': layer_out_channels['upconv2'],
                              'upconv4': layer_out_channels['upconv3'],
                              'upconv5': layer_out_channels['upconv4']
                            }
        layer = 'upconv1' 
        self.upconv1 = dcgan_upconv(layer_in_channels[layer],layer_out_channels[layer])        
        
        layer = 'upconv2' 
        self.upconv2 = dcgan_upconv(layer_in_channels[layer],layer_out_channels[layer])        
        
        layer = 'upconv3' 
        self.upconv3 = dcgan_upconv(layer_in_channels[layer],layer_out_channels[layer],3)        
        
        layer = 'upconv4' 
        self.upconv4 = dcgan_upconv(layer_in_channels[layer],layer_out_channels[layer],3)      
        
        self.upsample1 = nn.Upsample(scale_factor=2,mode='trilinear')
        
        #to convert from (16,16) to (14,14)
        layer = 'upconv5' 
        self.upconv5 =  nn.Sequential(
                        nn.Conv3d(in_channels = layer_in_channels[layer],out_channels = layer_out_channels[layer],
                                        kernel_size=(1,3,3), stride=1),
                        nn.BatchNorm3d(out_channels),
                        nn.ReLU( inplace=True))

        self.upconv6 = dcgan_upconv(layer_in_channels[layer],layer_out_channels[layer],3)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # print("weights initialized")

    def forward(self, embedding):

        bs = embedding.size()[0] 
        x = self.upconv1(embedding)
        x = self.upconv2(x)
        x = self.upconv3(x.unsqueeze(2))
        x = self.upconv4(x)
        x = self.upconv5(x)
        #x = self.upsample1(x)
        x2 = self.upconv6(x)
        return [x,x2]

if __name__ == '__main__':

  x = torch.randn(2,330,1,1)
  gen = ActionGenerator(330,256)
  y = gen(x)
  print(y[0].size())
  print(y[1].size())


