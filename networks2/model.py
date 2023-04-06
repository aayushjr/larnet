# phase 3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# from networks.image_encoder import vgg16
from networks2.modifiedI3d import InceptionI3d
from networks2.vgg import Vgg16_bn
from networks2.contrastive_loss import NTXentLoss
from networks2.perceptual_loss import InceptionI3d_perceptual
from networks2.convGRU import ConvGRUCell,ConvGRU
from networks2.action_generator import ActionGenerator, WordVec
from networks2.generator import Generator
#from torchsummary import summary

"""
Pipeline:
    i1 = first frame 
    vidr = video
    i2 = action word

    frame_rep = image_encoder(i1)  512*14*14,256x28x28,128x56x56,64x112x112
    action = action_generator(word2vec(i2)) 256*14*14
    vidg = generator(frame_Rep,action) (1,768,4,14,14)  -> ([1, 3, 16, 56, 56])

    rep = I3D(i3)  256*14*14
    loss1 = l1(vidr, vidg)
    loss2 = l1(rep,action)

"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

class FullNetwork(nn.Module):
    """
    Class combining the full Video from text Synthesis network architecture.
    """

    VALID_FRAME_COUNTS = (8, 16)

    def __init__(self, output_shape,dataset="ntu" ,perceptual=False,name='Full Network', sigma=0.1, i3d_weights_path="None"):
        """
        Initializes the Full Network.
        :param output_shape: (5-tuple) The desired output shape for generated videos. Must match video input shape.
                              Legal values: (bsz, 3, 8/16, 112, 112) and (bsz, 3, 16, 112, 112)
        :param name: (str, optional) The name of the network (default 'Full Network').
        :param sigma: default value 0.2 , for noise vector
        Raises:
            ValueError: if 'output_shape' does not contain a legal number of frames.
        """
        if output_shape[2] not in self.VALID_FRAME_COUNTS:
            raise ValueError('Invalid number of frames in desired output: %d' % output_shape[2])

        super(FullNetwork, self).__init__()

        self.net_name = name
        self.output_shape = output_shape
        self.out_frames = output_shape[2]
        self.sigma = sigma
        self.dataset = dataset
        
          
        if dataset=="ntu":
            caption_file = "./data_ntu/caption.txt"
        elif dataset=="penn":
            caption_file = "./data_penn/caption.txt"
        elif dataset=="utd":
            caption_file = "./data_utd/caption.txt"
    
        with open(caption_file) as f:
            self.caption_file = f.readlines()
        self.caption_file = [line.strip() for line in self.caption_file]
        
        # specs of various features
        self.action_feat = 256
        self.frame_feat = 512
        self.rep_frames = 4
        self.rep_size = 14
        self.rep_feat = 512

        self.vgg = Vgg16_bn()
        if i3d_weights_path == "None":
            self.i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=self.out_frames, pretrained=False)
        else:
            self.i3d = InceptionI3d(final_endpoint='Mixed_5c', in_frames=self.out_frames,
                                pretrained=True, weights_path=i3d_weights_path)
        
        self.pooldown = nn.MaxPool2d(kernel_size=(2,2))
        
        # self.i3d_perceptual = InceptionI3d_perceptual(weights_path=i3d_weights_path)
        # for param in self.i3d_perceptual.parameters():
        #         param.requires_grad = False

        WV = WordVec(wv_type = 'glove')
        self.word_vector, _ = WV.get_wv()
        self.action_generator = ActionGenerator(330,self.action_feat)
        self.generator = Generator()
        
        self.perceptual_loss = perceptual
        #contrastive_loss
        # self.down1 = nn.Sequential(
        #         nn.Conv3d(256, 64, 4, 2, 1),
        #         nn.BatchNorm3d(64),
        #         nn.ReLU(inplace=True),
        #         )
        # self.down2 = nn.Linear(64*2*7*7, 1024)
        # self.contrastive_loss = NTXentLoss()
        self.mse = nn.MSELoss()
        
    def action_generator_pipeline(self,captionlist,f_start,nf):

        
        embedding = []
        captionlist = captionlist.cpu().numpy()
        f_start = f_start.cpu().numpy()
        nf = nf.cpu().numpy()
        
        for label in captionlist:
            vec = np.zeros(300)
            caption = self.caption_file[int(label)-1]
            l = caption.split(' ')
            for word in l:
                vec = np.add(vec, self.word_vector[word])
            embedding.append(vec)
#         embedding = torch.FloatTensor(embedding)
        embedding = F.normalize(torch.FloatTensor(embedding))

        noise = self.sigma * Variable(torch.FloatTensor(torch.Size([embedding.size()[0],15])).normal_(), requires_grad=False)
        pos_enc = torch.FloatTensor(f_start/nf)
        pos_enc = torch.stack([pos_enc[i].expand((15)) for i in range(pos_enc.size()[0])])
#         position_enc = torch.tensor(pos_enc).expand([75])
        embedding = torch.cat([pos_enc,embedding,noise],1)
#         embedding = torch.cat((embedding,noise),1)
        embedding = (embedding.unsqueeze_(-1)).unsqueeze_(-1)
        embedding = embedding.to('cuda')
#         embedding = torch.nn.DataParallel(embedding) 
        action_rep = self.action_generator(embedding)
        return action_rep
        # return [word.lower() for word in wordlist]
        
    def contrastive_pipeline(self,rep1,rep2):

        if(rep1.size()[2] == 8):
            #256x8x28x28 is downsampled to 256x4x14x14
            rep1 = torch.nn.functional.interpolate(rep1, size=(4, 14, 14), mode='trilinear')
            rep2 = torch.nn.functional.interpolate(rep2, size=(4, 14, 14), mode='trilinear')
        
        bsz = rep1.size()[0]
        rep1 = self.down1(rep1)
        rep1 = rep1.reshape(bsz,-1)
        rep1 = self.down2(rep1)

        rep2 = self.down1(rep2)
        rep2 = rep2.reshape(bsz,-1)
        rep2 = self.down2(rep2)
        loss = self.contrastive_loss(rep1,rep2)

        return loss

    def forward(self, img1, caption, vidr,f_start,nf,epoch):
        """
        Function to compute a single forward pass through the network, according to the architecture.
        :param img1 (tensor) first frame . size = (bsz,3,56,56)
        :param caption: (list) A list of captions. total captions = bsz
        :return: The video_syntehesized (bsz,3,16,56,56) and the action_generator output (bsz,256,14,14)
        """
        
        # this will add the word_vectors of all the words in the caption
        action_rep = self.action_generator_pipeline(caption,f_start,nf)#256*4*14*14,256*8*28*28
        frame_rep = self.vgg(img1)#512*14*14,256*28*28,128*56*56
        
        frame_rep[0] = self.pooldown(frame_rep[0])
        frame_rep[1] = self.pooldown(frame_rep[1])
        frame_rep[2] = self.pooldown(frame_rep[2])
        
        vidg = self.generator(frame_rep,action_rep)
        # print(vid.size())

        action_real = self.i3d(vidr)
        
        if(self.perceptual_loss == True):
            perceptual_loss = self.i3d_perceptual(vidg,vidr)
        
        #for some initial epochs contrastive loss should be replace by mse loss
        # if epoch>30:
        #     contrastive_loss = 0.2*(self.contrastive_pipeline(action_real[0],action_rep[1])
        #                     + self.contrastive_pipeline(action_real[1],action_rep[0]))
        # else:
        rep_loss = self.mse(action_real[0],action_rep[1])+self.mse(action_real[1],action_rep[0])
    
        if(self.perceptual_loss == True):
            return action_rep[0], vidg, action_real[1], rep_loss, perceptual_loss
        else:
            return action_rep[0], vidg, action_real[1], rep_loss#, perceptual_loss

    

if __name__ == "__main__":
    print_summary = True
