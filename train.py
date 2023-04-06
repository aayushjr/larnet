from data_ntu.NTUDataLoader3 import NTUDataset
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

from torch.autograd import Variable
from networks2.model import FullNetwork
from networks2.discriminator import WDiscriminator
#from networks.modifiedI3D import
#from data.ucf101dataloader import UCF101Dataset

import torch.backends.cudnn as cudnn
from utils.modelIOFuncs import get_first_frame

#from tensorboardX import SummaryWriter
import pickle
import random
import numpy as np

DATASET = 'ntu'  

# Pretrained i3d weights location here
i3d_weights_path = "None" # 'path_to_i3d_pretrained_models/rgb_charades.pt'

# data parameters
BATCH_SIZE = 16
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112

# training parameters
NUM_EPOCHS = 42
LR = 1e-4
STDEV = 0.1

useparallel = True
pretrained = True
MIN_GLOSS = 1.0
MIN_DLOSS = 1.0
pretrained_epochs = 0

device='cuda'

useTensorboard = False


# ===================================================


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters for training on {}'.format(DATASET))
    print('Batch Size: {}'.format(BATCH_SIZE))
    print('Tensor Size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Close Views: {}'.format(CLOSE_VIEWS))
    print('Total Epochs: {}'.format(NUM_EPOCHS))
    print('Learning Rate: {}'.format(LR))

def create_mapping_ucf(mapping_file="mapping_train1.pickle"):
    """
    This will delete the previuos mapping_train.pickle and make a new random mapping
    """
    data_file="/home/namanb/Vidgen_from_text/data/train16.txt"
    with open(data_file) as f:
        x = f.readlines()
    f = [line.strip() for line in x]
    
    if(os.path.isfile(mapping_file)):
        os.system("rm "+mapping_file)
    mapping = dict()
    index = 0
    total_train_samples = 9363
    numclasses = 101
    for i in range(1,numclasses+1):
        list1 = [] 
        while index<total_train_samples and f[index].endswith(' '+str(i)):
            list1.append(f[index])
            index = index + 1
        list2 = random.sample(list1, len(list1))
        random_map = dict(zip(list1,list2))
        mapping[i] = random_map 
    
    os.system("touch "+mapping_file)
    pickle_out = open(mapping_file,"wb")
    pickle.dump(mapping, pickle_out)
    
def create_mapping_ntu(mapping_file="./data_ntu/mapping_train1.pickle"):
    """
    This will delete the previuos mapping.pickle and make a new random mapping
    """
    # os.system("rm "+mapping_file)
    data_file="./data_ntu/train16.txt"
    with open(data_file) as f:
        x = f.readlines()
    f = [line.strip() for line in x]
    
    if(os.path.isfile(mapping_file)):
        os.system("rm "+mapping_file)
    
    mapping = dict()
    # index = 0
    # numclasses = 101
    cnt=10966
    i=0
    for j in range(49):
        idx = int(f[i][0:f[i].index('/')])
        # print(idx) 
        list1 = [] 
        while i<cnt and idx == int(f[i][0:f[i].index('/')]):
            # print(f[i])
            list1.append(f[i])
            i = i + 1
        list2 = random.sample(list1, len(list1))
        random_map = dict(zip(list1,list2))
        mapping[int(idx)] = random_map

    os.system("touch "+mapping_file)
    pickle_out = open(mapping_file,"wb")
    pickle.dump(mapping, pickle_out)


def get_patch(vidg,vidr,t0=4,h0=44,w0=44):
    bsz, c, t, h, w = vidg.size()
    if (t-t0) == 0:
        t1 = 0
    else:
        t1 = random.randint(0, t-t0)
    h1 = random.randint(0, h-h0)
    w1 = random.randint(0, w-w0)
    
    patchg = torch.stack([vidg[:,:,t1+i,h1:h1+h0,w1:w1+w0] for i in range(t0)], 2 )
    patchr = torch.stack([vidr[:,:,t1+i,h1:h1+h0,w1:w1+w0] for i in range(t0)], 2 )
    return patchg,patchr


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=0.1, device='cuda'):
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

def get_new_disc():
    disc = WDiscriminator()
    disc = disc.to(device)
    optimizer_D = optim.Adam(disc.parameters(), lr=LR)
    return (disc, optimizer_D)

# ----------
#  Training
# ----------
def train_model(starting_epoch):
    n_iter = 0
    min_gloss = 1000.0
    min_dloss = 1000.0
    logs_dict = []

    for epoch in range(starting_epoch, NUM_EPOCHS):  # opt.n_epochs):

        # this will generate a new random mapping from a video to another video of same action 
        # create_mapping_ntu('./data_ntu/mapping_train1.pickle')
            
        generator.train()
        discriminator.train()
        disc_rep.train()
        
        running_g_loss = 0.0
        running_recon_loss = 0.0
        running_rep_loss = 0.0
        running_d_loss = 0.0
        running_drep_loss = 0.0
        running_p_loss = 0.0
        running_total_loss =  0.0
        
        for batch_idx, (caption, vid1, frame_index, nf_v1) in enumerate(trainloader):
            vid1 = vid1.to(device)#, vid2.to(device)
            img1 = get_first_frame(vid1)#, get_first_frame(vid2)
            img1 = img1.to(device)#, img2.to(device)
            caption=caption.to(device)
            batch_size = vid1.shape[0]
            if(batch_size!=8 and batch_size!=16):
                continue
            
            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)
            
            caption = Variable(LongTensor(caption))
            valid=valid.to(device)
            fake=fake.to(device)
            caption = caption.to(device)
            frame_index = frame_index.to(device)
            nf_v1 = nf_v1.to(device)
            
            # Configure input
            real_vids_v1= Variable(vid1.type(FloatTensor))#, Variable(vid2.type(FloatTensor))
            
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()
            optimizer_Drep.zero_grad()
            
            action_rep, vidg, vidrep2, rep_loss= generator(img1,caption,vid1, frame_index, nf_v1,epoch)

            #reconstruction losses for video and rep
            recon_loss = criterion(vidg, vid1)

            # ---------------------
            #  Video Discriminator
            # ---------------------            
            fake_vid = []
            for i in range(16):
                if(random.uniform(0,1) > 0.5):
                    fake_vid.append(vidg[:,:,i,:,:])
                else:
                    fake_vid.append(real_vids_v1[:,:,i,:,:])

            fake_vid = torch.stack(fake_vid,2)

            # Loss for real videos
            real_pred = discriminator(real_vids_v1)
            d_real_loss = -real_pred.mean()

            # Loss for fake videos
            fake_pred1 = (discriminator(fake_vid) + discriminator(vidg))/2
            d_fake_loss = fake_pred1.mean()

            #gradient penalty
            gradient_penalty = (calc_gradient_penalty(discriminator,real_vids_v1,fake_vid) + 
                                    calc_gradient_penalty(discriminator,real_vids_v1,vidg))/2
            
            # Total discriminator loss
            d_loss = 0.5*(d_real_loss + d_fake_loss + gradient_penalty)
            d_loss.backward(retain_graph=True)
            
            # ---------------------
            #  Action Discriminator
            # ---------------------
            # Loss for real actions
            real_pred = disc_rep(vidrep2)
            d_real_loss = -real_pred.mean()
            # d_real_loss.backward(retain_graph=True)

            # Loss for fake actions
            fake_pred2 = disc_rep(action_rep)
            d_fake_loss = fake_pred2.mean()
            # d_fake_loss.backward(retain_graph=True)

            #gradient penalty
            gradient_penalty = calc_gradient_penalty(disc_rep,vidrep2,action_rep)
            # gradient_penalty.backward()

            # Total discriminator loss
            drep_loss = 0.5*(d_real_loss + d_fake_loss + gradient_penalty)
            drep_loss.backward(retain_graph=True)
            
            # optimizer_D.step()
        
            # -----------------
            #  Train Generator
            # -----------------

            validity = fake_pred1
            g_loss1 = -validity.mean()
            
            validity = fake_pred2
            g_loss2 = -validity.mean()  
            g_loss = g_loss1+g_loss2
            
            total_g_loss = recon_loss + 0.3*rep_loss.sum() + 0.5*g_loss #+ perceptual_loss.sum()
            total_g_loss.backward()
            
            optimizer_G.step()
            optimizer_D.step()
            optimizer_Drep.step()
            
            # --------------
            # Log Progress
            # --------------
            running_g_loss += g_loss.item()
            running_recon_loss += recon_loss.item()
            running_rep_loss += rep_loss.sum().item()
            running_d_loss += d_loss.item()
            running_drep_loss += drep_loss.item()
            #running_p_loss += perceptual_loss.sum().item()
            running_total_loss += total_g_loss.item()
            
            if (batch_idx ) % 10 == 0:
                # print('\tBatch {}/{} GLoss:{} ReconLoss:{} VPLoss:{} DLoss:{}'.format(
                print('\tBatch {}/{} ReconLoss:{} RepLoss:{} G_Loss:{} G_repLoss:{} Total_g_loss:{} Disc_loss:{} DiscrepLoss:{}'.format(
                    batch_idx + 1,
                    len(trainloader),
                    # "{0:.5f}".format(g_loss),
                    "{0:.7f}".format(recon_loss),
                    "{0:.7f}".format(rep_loss.sum()),
                    "{0:.7f}".format(g_loss1),
                    "{0:.7f}".format(g_loss2),
#                     "{0:.7f}".format(perceptual_loss),
                    "{0:.7f}".format(total_g_loss),
                    "{0:.7f}".format(d_loss),
                    "{0:.7f}".format(drep_loss)
                    ))

        if useTensorboard:
            writer.add_scalar('epoch/recon_loss',running_recon_loss / len(trainloader),epoch)
            writer.add_scalar('epoch/rep_loss',running_rep_loss / len(trainloader),epoch)
            writer.add_scalar('epoch/g_loss',running_g_loss / len(trainloader) ,epoch)
            writer.add_scalar('epoch/d_loss',running_d_loss / len(trainloader) ,epoch)
            writer.add_scalar('epoch/total_g_loss',running_total_loss / len(trainloader) ,epoch)

        logs_dict.append({'epoch': epoch, 'recon_loss': running_recon_loss / len(trainloader), 
                'rep_loss': running_rep_loss / len(trainloader),'g_loss': running_g_loss / len(trainloader),
                          'd_loss': (running_d_loss + running_drep_loss) / len(trainloader),
                          'total_loss': running_total_loss / len(trainloader)})
  
        print('Training Epoch {}/{} ReconLoss:{} RepLoss:{} g_loss:{} d_loss:{}  Perceptual_Loss:{} Total_g_loss:{}'.format(
            epoch + 1,
            NUM_EPOCHS,
            "{0:.7f}".format((running_recon_loss / len(trainloader))),
            "{0:.7f}".format((running_rep_loss / len(trainloader))),
            "{0:.7f}".format((running_g_loss / len(trainloader))),
            "{0:.7f}".format( (running_d_loss / len(trainloader) + running_drep_loss / len(trainloader))),
            "{0:.7f}".format((running_p_loss / len(trainloader))),
            "{0:.7f}".format((running_total_loss / len(trainloader)))))
    
        
        avg_gloss = ((running_recon_loss)/len(trainloader))# + ((running_rep_loss)/len(trainloader)) 
        print("Average loss:{}".format("{0:.7f}".format(avg_gloss)))
        if (epoch+1)%5 == 0:
            min_gloss = avg_gloss
            gen_weight_file = './weights_ntu2/net_gen7_final_contrastive_ntu_{}_{}_{}_{}_{}_{}.pt'.format(epoch,BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                      NUM_EPOCHS, LR)
            torch.save(generator.module.state_dict(), gen_weight_file)
            
        avg_dloss = running_d_loss / len(trainloader)
        if (epoch+1)%5 == 0:
            min_dloss = avg_dloss
            disc_weight_file = './weights_ntu2/net_discv7_final_contrastive_ntu_{}_{}_{}_{}_{}_{}.pt'.format(epoch,BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                      NUM_EPOCHS, LR)
            torch.save(discriminator.module.state_dict(), disc_weight_file)
            disc_weight_file = './weights_ntu2/net_discr7_final_contrastive_ntu_{}_{}_{}_{}_{}_{}.pt'.format(epoch,BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                      NUM_EPOCHS, LR)
            torch.save(disc_rep.module.state_dict(), disc_weight_file)
            
    print(logs_dict)
    if(not os.path.isfile("./log_dict_final.pkl")):
        os.system("touch log_dict_final.pkl")
    with open('log_dict_final.pkl', 'wb') as f:
        pickle.dump(logs_dict, f)
    print(min_gloss)


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the generator and data loaders. Then, it trains the generator.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if useTensorboard:
        writer = SummaryWriter()

    # generator
    weights_path = "./weights_ntu/net_gen_final_contrastive_ntu_69_16_16_2_90_0.0001.pt"
    generator = FullNetwork(output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH),dataset="ntu", i3d_weights_path=i3d_weights_path)
    # generator.load_state_dict(torch.load(weights_path))
    generator = generator.to(device)
    generator =  torch.nn.DataParallel(generator)
    cudnn.benchmark=True
    
    # discriminator
    weights_path = "./weights_ntu/net_discv_final_contrastive_ntu_69_16_16_2_90_0.0001.pt"
    discriminator = WDiscriminator()
    #discriminator.load_state_dict(torch.load(weights_path))
    discriminator = discriminator.to(device)
    discriminator = torch.nn.DataParallel(discriminator)
 
    weights_path = "./weights_ntu/net_discr_final_contrastive_ntu_69_16_16_2_90_0.0001.pt"
    disc_rep = WDiscriminator(inchannels=256)
    #disc_rep.load_state_dict(torch.load(weights_path))
    disc_rep = disc_rep.to(device)
    disc_rep = torch.nn.DataParallel(disc_rep)

    # Loss functions
    criterion = nn.MSELoss()
    adversarial_loss = nn.BCELoss()
    # categorical_loss = torch.nn.CrossEntropyLoss()
    # continuous_loss = torch.nn.MSELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=LR)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR//2)
    optimizer_Drep = optim.Adam(disc_rep.parameters(), lr=LR//2)

    if DATASET.lower() == 'ucf101':
        data_root_dir = "/datasets/UCF-101/Frames/frames-128x128"
        train_split = "./data/train2.txt"

        #trainset = UCF101Dataset(root_dir=data_root_dir, data_file=train_split)
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    elif DATASET.lower() == 'ntu':
    
        data_root_dir = "/datasets/NTU-ARD/frames-240x135"
        train_split = "./data_ntu/train16.txt"

        trainset = NTUDataset(root_dir=data_root_dir, data_file=train_split)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        
    else:
        print('This network has only been set up to train on the UCF101/NTU dataset.')

    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor
    
    starting_epoch = 0
    train_model(starting_epoch)
