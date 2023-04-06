
import os

import PIL
import functools
import tqdm
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import config
import data
from models import dcgan, lstm, residual, vgg

import utils
import time

if config.USE_TENSORBOARD:
    writer = SummaryWriter()

config.log_dir = os.path.join(config.log_dir, ''.join(time.ctime().split(' '))+config.model + str(config.USE_RANDOM_FRAMES))
os.makedirs(config.log_dir, exist_ok=True)


# ---------------- loading data ----------------

def video_transform(video, image_transform):
    vid = []
    for im in video:
        vid.append(image_transform(im))

    vid = torch.stack(vid).permute(1, 0, 2, 3)

    return vid

n_channels = int(config.n_channels)

image_transforms = transforms.Compose([
    PIL.Image.fromarray,
    transforms.Resize(config.image_size),
    transforms.ToTensor(),
    lambda x: x[:n_channels, ::],
    transforms.Normalize((0.5, 0.5, .5), (0.5, 0.5, 0.5)),
])

video_transforms = functools.partial(video_transform, image_transform=image_transforms)

video_length = config.video_length
image_batch = config.image_batch
video_batch = config.video_batch

dim_z = config.dim_z
dim_eps = config.dim_eps
dim_z_category = config.dim_z_category
var_eps = float(config.var_eps)
mu_train_interval = float(config.mu_train_interval)



# ------------- dataloader ---------------

dataset = data.VideoFolderDataset(config.dataset, cache=os.path.join(config.dataset, 'local.db'))
image_dataset = data.ImageDataset(dataset, image_transforms)
image_loader = DataLoader(image_dataset, batch_size=image_batch, drop_last=True, num_workers=4, shuffle=True)

video_dataset = data.VideoDataset(dataset, 16, 2, video_transforms)
video_loader = DataLoader(video_dataset, batch_size=video_batch, drop_last=True, num_workers=4, shuffle=True)


# ---------------- device ----------------

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
# device = "cpu"

# ---------------- models and optimizer ----------------

if config.model == 'dcgan':
    encoder = dcgan.encoder().to(device)
    decoder = dcgan.decoder().to(device)
elif config.model == 'vgg':
    encoder = vgg.encoder().to(device)
    decoder = vgg.decoder().to(device)
elif config.model == 'residual':
    encoder = residual.encoder().to(device)
    decoder = residual.decoder().to(device)

bilstm = lstm.bilstm().to(device)
# z0_net  = lstm.z0_net().to(device)
decoder_lstm = lstm.decoder_lstm().to(device)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=config.en_lr, betas=(config.beta1, config.beta2))
decoder_optimizer = optim.Adam(decoder.parameters(), lr=config.de_lr, betas=(config.beta1, config.beta2))
bilstm_optimizer = optim.Adam(bilstm.parameters(), lr=config.bilstm_lr, betas=(config.beta1, config.beta2))
# z0_net_optimizer = optim.Adam(z0_net.parameters(), lr=config. lr, betas=(config.beta1, config.beta2))
decoder_lstm_optimizer = optim.Adam(decoder_lstm.parameters(), lr=config.de_lstm_lr, betas=(config.beta1, config.beta2))


# ------------- losses -------------------

mse_criterion = nn.MSELoss()

def kl_criterion(mu1, logvar1, mu2, logvar2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) = 
    #   log( sqrt(
    #frame
    frames = mu1.size(0)
    sigma1 = logvar1.mul(0.5).exp() 
    sigma2 = logvar2.mul(0.5).exp() 
    kld = torch.log(sigma2/sigma1) + (torch.exp(logvar1) + (mu1 - mu2)**2)/(2*torch.exp(logvar2)) - 1/2
    return kld.sum() / (config.video_batch * frames)


# -------------- train wrapper----------------

def random_ij():
    length = np.random.randint(8,16+1)
    i = np.random.randint(0,16-length+1)
    j = i+length-1
    return (i,j)

def train_batch(x):

    bilstm.zero_grad()
    decoder_lstm.zero_grad()
    encoder.zero_grad()
    decoder.zero_grad()

    # -------- random_function ----------
    if config.USE_RANDOM_FRAMES: 
        (f_start,f_end) = random_ij()
        x = x[:,:,f_start:f_end+1,:,:]
    

    data = x.to(device)

    
    encoded = {'z': [], 'mu': [], 'logvar': []}
    for j in range(data.shape[2]):
        z, mu, logvar = encoder(data[:,:,j,:,:])
        encoded['z'].append(z)
        encoded['mu'].append(mu)
        encoded['logvar'].append(logvar)

    z, mu, logvar = bilstm(encoded['z'])
    videorep = {'z': z, 'mu': mu, 'logvar': logvar}
    
    dec_z, dec_mu, dec_logvar = decoder_lstm(videorep['z'])
    decoded = {'z': dec_z, 'mu': dec_mu, 'logvar': dec_logvar}

    frames = [decoder(frame) for frame in decoded['z']]
    frames = torch.stack(frames, dim=2)

    #losses
    frames_loss = frames
    if config.USE_RANDOM_FRAMES:
        frames_loss = frames[:,:,f_start:f_end+1,:,:]

    mse = mse_criterion(frames_loss, data)

    mu1 = torch.stack(encoded['z'])
    logvar1 = torch.stack(encoded['logvar'])
    mu2 = torch.stack(decoded['z'])
    logvar2 = torch.stack(decoded['logvar'])

    if config.USE_RANDOM_FRAMES:
        mu2 = mu2[f_start:f_end+1,:,:]
        logvar2 = logvar2[f_start:f_end+1,:,:]

    kld = kl_criterion(mu1, logvar1, mu2, logvar2)


    loss = mse + kld*config.kll_beta
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    decoder_lstm_optimizer.step()
    bilstm_optimizer.step()

    return mse.item(), kld.item()

def eval_batch(x):

    # bilstm.zero_grad()
    # decoder_lstm.zero_grad()
    # encoder.zero_grad()
    # decoder.zero_grad()

    # -------- random_function ----------
    # if config.USE_RANDOM_FRAMES: 
    #     (f_start,f_end) = random_ij()
    #     x = x[:,:,f_start:f_end+1,:,:]
    
    with torch.no_grad():
        data = x.to(device)

        encoded = {'z': [], 'mu': [], 'logvar': []}
        for j in range(data.shape[2]):
            z, mu, logvar = encoder(data[:,:,j,:,:])
            encoded['z'].append(z)
            encoded['mu'].append(mu)
            encoded['logvar'].append(logvar)

        z, mu, logvar = bilstm(encoded['z'])
        videorep = {'z': z, 'mu': mu, 'logvar': logvar}
        
        dec_z, dec_mu, dec_logvar = decoder_lstm(videorep['z'])
        decoded = {'z': dec_z, 'mu': dec_mu, 'logvar': dec_logvar}

        frames = [decoder(frame) for frame in decoded['z']]
        frames_loss = torch.stack(frames, dim=2)

        #losses
        # frames_loss = frames
        mse = mse_criterion(frames_loss, data)

        mu1 = torch.stack(encoded['z'])
        logvar1 = torch.stack(encoded['logvar'])
        mu2 = torch.stack(decoded['z'])
        logvar2 = torch.stack(decoded['logvar'])

        kld = kl_criterion(mu1, logvar1, mu2, logvar2)


        loss = mse + kld*config.kll_beta

    return mse.item(), kld.item(), frames

# -------------- Training starts -------------

logs_dict = []
n_iter = 0

for epoch in range(config.nepochs):
    encoder.train()
    decoder.train()
    bilstm.train()
    decoder_lstm.train()
    
    epoch_mse = []
    epoch_kld = []

    t = tqdm.tqdm(video_loader, desc='epoch:{} mse:{:.3f} kld:{:.3f}'.format(epoch, 0.0, 0.0), leave=True)
    for x in t:
        
        img = x["images"]
        
        mse,kld = train_batch(img)

        epoch_mse.append(mse)
        epoch_kld.append(kld)

        if config.USE_TENSORBOARD:
            writer.add_scalar('Epoch/mse', mse, epoch)
            writer.add_scalar('Epoch/kld', kld, epoch)
            writer.add_scalar('Epoch/loss', mse + config.kll_beta * kld, epoch)

        t.set_description('epoch:{} mse:{:.3f} kld:{:.3f}'.format(epoch, mse, kld))
        t.refresh()

    logs_dict.append({'epoch': epoch, 'mse': epoch_mse, 'kld': epoch_kld})
    print('[%02d] mse loss: %.5f | kld loss: %.5f (%d)' % (epoch, sum(epoch_mse)/len(video_loader), sum(epoch_kld)/len(video_loader), epoch*len(video_loader)*config.video_batch))

    #save_model
    if((epoch)%1 == 0):

        #function to generate few videos
        # x_idx = np.random.randint(0, high=(len(video_loader)-3))
        encoder.eval()
        decoder.eval()
        bilstm.eval()
        decoder_lstm.eval()
        x_val = iter(video_loader).next()
        mse, kld, frames = eval_batch(x_val["images"])
        if config.USE_TENSORBOARD:
            writer.add_scalar('Epoch/mse_val', mse, epoch)
            writer.add_scalar('Epoch/kld_val', kld, epoch)
            writer.add_scalar('Epoch/loss_val', mse + config.kll_beta * kld, epoch)
        #print(len(frames), frames[0].size())
        frames_l = [[frame for frame in batch_frame] for batch_frame in frames]
        frames_l = list(map(list, zip(*frames_l)))
        #function to save video as gifs
        for i in range(len(frames_l)):
            utils.save_gif(config.log_dir + "/vid" + str(epoch) + "-"+str(i)+".gif", frames_l[i])
        
        if((epoch+1)%10 == 0):
            torch.save({
            'encoder': encoder,
            'decoder': decoder,
            'decoder_lstm': decoder_lstm,
            'bilstm': bilstm,
            'opt': 'opt_not_implemented'},
            config.log_dir + "/model" + str(epoch) + ".pth")
            # '%s/model.pth' % config.log_dir + str(epoch))
