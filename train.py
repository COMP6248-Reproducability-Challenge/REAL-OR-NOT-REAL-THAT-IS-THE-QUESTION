import os
import numpy as np
import random
import time
import numpy
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader as DataLoader
from model.loss import learnD_Realness, learnG_Realness, KLLoss
from torch.nn.init import normal_


seed=1
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"
name = "RealnessGAN-CelebA256-"+str(seed)
lr_D = 0.0002
lr_G = 0.0002
beta1 = 0.5
beta2 = 0.999
gen_every = 10000
print_every = 1000
gen_extra_images = 5000
G_h_size = 32
D_h_size = 32
z_size = 128
dataset = 'dataset/CelebA/'
target_image_size = 256
total_iters = 520000
num_workers = 4
batch_size = 32
positive_skew = 1.0
negative_skew = -1.0
num_outcomes = 51
use_adaptive_reparam = True
relativisticG = True
output_folder = './OUTPUT/'+name
n_channels = 3


start = time.time()
title = 'SEED-%d' % seed

if gen_extra_images > 0 and not os.path.exists(f"{output_folder}"):
    os.mkdir(f"{output_folder}")

to_img = transforms.ToPILImage()
torch.utils.backcompat.broadcast_warning.enabled=True

random.seed(seed)
numpy.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

trans = transforms.Compose([
    transforms.Resize((target_image_size, target_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

data = datasets.ImageFolder(root=dataset, transform=trans)

class DataProvider:
    def __init__(self, data, batch_size):
        self.data_loader = None
        self.iter = None
        self.batch_size = batch_size
        self.data = data
        self.data_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        self.build()
    def build(self):
        self.iter = iter(self.data_loader) 
    def __next__(self):
        try:
            return self.iter.next()
        except StopIteration:
            self.build()
            return self.iter.next()

random_sample = DataProvider(data, batch_size)

from model.DCGAN_model import DCGAN_G, DCGAN_D
G = DCGAN_G(target_image_size, z_size,G_h_size,n_channels)
D = DCGAN_D(target_image_size, D_h_size,n_channels,num_outcomes,use_adaptive_reparam)
print('Using feature size of '+str(num_outcomes))
KL_divergence = KLLoss(atoms=num_outcomes, v_max=positive_skew, v_min=negative_skew)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        normal_(m.weight.data,0,0.02)
    elif classname.find('BatchNorm') != -1:
        normal_( m.weight.data,1,0.02)
        m.bias.data.fill_(0)

print("Initialized weights")
G.apply(weights_init)
D.apply(weights_init)

G = G.to(device)
D = D.to(device)
KL_divergence.to(device)
x = torch.FloatTensor(batch_size, n_channels, target_image_size, target_image_size).to(device)
optimizerD = torch.optim.Adam(D.parameters(), lr=lr_D, betas=(beta1, beta2), eps=1e-08)
optimizerG = torch.optim.Adam(G.parameters(), lr=lr_G, betas=(beta1, beta2))
decayD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=1)
decayG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=1)

current_set_images = 0

print(G)
print(D)

gauss = np.random.normal(0, 0.1, 1000)
count, bins = np.histogram(gauss, num_outcomes)
anchor0 = count / sum(count)

unif = np.random.uniform(-1, 1, 1000)
count, bins = np.histogram(unif, num_outcomes)
anchor1 = count / sum(count)

for i in range(total_iters):
    print('***** start training iter %d *******'%i)
    D.train()
    G.train()
    images, _ = random_sample.__next__()
    x.copy_(images)
    del images
    z1 = torch.FloatTensor(batch_size, z_size, 1, 1)
    z1 = z1.to(device)
    z2 = torch.FloatTensor(batch_size, z_size, 1, 1)
    z2 = z2.to(device)
    anchor_real = torch.zeros((x.shape[0], num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor1, dtype=torch.float).to(device)
    anchor_fake = torch.zeros((x.shape[0], num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor0, dtype=torch.float).to(device)
    lossD = learnD_Realness(negative_skew, positive_skew, D, G, optimizerD, KL_divergence, x, z1, anchor_real, anchor_fake)
    lossG = learnG_Realness(negative_skew, positive_skew, relativisticG, D, G, optimizerG, KL_divergence, x, z2, anchor_real, anchor_fake)

    decayD.step()
    decayG.step()

    if i < 1000 or (i+1) % 100 == 0:
        if (lossD is not None) and (lossG is not None):
            diff = -lossD.data.item() + lossG.data.item()
        else:
            diff=-1.0
        s = '['+str(i+1)+'/'+str(total_iters)+' seed: '+str(seed)+' Diff: '+str(diff)+' loss_D: '+str(lossD.data.item())+' loss_G: '+str(lossG.data.item())
        print(s)

    if (i+1) % gen_every == 0:
        current_set_images += 1
        if not os.path.exists('%s/models/' % (output_folder)):
            os.mkdir('%s/models/' % (output_folder))
        torch.save({
            'i': i + 1,
            'current_set_images': current_set_images,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'G_optimizer': optimizerG.state_dict(),
            'D_optimizer': optimizerD.state_dict(),
            'G_scheduler': decayG.state_dict(),
            'D_scheduler': decayD.state_dict(),
        }, '%s/models/state_%02d.pth' % (output_folder, current_set_images))
        print('Model saved.')

        if os.path.exists('%s/%01d/' % (output_folder, current_set_images)):
            for root, dirs, files in os.walk('%s/%01d/' % (output_folder, current_set_images)):
                for f in files:
                    os.unlink(os.path.join(root, f))
        else:
            os.mkdir('%s/%01d/' % (output_folder, current_set_images))

        G.eval()
        extra_batch = 100 if target_image_size <= 256 else batch_size
        with torch.no_grad():
            ext_curr = 0
            z_extra = torch.FloatTensor(extra_batch, z_size, 1, 1)
            z_extra = z_extra.to(device)
            for ext in range(int(gen_extra_images/extra_batch)):
                fake_test = G(z_extra.normal_(0, 1)) 
                    
                for ext_i in range(fake_test.size(0)):
                    vutils.save_image((fake_test[ext_i]*.50)+.50, '%s/%01d/fake_samples_%05d.png' % (output_folder, current_set_images, ext_curr),
                        normalize=False, padding=0)
                    ext_curr += 1
            del z_extra
            del fake_test
        G.train()
        print('Finished generating extra samples at iteration %d'%((i+1)))
        







    
    

    






