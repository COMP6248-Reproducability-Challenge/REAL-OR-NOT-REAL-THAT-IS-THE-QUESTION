import torch
import torch.nn as nn

class KLLoss(nn.Module):
    def __init__(self, atoms=51, v_max=10, v_min=-10):
        super(KLLoss, self).__init__()
        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms)
        self.delta = (v_max - v_min) / (atoms - 1)

    def to(self, device):
        self.device = device
        self.supports = self.supports.to(device)

    def forward(self, anchor, feature, skewness=0.0):
        batch_size = feature.shape[0]
        skew = torch.zeros((batch_size, self.atoms)).to(self.device).fill_(skewness)

        Tz = skew + self.supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1).to(self.device)
        for i in range(Tz.size()[0]):
            for j in range(Tz.size()[1]):
                if Tz[i, j] > self.v_max:
                    Tz[i, j] = self.v_max
                if Tz[i, j] < self.v_min:
                    Tz[i, j] = self.v_min
        b = (Tz - self.v_min) / self.delta
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms).to(self.device)
        skewed_anchor = torch.zeros(batch_size, self.atoms).to(self.device)
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))  
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))  

        loss = nn.functional.kl_div((1e-16+feature).log(), skewed_anchor, reduction='batchmean')

        return loss


def learnD_Realness(negative_skew, positive_skew, D, G, optimizerD, KL_divergence, x, z, anchor_real, anchor_fake):
    for p in D.parameters():
        p.requires_grad = True

    D.zero_grad()
    optimizerD.zero_grad()

    feat_real = D(x).log_softmax(1).exp()

    z.normal_(0, 1)
    imgs_fake = G(z)
    feat_fake = D(imgs_fake.detach()).log_softmax(1).exp()

    lossD_real = KL_divergence(anchor_real, feat_real, skewness=positive_skew)
    lossD_real.backward()

    lossD_fake = KL_divergence(anchor_fake, feat_fake, skewness=negative_skew)
    lossD_fake.backward()

    lossD = lossD_real + lossD_fake

    optimizerD.step()

    return lossD


def learnG_Realness(negative_skew, positive_skew, relativisticG, D, G, optimizerG, KL_divergence, x, z, anchor_real,
                    anchor_fake):
    G.train()
    for p in D.parameters():
        p.requires_grad = False

    G.zero_grad()
    optimizerG.zero_grad()

    feat_real = D(x).log_softmax(1).exp()

    z.normal_(0, 1)
    imgs_fake = G(z)
    feat_fake = D(imgs_fake).log_softmax(1).exp()

    if relativisticG:
        lossG = -KL_divergence(anchor_fake, feat_fake, skewness=negative_skew) + KL_divergence(feat_real, feat_fake)
    else:
        lossG = -KL_divergence(anchor_fake, feat_fake, skewness=negative_skew) + KL_divergence(anchor_real, feat_fake,
                                                                                             skewness=positive_skew)
    lossG.backward()

    optimizerG.step()

    return lossG