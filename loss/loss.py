import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.nn as nn

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class ClipStyleContrastiveLoss(nn.Module):
    def __init__(self, tau_plus=0.1, beta=1.0, estimator='hard', temperature=0.5):
        super().__init__()
        self.tau_plus = tau_plus
        self.beta = beta
        self.estimator = estimator
        self.temperature = temperature

    def _get_positive_pairs(self, logits):#取logits对角线作正样本
        return torch.cat([logits.diag(), logits.diag()])  # [2B] 1 4 1 4

    def _get_negative_samples(self, logits):
        batch_size = logits.size(0)
        device = logits.device
        
        sim_matrix = torch.cat([                                   #1 3      1 3 1 2  mask 0 1 0 1   3 2 2 3 2 3 3 2 ---> 3 2
            torch.cat([logits, logits.t()], dim=1),                #2 4      2 4 3 4       1 0 1 0                        2 3
            torch.cat([logits.t(), logits], dim=1)                 #         1 2 1 3       0 1 0 1                        3 2
        ], dim=0)                                                  #         3 4 2 4       1 0 1 0                        2 3
        
        mask = torch.ones(2*batch_size, 2*batch_size, dtype=torch.bool, device=device)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size+i] = 0
            mask[batch_size+i, i] = 0
        
        return sim_matrix.masked_select(mask).view(2*batch_size, -1)  # [2B x (2B-2)]

    def forward(self, logits, ground_truth=None):
        batch_size = logits.size(0)
        pos = self._get_positive_pairs(logits)  # [2B]
        pos = torch.exp(pos / self.temperature)
        
        neg_matrix = self._get_negative_samples(logits)  # [2B x (2B-2)]
        neg = torch.exp(neg_matrix / self.temperature)
        
        if self.estimator == 'hard':
            N = 2 * batch_size - 2
            imp = (self.beta * neg.log()).exp()
            reweight_neg = (imp * neg).sum(dim=1) / imp.mean(dim=1)
            Ng = (-self.tau_plus * N * pos + reweight_neg) / (1 - self.tau_plus)
            Ng = torch.clamp(Ng, min=N * torch.exp(torch.tensor(-1/self.temperature).cuda()))
        else:
            Ng = neg.sum(dim=1)
        
        loss = (-torch.log(pos / (pos + Ng + 1e-8))).mean()
        return loss






class ClipInfoCELoss(_Loss):
    def __init__(self):
        super(ClipInfoCELoss, self).__init__()

    def forward(self, logits, ground_truth):
       # labels = torch.arange(len(logits)).to(device)
       # print('labels:\n', labels.size(), '\n', labels)
       # print('logits:\n', logits.size(), '\n', logits)
       loss_i = F.cross_entropy(logits, ground_truth)
       loss_t = F.cross_entropy(logits.t(), ground_truth)
       loss = (loss_i+loss_t)/2
       # print('loss_i:', loss_i, '\nloss_t:', loss_t, '\ntotal_loss:', loss)
       return loss


def pixel_loss(out, target):
    loss = F.l1_loss(out, target, reduction="mean")
    # loss = F.mse_loss(out, target)
    return loss


def gan_g_loss(fake_font, fake_uni):
    g_loss = -(fake_font.mean() + fake_uni.mean())
    return g_loss


def gan_d_loss(real_font, real_uni, fake_font, fake_uni):
    d_loss = (F.relu(1. - real_font).mean() + F.relu(1. + fake_font).mean()) + \
             F.relu(1. - real_uni).mean() + F.relu(1. + fake_uni).mean()
    return d_loss

if __name__ == "__main__":
    logits=torch.randn(32,32)
    criterion = ClipStyleContrastiveLoss()
    loss = criterion(logits, ground_truth=None)
    print(loss)