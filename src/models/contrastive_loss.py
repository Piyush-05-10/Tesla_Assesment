import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size = z1.shape[0]
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        representations = torch.cat([z1, z2], dim=0)
        similarity = torch.mm(representations, representations.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z1.device).bool()
        similarity.masked_fill_(mask, -9e15)

        pos_sim = torch.sum(z1 * z2, dim=1) / self.temperature
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

        loss = -pos_sim + torch.logsumexp(similarity, dim=1)
        return loss.mean()
