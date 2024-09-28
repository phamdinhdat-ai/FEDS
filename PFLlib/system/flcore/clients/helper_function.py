import torch 
import torch.nn as nn 
import torch.nn.functional as F 



# unsupervised contrastive loss

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, x1, x2):
        x1 = F.normalize(x1, dim=1)
        x2 = F.normalize(x2, dim=1)
        batch_size = x1.size(0)
        out = torch.cat([x1, x2], dim=0)
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(x1 * x2, dim=-1) / self.temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss 
#relational knowledge distillation loss

class RKDLoss(nn.Module):
    def __init__(self, t_1 = 0.1, t_2 = 0.01):
        super(RKDLoss, self).__init__()
        self.t_1 = t_1
        self.t_2 = t_2
                
    
    def forward(self, z1, z2, za):
        
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        za = F.normalize(za, dim=1)
        
        N = z1.size(0)
        sim_1  = torch.mm(z1, za.t().contiguous())
        sim_2  = torch.mm(z2, za.t().contiguous())
        
        inputs1 = sim_1 / self.t_1
        inputs2 = sim_2 / self.t_2
        targets = (F.softmax(sim_1, dim=1) + F.softmax(sim_2, dim=1)) / 2
        
        js_div1 = F.kl_div(F.log_softmax(inputs1, dim=1), F.softmax(targets, dim=1), reduction='batchmean')
        js_div2 = F.kl_div(F.log_softmax(inputs2, dim=1), F.softmax(targets, dim=1), reduction='batchmean')
        
        return (js_div1 + js_div2) / 2.0 
        
    
    
# #contrastiveloss
# import torch
# import torch.nn.functional as F

# class ContrastiveLoss(torch.nn.Module):

#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         cosin_sim = F.cosine_similarity(output1, output2)
#         pos =  (1-label) * torch.pow(cosin_sim, 2)
#         neg = (label) * torch.pow(torch.clamp(self.margin - cosin_sim, min=0.0), 2)
#         loss_contrastive = torch.mean( pos + neg )
#         return loss_contrastive   
        
        
        

