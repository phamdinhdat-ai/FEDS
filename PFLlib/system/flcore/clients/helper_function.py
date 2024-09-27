import torch 
import torch.nn as nn 
import torch.nn.functional as F 



# supervisecontrastive loss 
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features,  labels):
        
        
        device  = features.device
        batch_size = features.shape[0]
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.T)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        labels_qual = labels == labels.T 
        
        positives_mask = labels_qual & ~mask
        
        negatives_mask = ~labels_qual
        positives = torch.exp(similarity_matrix / self.temperature) * positives_mask.float()
        positives = torch.sum(positives, dim=1)
        
        negatives = torch.exp(similarity_matrix / self.temperature) * negatives_mask.float()
        negatives = torch.sum(negatives, dim=1)
        
        loss = -torch.log(positives / (positives + negatives))
        
        return loss.mean()
    
    

    
    
#contrastiveloss
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosin_sim = F.cosine_similarity(output1, output2)
        pos =  (1-label) * torch.pow(cosin_sim, 2)
        neg = (label) * torch.pow(torch.clamp(self.margin - cosin_sim, min=0.0), 2)
        loss_contrastive = torch.mean( pos + neg )
        return loss_contrastive   
        
        
        
        