

import torch 
import torch.nn as nn
import torch.nn.functional as F
import math  


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # get dropout
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        # x ~ [sq_len, batch, dim]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    
    def __init__(self,d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(self.max_len, self.d_model)
        pos = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        i_2 = torch.arange(0, self.d_model, 2).float()
        div = torch.exp(i_2 * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x:torch.Tensor):
        emb  = self.pe[: x.size(0), : ]
        print(emb.shape)
        return x +  emb


class TransAm(nn.Module):
    def __init__(self, in_dim = 3, d_model=128, seq_len = 20, n_class = 12, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        self.n_classes = n_class
        self.src_mask = None
        self.proj_inp = nn.Linear(in_features=in_dim, out_features=d_model)
        self.pos_emb = LearnablePositionalEncoding(d_model=d_model)
        # self.pos_encoder = PositionalEncoding(d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.avg_pool = nn.AdaptiveAvgPool1d(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.fc = nn.Linear(d_model * seq_len,self.n_classes)
        # self.softmax = nn.Softmax(dim =1 )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange) # init weight unifor

    def forward(self,src):
        batch, dim, _ , sq_len = src.size()
        src = src.reshape(batch, sq_len, dim)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        input = self.proj_inp(src) # [batch, sq_len, d_model]
        # print("out_proj: ", input.shape) 
        input = input.permute(1, 0, 2) #[sq_len, batch, d_model]
        output = self.pos_emb(input) 
        # output = output.transpose(1, 2)
        
        # print("out_pos:", output.shape)
        # output = output.permute(1, 0 , 2)
        # output = self.batch_norm(output)
        output = self.avg_pool(output)
        # print("out_avg:", output.shape)
        output = output.permute(1, 0 , 2) # batch, sq_len, d_model
        output = self.transformer_encoder(output,self.src_mask) # [batch, sq_len, d_model]
        # print("out_encoder:", output.shape)
        output = output.reshape(output.shape[0], -1) #[batch, sq_len * d_model]
        output = self.fc(output)
        # print("out_decoder:", output.shape) 

        # output = self.softmax(output) # [batch, num_classes]
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask # [batch, sq_len]
        


