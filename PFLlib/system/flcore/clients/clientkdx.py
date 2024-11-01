
import copy
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import svds
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client
from flcore.clients.augment_sleep import augment_data
from flcore.clients.helper_function import ContrastiveLoss, RKDLoss



class clientKDX(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mentee_learning_rate = args.mentee_learning_rate

        self.global_model = copy.deepcopy(args.model)
        self.optimizer_g = torch.optim.SGD(self.global_model.parameters(), lr=self.mentee_learning_rate)
        self.learning_rate_scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_g, 
            gamma=args.learning_rate_decay_gamma
        )

        self.feature_dim = list(args.model.head.parameters())[0].shape[1]
        # print(list(args.model.head.parameters())[0].shape)
        # self.W_h = nn.Linear(self.feature_dim, self.feature_dim, bias=False).to(self.device)
        self.g_w = nn.Linear(self.num_classes, self.num_classes, bias=False).to(self.device)
        
        self.optimizer_W = torch.optim.SGD(self.g_w.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler_W = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer_W, 
            gamma=args.learning_rate_decay_gamma
        )
        
        self.contrastive_loss  = ContrastiveLoss()
        self.rkd_loss = RKDLoss()
        
        self.KL = nn.KLDivLoss()
        self.MSE = nn.MSELoss()
        self.contrastive_loss = ContrastiveLoss()
        self.kd_loss = RKDLoss()

        self.compressed_param = {}
        self.energy = None


    def train(self):
        trainloader = self.load_train_data()
        random_loader = copy.deepcopy(trainloader)
        random_dataloader = iter(random_loader)
        
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        
        
        for epoch in range(max_local_epochs):

            loss_e = 0
            loss_ct_e  = 0 
            loss_rl_e = 0 
            loss_g_e = 0


            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x_ = x.clone().numpy()
                    x_au = augment_data(x_)
                    x_au = x_au.to(self.device, dtype = x.dtype)
                    x = x.to(self.device)
                try: 
                    random_x, _  = next(random_dataloader)
                except:
                    random_dataloader  =  iter(random_loader)
                    random_x, _ = next(random_dataloader)
                random_x = random_x.to(self.device)

                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                    
                self.optimizer.zero_grad()
                self.optimizer_g.zero_grad()
                self.optimizer_W.zero_grad()
                
                # representative projection of local model
                rep = self.model.base(x)
                rep_au = self.model.base(x_au)
                rep_rd = self.model.base(random_x)
                # representative projection of global model
                with torch.no_grad():
                    rep_g = self.global_model.base(x)
                    rep_au_g = self.global_model.base(x_au)
                    rep_rd_g = self.global_model.base(random_x)
                # local prediction
                
                output = self.model.head(rep)
                output_au = self.model.head(rep_au)
                output_rd = self.model.head(rep_rd)
                # global prediction 
                output_rd_g = self.global_model.head(rep_rd_g)
                output_g = self.global_model.head(rep_g)
                
                zi = self.g_w(output)
                zj = self.g_w(output_au)
                zrd = self.g_w(output_rd)
                
                ## add new contrastive loss and relative loss
                # contrastive loss 
                
                ct_local = self.contrastive_loss(rep, rep_au)
                ct_global = self.contrastive_loss(rep_g, rep_au_g)
                # ct_local = self.contrastive_loss(zi, zj)
                # ct_global = self.contrastive_loss(zj, zrd)

                # relative loss 
                rl_local = self.rkd_loss(zi, zj, zrd)
                
                rl_global = self.rkd_loss(output_g, output_au,  output_rd_g)
                
                loss_ct = ct_local + ct_global
                loss_rl = rl_local 
                
                
                
                
                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                # L_d_g = self.KL(F.log_softmax(output_g, dim=1), F.softmax(output, dim=1)) / (CE_loss + CE_loss_g)
                # L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)
                # L_h_g = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)


                
                # loss = CE_loss + L_d + L_h 
                # loss_g = CE_loss_g + L_d_g + L_h_g

                loss = CE_loss + 0.1 * loss_ct + 0.1*loss_rl + L_d
                loss_g = CE_loss_g + ct_global + rl_global

                
                loss.backward(retain_graph=True)
                # loss_g.backward()
                # prevent divergency on specifical tasks
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.g_w.parameters(), 10)
                self.optimizer.step()
                # self.optimizer_g.step()
                self.optimizer_W.step()
                loss_e += loss.item()
                loss_g_e += loss_g.item()
                loss_ct_e += loss_ct.item()
                loss_rl_e += loss_rl.item()
            print(f"Epoch: {epoch}|  CT loss: {round(loss_ct_e/len(trainloader),4)}| RL loss: {round(loss_rl_e/len(trainloader),4)} ")
            print(f"Epoch: {epoch}|  Loss:  {round(loss_e/len(trainloader), 4)} |Global loss: {round(loss_g_e/len(trainloader), 4)}| Local CE loss: {round(CE_loss.item(), 4)}  | Global CE loss: {round(CE_loss_g.item(), 4)}")
            
        # self.model.cpu()

        self.decomposition()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            self.learning_rate_scheduler_g.step()
            self.learning_rate_scheduler_W.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        
    def set_parameters(self, global_param, energy):
        # recover
        for k in global_param.keys():
            if len(global_param[k]) == 3:
                # use np.matmul to support high-dimensional CNN param
                global_param[k] = np.matmul(global_param[k][0] * global_param[k][1][..., None, :], global_param[k][2])
        
        for name, old_param in self.global_model.named_parameters():
            if name in global_param:
                old_param.data = torch.tensor(global_param[name], device=self.device).data.clone()
        self.energy = energy

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        
        # random_loader = iter(trainloader)
        random_loader = copy.deepcopy(trainloader)
        random_loader = iter(random_loader)
        
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                # else:
                #     x = x.to(self.device)
                else:
                    # print(x.shape)
                    x_ = x.clone().numpy()
                    x_au = augment_data(x_)
                    x_au = x_au.to(self.device, dtype = x.dtype)
                    x = x.to(self.device)
                try: 
                    random_x, _  = next(random_dataloader)
                except:
                    random_dataloader  =  iter(random_loader)
                    random_x, _ = next(random_dataloader)
                random_x = random_x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                rep_g = self.global_model.base(x)
                output = self.model.head(rep)
                output_g = self.global_model.head(rep_g)
                # representative projection of local model
                rep = self.model.base(x)
                rep_au = self.model.base(x_au)
                rep_rd = self.model.base(random_x)
                # # representative projection of global model
                with torch.no_grad():
                    rep_g = self.global_model.base(x)
                    rep_au_g = self.global_model.base(x_au)
                    rep_rd_g = self.global_model.base(random_x)
                # # local prediction
                
                output = self.model.head(rep)
                output_au = self.model.head(rep_au)
                output_rd = self.model.head(rep_rd)
                # global prediction 
                output_rd_g = self.global_model.head(rep_rd_g)
                output_g = self.global_model.head(rep_g)
                
                
                ### add new contrastive loss and relative loss
                #contrastive loss 
                
                zi = self.g_w(output)
                zj = self.g_w(output_au)
                zrd = self.g_w(output_rd)
                
                ## add new contrastive loss and relative loss
                # contrastive loss 
                
                ct_local = self.contrastive_loss(rep, rep_au)
                ct_global = self.contrastive_loss(rep_g, rep_au_g)

                # # relative loss 
                rl_local = self.rkd_loss(output, output_au, output_rd)
                
                # rl_global = self.rkd_loss(rep, rep_au_g,  rep_rd_g)
                
                loss_ct = ct_local + ct_global
                loss_rl = rl_local 
                
                CE_loss = self.loss(output, y)
                CE_loss_g = self.loss(output_g, y)
                # ct_loss = self.contrastive_loss(rep, rep_g)
                # nt_loss = self.kd_loss(rep, rep_g, self.W_h(rep_g))
                L_d = self.KL(F.log_softmax(output, dim=1), F.softmax(output_g, dim=1)) / (CE_loss + CE_loss_g)
                # L_h = self.MSE(rep, self.W_h(rep_g)) / (CE_loss + CE_loss_g)

                loss = CE_loss + 0.1 * loss_ct + 0.1*loss_rl + L_d
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
    def decomposition(self):
        self.compressed_param = {}
        for name, param in self.global_model.named_parameters():
            # print("Name Layers: ", name)
            
            param_cpu = param.detach().cpu().numpy()
            # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
            if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
                u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
                # u_s, sigma_s, v_s = svds(param_cpu, k=4)
                
                # support high-dimensional CNN param
                if len(u.shape)==4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (2, 0, 1))
                    v = np.transpose(v, (2, 3, 0, 1))
                    # print("SVD Layers: ", name)
                    
                    # print("U: ", u.shape)
                    # print("Sigma: ", sigma.shape)
                    # print("V: ", v.shape)
                    # print(param_cpu.shape)
                    
                threshold=0
                if np.sum(np.square(sigma))==0:
                    compressed_param_cpu=param_cpu
                else:
                    for singular_value_num in range(len(sigma)):
                        if np.sum(np.square(sigma[:singular_value_num]))>self.energy*np.sum(np.square(sigma)):
                            threshold=singular_value_num
                            break
                    u=u[:, :threshold]
                    sigma=sigma[:threshold]
                    v=v[:threshold, :]
                    # print(threshold)
                    # support high-dimensional CNN param
                    if len(u.shape)==4:
                        u = np.transpose(u, (2, 3, 0, 1))
                        sigma = np.transpose(sigma, (1, 2, 0))
                        v = np.transpose(v, (2, 3, 0, 1))
                    compressed_param_cpu=[u,sigma,v]
            elif 'embeddings' not in name:
                compressed_param_cpu=param_cpu

            self.compressed_param[name] = compressed_param_cpu
            
            
    def decomposition_v2(self):
        self.compressed_param = {}
        for name, param in self.global_model.named_parameters():
            param_cpu = param.detach().cpu().numpy()
            # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
            if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
                u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
                # support high-dimensional CNN param
                if len(u.shape)==4:
                    u = np.transpose(u, (2, 3, 0, 1))
                    sigma = np.transpose(sigma, (2, 0, 1))
                    v = np.transpose(v, (2, 3, 0, 1))
                threshold=0
                if np.sum(np.square(sigma))==0:
                    compressed_param_cpu=param_cpu
                else:
                    for singular_value_num in range(len(sigma)):
                        if np.sum(np.square(sigma[:singular_value_num]))>self.energy*np.sum(np.square(sigma)):
                            threshold=singular_value_num
                            break
                    u=u[:, :threshold]
                    sigma=sigma[:threshold]
                    v=v[:threshold, :]
                    # support high-dimensional CNN param
                    if len(u.shape)==4:
                        u = np.transpose(u, (2, 3, 0, 1))
                        sigma = np.transpose(sigma, (1, 2, 0))
                        v = np.transpose(v, (2, 3, 0, 1))
                    compressed_param_cpu=[u,sigma,v]
            elif 'embeddings' not in name:
                compressed_param_cpu=param_cpu

            self.compressed_param[name] = compressed_param_cpu  