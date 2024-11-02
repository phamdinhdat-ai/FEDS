import copy
import random
import time
import numpy as np
from flcore.clients.clientkdx import clientKDX
from flcore.servers.serverbase import Server
from threading import Thread
import torch


class FedKDX(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientKDX)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.T_start = args.T_start
        self.T_end = args.T_end
        self.energy = self.T_start
        self.compressed_param = {}


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()
            self.decomposition()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientKDX)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
            

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.compressed_param, self.energy)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_models = []
        self.mean_client = {}
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.uploaded_ids.append(client.id)
                # recover
                
                for k in client.compressed_param.keys():
                    
                    
                    
                    if len(client.compressed_param[k]) == 3:
                        # use np.matmul to support high-dimensional CNN param
                        client.compressed_param[k] = np.matmul(
                            client.compressed_param[k][0] * client.compressed_param[k][1][..., None, :], 
                                client.compressed_param[k][2])
                        
                        if len(self.mean_client) != len(client.compressed_param): 
                            self.mean_client[k] = client.compressed_param[k]
                        else: 
                            self.mean_client[k]  += client.compressed_param[k]
                    
                    
                    else: 
                        if len(self.mean_client) != len(client.compressed_param): 
                            self.mean_client[k] = client.compressed_param[k]
                        else: 
                            self.mean_client[k]  += client.compressed_param[k]
                self.uploaded_models.append(client.compressed_param)
            
            # for k, param in self.mean_client.items():
            #         # print(param)
            #         self.mean_client[k] = param / len(self.uploaded_models)
            
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        # print(self.global_model.keys())
        
        for k in self.global_model.keys(): # not good 
            self.global_model[k] = np.zeros_like(self.global_model[k])
            
        # use 1/len(self.uploaded_models) as the weight for privacy and fairness
        for client_model in self.uploaded_models:
            self.add_parameters(1/len(self.uploaded_models), client_model)

    def add_parameters(self, w, client_model):
        # self.add_bn_weight()
        for server_k, client_k, mean_k in zip(self.global_model.keys(), client_model.keys(), self.mean_client.keys()):
            
            # if 'bn' in client_k and 'weight' in client_k: 
                # print(client_k)
                # print(client_model[client_k].shape)
            self.global_model[server_k]  += 0.5 * (self.mean_client[mean_k] * w - client_model[client_k])
            # self.global_model[server_k] += client_model[client_k] * w # oringinal update
    
    # def add_bn_weight(self):
    #     for client_model in self.uploaded_models:
    #         mean_c = None
    #         mean_s = None 
    #         var_c  = None 
    #         var_s  = None 
    #         for server_k, client_k in zip(self.global_model.keys(), client_model.keys()):
                
    #             if 'bn' in client_k and 'weight' in client_k: 
    #                 mean_c = client_model[client_k]
    #                 mean_s = self.global_model[server_k]
    #                 print(server_k)
                    
                    
    #             if 'bn' in client_k and 'bias' in client_k:
    #                 var_c = client_model[client_k]
    #                 var_s = self.global_model[server_k]
    #                 print(server_k)
                
                
    #             if mean_c is not None and mean_s is not None and  var_c is not None and  var_s is not None :
    #                 print("Mean Clients: ", mean_c[:5])
    #                 print("Mean Server: ", mean_s[:5])
    #                 print("Var Client: ", var_c[:5])
    #                 print("Var Server: ", var_s[:5])
                    
        
        
    
    
    def decomposition(self):
        self.compressed_param = {}
        for name, param_cpu in self.global_model.items():
            # refer to https://github.com/wuch15/FedKD/blob/main/run.py#L187
            
            if param_cpu.shape[0]>1 and len(param_cpu.shape)>1 and 'embeddings' not in name:
                u, sigma, v = np.linalg.svd(param_cpu, full_matrices=False)
                # support high-dimensional CNN param
                # print("Layer's named: ", name)
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
                    u=u[:,:threshold]
                    sigma=sigma[:threshold]
                    v=v[:threshold,:]
                    # support high-dimensional CNN param
                    if len(u.shape)==4:
                        u = np.transpose(u, (2, 3, 0, 1))
                        sigma = np.transpose(sigma, (1, 2, 0))
                        v = np.transpose(v, (2, 3, 0, 1))
                    compressed_param_cpu=[u,sigma,v]
            elif 'embeddings' not in name:
                compressed_param_cpu=param_cpu

            self.compressed_param[name] = compressed_param_cpu
            