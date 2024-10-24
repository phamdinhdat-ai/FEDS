


from audiomentations import Compose, TimeStretch, \
                            PitchShift, Shift, ClippingDistortion, \
                            Gain, GainTransition, Reverse, AddGaussianNoise
import numpy as np
from tqdm import tqdm
import random
import torch 
# clipping1 = ClippingDistortion(min_percentile_threshold=2, max_percentile_threshold=4, p=1.0,)
clipping = ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=1, p=0.3)
gain = Gain(min_gain_in_db=0, max_gain_in_db=0.2, p=0.3)
# gain2 = Gain(min_gain_in_db=-3.0, max_gain_in_db=-2.1, p=1.0)
gaintransition = GainTransition(min_gain_in_db=1.1, max_gain_in_db=2.0, p=1.0)
gaussnoise = AddGaussianNoise(min_amplitude=0.1, max_amplitude=0.2, p=0.3)
timestretch = TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
pitchshift = PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
reverse = Reverse(p=1.0)

augments = [
    clipping,
    gain,
    gaintransition,
    # gaussnoise,
    # timestretch,
    # pitchshift,
    reverse,
    # shift,
]
def augment_data(data):
    
    b, d , k , s = data.shape
    # data = data.reshape(b, s, d)
    total_au = []
    # print(k)
    for i in range(k):
        data_aug = np.array([]) 
        data_dv = data[:, :, i, :].reshape(b, s, d)
        # print(data.shape)
        for X in data_dv: 
            dim_features  = X.shape[-1]
            stack_dim = []
            for dim in range(dim_features):
                x = X[:, dim]
                method = random.choice(augments)
                X_aug = method(samples=x, sample_rate=50)
                # Y_aug = method(samples=y, sample_rate=8000)
                # Z_aug = method(samples=z, sample_rate=8000)
                stack_dim.append(X_aug)
            aug_data = np.transpose(np.array(stack_dim))
            
            if data_aug.shape[0] == 0:
                data_aug = np.expand_dims( np.transpose(np.array(stack_dim)),  axis=0)
                
            else:
                data_aug = np.concatenate([data_aug,np.expand_dims( np.transpose(np.array(stack_dim)),  axis=0)], axis = 0)
        
        total_au.append(data_aug)    
        # if total_au.shape[0] == 0:
        #         total_au = np.expand_dims( np.transpose(np.array(total_au)),  axis=0)
                
        # else:
        #     total_au = np.concatenate([total_au,np.expand_dims(np.transpose(np.array(data_aug)),  axis=0)], axis = 0)
        
    return torch.tensor(np.array(total_au).reshape(b, d, k, s), dtype=torch.float64)
    # return torch.tensor(data_aug.reshape(b, d, 1, s), dtype = torch.float64)



# from audiomentations import Compose, TimeStretch, \
#                             PitchShift, Shift, ClippingDistortion, \
#                             Gain, GainTransition, Reverse, AddGaussianNoise
# import numpy as np
# from tqdm import tqdm
# import random
# import torch 
# # clipping1 = ClippingDistortion(min_percentile_threshold=2, max_percentile_threshold=4, p=1.0,)
# clipping = ClippingDistortion(min_percentile_threshold=1, max_percentile_threshold=2, p=1.0)
# gain = Gain(min_gain_in_db=-2.0, max_gain_in_db=-1.1, p=1.0)
# # gain2 = Gain(min_gain_in_db=-3.0, max_gain_in_db=-2.1, p=1.0)
# gaintransition = GainTransition(min_gain_in_db=1.1, max_gain_in_db=2.0, p=1.0)
# gaussnoise = AddGaussianNoise(min_amplitude=0.1, max_amplitude=1.2, p=0.5)
# timestretch = TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5)
# pitchshift = PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
# reverse = Reverse(p=1.0)
# augments = [
#     clipping,
#     gain,
#     gaintransition,
#     # gaussnoise,
#     # timestretch,
#     # pitchshift,
#     reverse,
#     # shift,
# ]
# def augment_data(data):
    
#     b, d , _ , s = data.shape
#     data = data.reshape(b, s, d)
#     data_aug = np.array([]) 
#     for X in data: 
#         dim_features  = X.shape[-1]
#         stack_dim = []
#         for dim in range(dim_features):
#             x = X[:, dim]
#             method = random.choice(augments)
#             X_aug = method(samples=x, sample_rate=8000)
#             # Y_aug = method(samples=y, sample_rate=8000)
#             # Z_aug = method(samples=z, sample_rate=8000)
#             stack_dim.append(X_aug)
#         aug_data = np.transpose(np.array(stack_dim))
        
#         if data_aug.shape[0] == 0:
#             data_aug = np.expand_dims( np.transpose(np.array(stack_dim)),  axis=0)
            
#         else:
#             data_aug = np.concatenate([data_aug,np.expand_dims( np.transpose(np.array(stack_dim)),  axis=0)], axis = 0)
#     return torch.tensor(data_aug.reshape(b, d, 1, s), dtype = torch.float64)





# # def augment_data(data):
    
# #     b, d , _ , s = data.shape
# #     data = data.reshape(b, s, d)
# #     data_aug = np.array([]) 
# #     for X in data: 
# #         dim_features  = X.shape[-1]
# #         stack_dim = []
# #         for dim in range(dim_features):
# #             x = X[:, dim]
# #             method = random.choice(augments)
# #             X_aug = method(samples=x, sample_rate=8000)
# #             # Y_aug = method(samples=y, sample_rate=8000)
# #             # Z_aug = method(samples=z, sample_rate=8000)
# #             stack_dim.append(X_aug)
# #         aug_data = np.transpose(np.array(stack_dim))
        
# #         if data_aug.shape[0] == 0:
# #             data_aug = np.expand_dims( np.transpose(np.array(stack_dim)),  axis=0)
            
# #         else:
# #             data_aug = np.concatenate([data_aug,np.expand_dims( np.transpose(np.array(stack_dim)),  axis=0)], axis = 0)
# #     return torch.tensor(data_aug.reshape(b, d, 1, s), dtype = torch.float64)