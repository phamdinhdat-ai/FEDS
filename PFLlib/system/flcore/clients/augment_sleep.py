from audiomentations import Compose, TimeStretch, \
                            PitchShift, Shift, ClippingDistortion, \
                            Gain, GainTransition, Reverse, AddGaussianNoise
import numpy as np
from tqdm import tqdm
import random
import torch 
# clipping1 = ClippingDistortion(min_percentile_threshold=2, max_percentile_threshold=4, p=1.0,)
clipping = ClippingDistortion(min_percentile_threshold=1, max_percentile_threshold=2, p=1.0)
gain = Gain(min_gain_in_db=-2.0, max_gain_in_db=-1.1, p=1.0)
# gain2 = Gain(min_gain_in_db=-3.0, max_gain_in_db=-2.1, p=1.0)
gaintransition = GainTransition(min_gain_in_db=1.1, max_gain_in_db=2.0, p=1.0)
gaussnoise = AddGaussianNoise(min_amplitude=0.1, max_amplitude=1.2, p=0.5)
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
    
    b, d , _ , s = data.shape
    data = data.reshape(b, s, d)
    data_aug = np.array([]) 
    for X in data: 
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
        method = random.choice(augments)
        X_aug = method(samples=x, sample_rate=8000)
        Y_aug = method(samples=y, sample_rate=8000)
        Z_aug = method(samples=z, sample_rate=8000)
        aug_data = np.transpose(np.array([X_aug, Y_aug, Z_aug]))
        
        if data_aug.shape[0] == 0:
            data_aug = np.expand_dims( np.transpose(np.array([X_aug, Y_aug, Z_aug])),  axis=0)
            
        else:
            data_aug = np.concatenate([data_aug,np.expand_dims( np.transpose(np.array([X_aug, Y_aug, Z_aug])),  axis=0)], axis = 0)
    return torch.tensor(data_aug.reshape(b, d, 1, s), dtype = torch.float64)