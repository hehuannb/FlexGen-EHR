import torch
import torch.utils.data
import os
import numpy as np
import pandas as pd

DATA_FILEPATH     = 'all_hourly_data.h5'
GAP_TIME          = 6  # In hours
WINDOW_SIZE       = 24 # In hours
SEED              = 1
ID_COLS           = ['subject_id', 'hadm_id', 'icustay_id']
 
seed = 804
torch.manual_seed(seed)
statics        = pd.read_hdf(DATA_FILEPATH, 'patients')