import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset

def normalize_df(dataframe):
    df_copy = dataframe.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = min_max_scaler.fit_transform(df_copy)
    
    return pd.DataFrame(df_scaled)

class WindSpeedDataset(Dataset):
    def __init__(self, dataframe, suffle=False, transform=None):
        self.dataframe = dataframe.copy()
        dataframe1 = dataframe.copy()
        self.suffle = suffle
        self.transform = transform
        
        if 'time' in dataframe1:
            dataframe1.pop('time')
        if 'wind_speed' in dataframe1:
            self.labelset = dataframe1.pop('wind_speed')
            
        self.featureset = dataframe1
        
    def __len__(self):
        return len(self.featureset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        
        label = np.array([self.labelset.iloc[idx]])
        features = self.featureset.iloc[idx].to_numpy()
        
        sample =(features, label)
        
        if self.transform:
            sample = self.transform(sample)

        return sample

class ComposeTransform(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> ComposeTransform([
        >>>     ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            img = t(data)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample
        return (torch.from_numpy(data),torch.from_numpy(label))