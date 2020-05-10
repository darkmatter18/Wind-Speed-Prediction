#    Copyright 2020 Arkadip Bhattacharya

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class WindSpeedDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        dataframe1 = dataframe.copy()
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
    
class WindSpeedTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_size=6, transform=None):
        dataframeC = dataframe.copy()

        self.transform = transform
        self.window_size = window_size
        
        if 'time' in dataframeC:
            dataframeC.pop('time')
            
        if 'wind_speed' in dataframeC:
            self.labelset = dataframeC['wind_speed']
            
        self.featureset = dataframeC

    def __len__(self):
        return math.floor(len(self.featureset) - self.window_size) - 1
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        
        label = np.array([self.labelset.iloc[idx+self.window_size]])
        features = self.featureset.iloc[idx:idx+self.window_size].to_numpy()
        
        sample = (features, label)
        
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