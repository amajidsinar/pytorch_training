import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Sampler


class StratifiedSampler(Sampler):
    def __init__(self, data_source, n_splits, val_size, random_state, sample_train=True):
        self.train_idx, self.val_idx = self._stratify_split(data_source, n_splits, val_size, random_state)
        self.sample_train = sample_train
    def __iter__(self):  
        if self.sample_train == True:
            return iter(self.train_idx)
        else:
            return iter(self.val_idx)
    def __len__(self):
        if self.sample_train == True:
            return len(self.train_idx)
        else:
            return len(self.val_idx)
        
    def _stratify_split(self,data_source, n_splits=1, val_size=0.3, random_state=69):
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
        X = np.array(data_source.imgs)[:,0]
        y = np.array(data_source.imgs)[:,1].astype("int")
        for train_idx, val_idx in sss.split(X,y):
            train_idx = train_idx
            val_idx = val_idx

        return train_idx, val_idx
