from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, mode='slow', flag='train', data_idx=None):
        self.mode = mode
        self.data_idx = data_idx
        if mode == 'slow':
            dataset = np.load('slow_dataset.npz', allow_pickle=True)
        elif mode == 'fast':
            dataset = np.load('fast_dataset.npz', allow_pickle=True)
        elif mode == 'full':
            dataset = np.load('full_dataset_filtered.npz', allow_pickle=True)
        else:
            raise NotImplementedError

        inp = dataset['inp'] # ['Vp', 'Alpha_ratio', 'entropy', 'Alfvenicity', 'Np', 'Tp', 'coulomb num']
        tar = dataset['tar']
        tar_time = dataset['time']

        ## exclude cme period
        mask = dataset['is_cme'] == 0
        inp, tar, tar_time = inp[mask], tar[mask], tar_time[mask]

        ## exclude data where nan value is in the target
        valid_idx = sorted(list(set(range(len(tar))) - set(np.where(np.isnan(tar))[0])))
        inp, tar, tar_time = inp[valid_idx], tar[valid_idx], tar_time[valid_idx]
        inp[:, :, 3] = abs(inp[:, :, 3])
        
        inp, tar = self._data_preprocess(inp, tar)


        months = np.array([t.month for t in pd.to_datetime(tar_time)])

        if flag == 'train':
            mask = months < 9
            self.inp, self.tar, self.tar_time = inp[mask], tar[mask], tar_time[mask]
        elif flag == 'valid':
            mask = months == 9
            self.inp, self.tar, self.tar_time = inp[mask], tar[mask], tar_time[mask]
        elif flag == 'test':
            mask = months > 9
            self.inp, self.tar, self.tar_time = inp[mask], tar[mask], tar_time[mask]
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.tar)

    def __getitem__(self, idx):
        idxs = self.data_idx
        return torch.tensor(np.take(self.inp[idx], idxs, axis=-1), dtype=torch.float32), torch.tensor(self.tar[idx], dtype=torch.float32)

    def _data_preprocess(self, inp, tar):
        ## input interpolation and normalization ##
        df_interp = []
        for b in range(inp.shape[0]):
            df = pd.DataFrame(inp[b])
            df_i = df.interpolate(method='linear', axis=0).bfill().ffill()
            df_interp.append(df_i.values)
            
        inp_interp = np.stack(df_interp, axis=0)
        inp_interp_norm = inp_interp / np.max(inp_interp, axis=(0, 1))
        x = inp_interp_norm
#        y = np.log1p(tar)
        tar_log = np.log10(tar)
        y_norm = (tar_log - tar_log.mean(axis=0)) / tar_log.std(axis=0)
        return x, y_norm[:, 1].reshape(-1, 1)

if __name__ == '__main__':
    ds = CustomDataset(mode='full', flag='train', data_idx=[0, 1, 2, 3, 4, 5, 6])
    tar_norm = ds.tar
    print(tar_norm[:128].mean())
    
    dataset = np.load('full_dataset_filtered.npz', allow_pickle=True)
    tar, time = dataset['tar'], dataset['time']
    tar, time = tar[dataset['is_cme']==0], time[dataset['is_cme']==0]
    valid_idx = sorted(list(set(range(len(tar))) - set(np.where(np.isnan(tar))[0])))
    tar = tar[valid_idx]
    time = time[valid_idx]

    
    months = np.array([t.month for t in pd.to_datetime(time)])
    mask = months < 9

    tar_denorm = tar_norm * tar.std(axis=0)[0] + tar.mean(axis=0)[0]
    
    tar = tar[mask]

    print(tar_denorm[:5])
    # print(tar.mean(axis=0), tar.std(axis=0))
    # print(tar_denorm.shape, tar.shape)
    print(tar[:5, 0])

