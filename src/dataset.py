import os
import pandas as pd
from torch.utils.data import Dataset


class KeystrokeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        keycode = sample['KEYCODE']
        df = sample['df']
        ht = sample['ht']
        label = sample['label']
        return keycode, df, ht, label


def load_data(path):
    data = []
    for file in os.listdir(path):
        if file.endswith('.csv'):
            user_data = pd.read_csv(os.path.join(path, file))
            user_id = int(file.split('_')[0])
            user_data['USER_ID'] = user_id
            user_data['KEYCODE'] = user_data['KEYCODE'].astype(int)
            user_data['df'] = user_data['df'].astype(float)
            user_data['ht'] = user_data['ht'].astype(float)
            data.append(user_data)
    return pd.concat(data, ignore_index=True)
