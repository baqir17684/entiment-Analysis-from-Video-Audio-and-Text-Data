import torch
from torch.utils.data import Dataset

class CMUMOSEIDataset(Dataset):
    def __init__(self, data, modalities=['glove_vectors', 'COVAREP', 'OpenFace_2'], max_seq_len=50):
        self.data = data
        self.modalities = modalities
        self.max_seq_len = max_seq_len
        
        # 每个 sample 是完整的 video_id[segment]
        self.samples = list(data['All Labels'].keys())

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        full_id = self.samples[idx]
        video_id = full_id.split('[')[0]  # 拆出 video_id
        segment_index = int(full_id.split('[')[1][:-1])

        features = []
        for modality in self.modalities:
            # 拿 interval
            interval = self.data[modality].data[full_id]['intervals'][0]
            start, end = interval  # interval: [start_time, end_time]

            # 拿原始 feature
            feature = self.data[modality].data[video_id]['features']

            # 根据 interval 裁剪
            feature = torch.tensor(feature[start:end], dtype=torch.float)
            feature = self._pad_or_truncate(feature)
            features.append(feature)

        multimodal_feature = torch.cat(features, dim=-1)  # (max_seq_len, total_dim)

        # Label
        label = self.data['All Labels'][full_id]['features'][0]  # label 直接拿完整 id

        return multimodal_feature, torch.tensor(label, dtype=torch.float)

    def _pad_or_truncate(self, feature):
        T, D = feature.shape
        if T >= self.max_seq_len:
            return feature[:self.max_seq_len]
        else:
            pad = torch.zeros((self.max_seq_len - T, D))
            return torch.cat([feature, pad], dim=0)