import torch
from torch.utils.data import Dataset

class CMUMOSEIDataset(Dataset):
    def __init__(self, data, modalities=['glove_vectors', 'COVAREP', 'OpenFace_2'], max_seq_len=50):
        self.data = data
        self.modalities = modalities
        self.max_seq_len = max_seq_len

        self.samples = []  # 存储所有的 video_id[segment]

        # 这里直接遍历所有的完整 id
        for full_id in data['All Labels'].data:
            # full_id example: '-3g5yACwYnA[2]'
            if 'features' in data['All Labels'].data[full_id]:  # 确保有 feature
                self.samples.append(full_id)

        print("🔍 检查数据结构...")
        for full_id in self.samples[:5]:  # 打印前5个样本看一下
            print(f"full_id: {full_id}")
            print(f"  intervals shape: {self.data['All Labels'].data[full_id]['intervals'].shape}")
            print(f"  features shape: {self.data['All Labels'].data[full_id]['features'].shape}")

        print(f"✅ 收集到样本数量：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_id = self.samples[idx]

        text_feature = torch.tensor(self.data['glove_vectors'].data[full_id]['features'], dtype=torch.float)
        audio_feature = torch.tensor(self.data['COVAREP'].data[full_id]['features'], dtype=torch.float)
        visual_feature = torch.tensor(self.data['OpenFace_2'].data[full_id]['features'], dtype=torch.float)
        text_feature = self._pad_or_truncate(text_feature)
        audio_feature = self._pad_or_truncate(audio_feature)
        visual_feature = self._pad_or_truncate(visual_feature)

        label = self.data['All Labels'].data[full_id]['features'][0][0]
        return (text_feature, audio_feature, visual_feature), torch.tensor(label, dtype=torch.float)





    def _pad_or_truncate(self, feature):
        T, D = feature.shape
        if T >= self.max_seq_len:
            return feature[:self.max_seq_len]
        else:
            pad = torch.zeros((self.max_seq_len - T, D), dtype=feature.dtype)
            return torch.cat([feature, pad], dim=0)
