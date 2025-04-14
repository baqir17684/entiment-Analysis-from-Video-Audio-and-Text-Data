# test_emoreact_cls.py
import torch
from torch.utils.data import DataLoader
from project1 import CMUMOSEIDataset
from test_mosi import EmoReact  # ⚡ 注意是EmoReact，不是MFN
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# ========== 加载数据 ==========
with open("cmumosei_highlevel.pkl", "rb") as f:
    cmumosei_highlevel = pickle.load(f)

dataset = CMUMOSEIDataset(cmumosei_highlevel)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# ========== 配置 ==========
config = {
    "input_dims": [300, 74, 713],
    "hidden_size": 128,
    "output_dim": 2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚡ 关键：换成EmoReact
model = EmoReact(config).to(device)
model.load_state_dict(torch.load('emoreact_cls_model.pt', map_location=device))
model.eval()

# ========== 开始评估 ==========
all_preds = []
all_labels = []

with torch.no_grad():
    for (text, audio, visual), labels in dataloader:
        text = text.permute(1, 0, 2).to(device)
        audio = audio.permute(1, 0, 2).to(device)
        visual = visual.permute(1, 0, 2).to(device)
        labels = (labels > 0).long().to(device)

        outputs = model(text, audio, visual)
        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

# ========== 计算分类指标 ==========
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

# ========== 打印结果 ==========
print("\n✅ 测试结果：")
print(f"  Test Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
