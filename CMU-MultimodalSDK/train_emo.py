# train_emo.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from project1 import CMUMOSEIDataset  # 你的Dataset
from test_mosi import EmoReact        # 我们刚加的模型在这里
import pickle

# ========== 加载数据 ==========

# 读数据
with open("cmumosei_highlevel.pkl", "rb") as f:
    cmumosei_highlevel = pickle.load(f)

# 准备 Dataset & DataLoader
dataset = CMUMOSEIDataset(cmumosei_highlevel)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ========== 配置 ==========

config = {
    "input_dims": [300, 74, 713],  # Text, Audio, Visual特征维度
    "hidden_size": 128,            # 每个LSTM隐藏层大小
    "output_dim": 2                # 回归任务输出一个值
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmoReact(config).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # 学习率可以稍微高一点
num_epochs = 10

# ========== 开始训练 ==========

print("🔁 开始训练 EmoReact 模型...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, ((text, audio, visual), labels) in enumerate(dataloader):
        text = text.permute(1, 0, 2).to(device)    # (T, B, D)
        audio = audio.permute(1, 0, 2).to(device)
        visual = visual.permute(1, 0, 2).to(device)
        labels = (labels > 0).long().to(device)  # 正负情感分类标签


        optimizer.zero_grad()

        outputs = model(text, audio, visual)   # (B, 1)
        if torch.isnan(outputs).any():
            print(f"[Batch {batch_idx}] NaN detected in model output!")
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 防止梯度爆炸
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# ========== 保存模型 ==========

torch.save(model.state_dict(), "emoreact_cls_model.pt")
print("✅ EmoReact 模型已保存为 emoreact_cls_model.pt")
