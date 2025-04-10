import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from project1 import CMUMOSEIDataset   # 替换成你的 Dataset 文件
from test_mosi import MFN              # 替换成你的模型文件
import pickle

# ========== 加载数据 ==========
with open("cmumosei_highlevel.pkl", "rb") as f:
    cmumosei_highlevel = pickle.load(f)

# ========== 参数配置 ==========
batch_size = 32
num_epochs = 10
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 数据加载 ==========
dataset = CMUMOSEIDataset(cmumosei_highlevel)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ========== 模型配置 ==========
config = {
    "input_dims": [300, 74, 713],
    "h_dims": [128, 32, 32],
    "memsize": 64,
    "windowsize": 2
}
NN1Config = {"shapes": 128, "drop": 0.3}
NN2Config = {"shapes": 64, "drop": 0.3}
gamma1Config = {"shapes": 64, "drop": 0.3}
gamma2Config = {"shapes": 64, "drop": 0.3}
outConfig = {"shapes": 32, "drop": 0.3}

model = MFN(config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig).to(device)

# ========== 损失函数 & 优化器 ==========
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ========== 开始训练 ==========
print("🔁 开始训练 MFN 模型...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for (text, audio, visual), y in dataloader:
        # 分别处理 text, audio, visual
        text = text.permute(1, 0, 2).to(device)   # (T, B, D_text)
        audio = audio.permute(1, 0, 2).to(device) # (T, B, D_audio)
        visual = visual.permute(1, 0, 2).to(device) # (T, B, D_visual)
        y = y.to(device).float()  # (B,)

        optimizer.zero_grad()
        output = model(text, audio, visual)  # 传三个模态进去
        loss = criterion(output.squeeze(), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# ========== 保存模型 ==========
torch.save(model.state_dict(), "mfn_model.pt")
print("✅ 模型已保存为 mfn_model.pt")

