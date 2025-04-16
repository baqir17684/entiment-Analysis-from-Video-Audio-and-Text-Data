# train_small_umt.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from project1 import CMUMOSEIDataset
from test_mosi import SmallUMT
import pickle
import os

with open("cmumosei_highlevel.pkl", "rb") as f:
    cmumosei_highlevel = pickle.load(f)

dataset = CMUMOSEIDataset(cmumosei_highlevel)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # ⚡ batch=8

config = {
    "input_dims": [300, 74, 713],
    "hidden_size": 256,
    "num_heads": 4,
    "ffn_dim": 512,
    "num_layers": 6,
    "dropout": 0.1,
    "output_dim": 2
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallUMT(config).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

scaler = torch.cuda.amp.GradScaler()

epochs = 30
save_path = 'small_umt_model.pt'

print("\n🔁 start training SmallUMT")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, ((text, audio, visual), labels) in enumerate(dataloader):
        text = text.permute(1, 0, 2).to(device)    # (T, B, D) → (B, T, D)
        audio = audio.permute(1, 0, 2).to(device)
        visual = visual.permute(1, 0, 2).to(device)
        labels = (labels > 0).long().to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = model(text, audio, visual)  # (B, 2)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), save_path)
print(f"\n✅ SmallUMT 模型已保存为 {save_path}")
