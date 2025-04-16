import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from project1 import CMUMOSEIDataset
from test_mosi import MFN            
import pickle

with open("cmumosei_highlevel.pkl", "rb") as f:
    cmumosei_highlevel = pickle.load(f)

batch_size = 32
num_epochs = 10
learning_rate = 1e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CMUMOSEIDataset(cmumosei_highlevel)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

config = {
    "input_dims": [300, 74, 713],
    "h_dims": [128, 32, 32],
    "memsize": 64,
    "windowsize": 2,
    "output_dim": 2
}
NN1Config = {"shapes": 128, "drop": 0.3}
NN2Config = {"shapes": 64, "drop": 0.3}
gamma1Config = {"shapes": 64, "drop": 0.3}
gamma2Config = {"shapes": 64, "drop": 0.3}
outConfig = {"shapes": 32, "drop": 0.3}

model = MFN(config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print("🔁 start training MFN...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, ((text, audio, visual), y) in enumerate(dataloader):
        text = text.permute(1, 0, 2).to(device)   # (T, B, D_text)
        audio = audio.permute(1, 0, 2).to(device)  # (T, B, D_audio)
        visual = visual.permute(1, 0, 2).to(device)  # (T, B, D_visual)
        y = (y > 0).long().to(device)  # (B,)

        if torch.isnan(text).any():
            print(f"[Batch {batch_idx}] NaN in text input!")
        if torch.isnan(audio).any():
            print(f"[Batch {batch_idx}] NaN in audio input!")
        if torch.isnan(visual).any():
            print(f"[Batch {batch_idx}] NaN in visual input!")
        if torch.isnan(y).any():
            print(f"[Batch {batch_idx}] NaN in labels!")

        optimizer.zero_grad()

        output = model(text, audio, visual)
        if torch.isnan(output).any():
            print(f"[Batch {batch_idx}] NaN in model output!")
        loss = criterion(output, y)

        if torch.isnan(loss):
            print(f"[Warning] Loss is NaN at batch {batch_idx}, skipping...")
            continue

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "mfn_cls_model.pt")
print("✅ 模型已保存为 mfn_model.pt")

