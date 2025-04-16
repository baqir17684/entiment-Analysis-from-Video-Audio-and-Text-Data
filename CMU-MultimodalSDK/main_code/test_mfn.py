# test_model_cls.py
import torch
from torch.utils.data import DataLoader
from project1 import CMUMOSEIDataset
from test_mosi import MFN
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

with open("cmumosei_highlevel.pkl", "rb") as f:
    cmumosei_highlevel = pickle.load(f)

dataset = CMUMOSEIDataset(cmumosei_highlevel)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MFN(config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig).to(device)
model.load_state_dict(torch.load('mfn_cls_model.pt', map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for (text, audio, visual), labels in dataloader:
        text = text.permute(1, 0, 2).to(device)
        audio = audio.permute(1, 0, 2).to(device)
        visual = visual.permute(1, 0, 2).to(device)
        labels = (labels > 0).long().to(device)

        outputs = model(text, audio, visual)  # (B, 2)
        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds, digits=4)

print("\n✅ test result：")
print(f"  Test Accuracy: {acc * 100:.2f}%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)

