import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MFN(nn.Module):
    def __init__(self, config, NN1Config, NN2Config, gamma1Config, gamma2Config, outConfig):
        super(MFN, self).__init__()
        [self.d_l, self.d_a, self.d_v] = config["input_dims"]
        [self.dh_l, self.dh_a, self.dh_v] = config["h_dims"]
        total_h_dim = self.dh_l + self.dh_a + self.dh_v
        self.mem_dim = config["memsize"]
        window_dim = config["windowsize"]
        output_dim = 2

        attInShape = total_h_dim * window_dim
        gammaInShape = attInShape + self.mem_dim
        final_out = total_h_dim + self.mem_dim
 
        h_att1 = NN1Config["shapes"]
        h_att2 = NN2Config["shapes"]
        h_gamma1 = gamma1Config["shapes"]
        h_gamma2 = gamma2Config["shapes"]
        h_out = outConfig["shapes"]

        att1_dropout = NN1Config["drop"]
        att2_dropout = NN2Config["drop"]
        gamma1_dropout = gamma1Config["drop"]
        gamma2_dropout = gamma2Config["drop"]
        out_dropout = outConfig["drop"]

        # LSTM for each modality
        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        # Attention network
        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        # Gamma networks
        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        # Output network
        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, x_l, x_a, x_v):  # 👈 修改这里，分别接收 text, audio, visual
        t, n, _ = x_l.shape  # T, B, D

        device = x_l.device

        self.h_l = torch.zeros(n, self.dh_l, device=device)
        self.h_a = torch.zeros(n, self.dh_a, device=device)
        self.h_v = torch.zeros(n, self.dh_v, device=device)
        self.c_l = torch.zeros(n, self.dh_l, device=device)
        self.c_a = torch.zeros(n, self.dh_a, device=device)
        self.c_v = torch.zeros(n, self.dh_v, device=device)
        self.mem = torch.zeros(n, self.mem_dim, device=device)

        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_mems = []

        for i in range(t):
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))

            prev_cs = torch.cat([self.c_l, self.c_a, self.c_v], dim=1)
            new_cs = torch.cat([new_c_l, new_c_a, new_c_v], dim=1)

            cStar = torch.cat([prev_cs, new_cs], dim=1)  # (B, 2 * total_h_dim)

            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))), dim=1)
            attended = attention * cStar

            cHat = torch.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))

            both = torch.cat([attended, self.mem], dim=1)

            gamma1 = torch.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = torch.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))

            self.mem = gamma1 * self.mem + gamma2 * cHat

            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v

            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_mems.append(self.mem)

        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]

        last_hs = torch.cat([last_h_l, last_h_a, last_h_v, last_mem], dim=1)  # (B, total)

        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))  # (B, 1)

        return output

class EmoReact(nn.Module):
    def __init__(self, config):
        super(EmoReact, self).__init__()
        [self.d_l, self.d_a, self.d_v] = config["input_dims"]
        hidden_size = config.get("hidden_size", 128)
        output_dim = config.get("output_dim", 2)

        # LSTM for each modality
        self.lstm_l = nn.LSTM(self.d_l, hidden_size, batch_first=False)
        self.lstm_a = nn.LSTM(self.d_a, hidden_size, batch_first=False)
        self.lstm_v = nn.LSTM(self.d_v, hidden_size, batch_first=False)

        # LayerNorm after LSTM outputs
        self.ln_l = nn.LayerNorm(hidden_size)
        self.ln_a = nn.LayerNorm(hidden_size)
        self.ln_v = nn.LayerNorm(hidden_size)

        # Fusion MLP
        fusion_input_dim = hidden_size * 3
        self.fc1 = nn.Linear(fusion_input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_dim)

        self.dropout = nn.Dropout(0.3)

        # Initialize weights properly
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_l, x_a, x_v):
        # 🚨 防止爆炸，输入特征先tanh
        x_l = torch.tanh(x_l)
        x_a = torch.tanh(x_a)
        x_v = torch.tanh(x_v)

        # LSTM
        _, (h_l, _) = self.lstm_l(x_l)
        _, (h_a, _) = self.lstm_a(x_a)
        _, (h_v, _) = self.lstm_v(x_v)

        h_l = h_l.squeeze(0)
        h_a = h_a.squeeze(0)
        h_v = h_v.squeeze(0)

        # Apply LayerNorm
        h_l = self.ln_l(h_l)
        h_a = self.ln_a(h_a)
        h_v = self.ln_v(h_v)

        # Fusion
        fused = torch.cat([h_l, h_a, h_v], dim=1)

        # MLP with BatchNorm and Dropout
        x = F.relu(self.bn1(self.fc1(fused)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        output = self.fc3(x)

        return output

class SmallUMT(nn.Module):
    def __init__(self, config):
        super(SmallUMT, self).__init__()
        [self.d_l, self.d_a, self.d_v] = config["input_dims"]
        hidden_size = config.get("hidden_size", 256)  # 比EmoReact大一点
        output_dim = config.get("output_dim", 2)

        # 先各自映射到hidden_size
        self.text_proj = nn.Linear(self.d_l, hidden_size)
        self.audio_proj = nn.Linear(self.d_a, hidden_size)
        self.visual_proj = nn.Linear(self.d_v, hidden_size)

        # TransformerEncoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=config.get("num_heads", 4),           # 4头
                dim_feedforward=config.get("ffn_dim", 512), # FFN隐层
                dropout=config.get("dropout", 0.1),
                activation='gelu',
                batch_first=True
            ),
            num_layers=config.get("num_layers", 6)         # 6层
        )

        # 最后分类头
        self.fc_out = nn.Linear(hidden_size, output_dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_l, x_a, x_v):
        # (T, B, D) → (B, T, D)
        x_l = torch.tanh(x_l.permute(1, 0, 2))
        x_a = torch.tanh(x_a.permute(1, 0, 2))
        x_v = torch.tanh(x_v.permute(1, 0, 2))

        # 分别映射到hidden_size
        text_feat = self.text_proj(x_l)
        audio_feat = self.audio_proj(x_a)
        visual_feat = self.visual_proj(x_v)

        # Concatenate 三个模态特征，序列拉长
        fused = torch.cat([text_feat, audio_feat, visual_feat], dim=1)  # (B, 3T, hidden_size)

        # Transformer编码
        encoded = self.transformer_encoder(fused)  # (B, 3T, hidden_size)

        # 取第一帧，或者平均池化
        pooled = encoded[:, 0, :]  # (B, hidden_size)

        # 分类输出
        output = self.fc_out(pooled)  # (B, output_dim)

        return output

