import os
import numpy as np
import pandas as pd
from math import sqrt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.enabled = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

torch.manual_seed(42)
np.random.seed(42)

SEQ_LEN     = 200          # 时间窗口长度
BATCH_SIZE  = 256          # batch 大小，可按显存改小一点，如 128
NUM_EPOCHS  = 30           # 最大训练轮数
PATIENCE    = 5            # 早停轮数
LR          = 1e-3         # 最优学习率
HIDDEN_SIZE = 64           # 最优 hidden_size
NUM_LAYERS  = 3            # 最优 LSTM 层数
DROPOUT     = 0.3          # 最优 dropout

FEATURE_COLS = [
    "log_rho_pre_norm",
    "F10.7_OBS_norm",
    "F10.7_OBS_CENTER81_norm",
    "AP_norm",
]
TARGET_COL = "log_rho_true_norm"

# ==================== 1. 读取训练集 / 验证集 ====================

train_df = pd.read_csv("train_data.csv")
val_df   = pd.read_csv("yanzheng.csv")

train_df = train_df.sort_values(by=["day_index", "GPS Time (sec)"]).reset_index(drop=True)
val_df   = val_df.sort_values(by=["day_index", "GPS Time (sec)"]).reset_index(drop=True)

print("train rows:", len(train_df))
print("val rows:", len(val_df))


# ==================== 2. 定义时间序列 Dataset ====================

class TimeSeriesDataset(Dataset):
    """
    以滑动窗口构造样本：
    输入 X: [seq_len, num_features]
    输出 y: 标量，对应窗口最后一个时刻的 log_rho_true_norm
    """
    def __init__(self, df, feature_cols, target_col, seq_len):
        self.seq_len = seq_len
        features = df[feature_cols].values.astype(np.float32)  # [N, F]
        targets  = df[target_col].values.astype(np.float32)    # [N]

        self.features = torch.from_numpy(features)
        self.targets  = torch.from_numpy(targets)
        self.n_seq = len(df) - seq_len + 1
        if self.n_seq <= 0:
            raise ValueError("数据长度不足以构造序列，请检查 SEQ_LEN 和数据行数。")

    def __len__(self):
        return self.n_seq

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]       
        y = self.targets[idx + self.seq_len - 1]         
        return x, y


train_dataset = TimeSeriesDataset(train_df, FEATURE_COLS, TARGET_COL, SEQ_LEN)
val_dataset   = TimeSeriesDataset(val_df,   FEATURE_COLS, TARGET_COL, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print("train sequences:", len(train_dataset))
print("val sequences:", len(val_dataset))


# ==================== 3. 定义 LSTM 模型 ====================

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x: [batch, seq_len, input_size]
        """
        out, (h_n, c_n) = self.lstm(x)   # out: [B, T, H]
        last_out = out[:, -1, :]         # 取最后一个时间步 [B, H]
        y_hat = self.fc(last_out)        # [B, 1]
        return y_hat


model = LSTMModel(
    input_size=len(FEATURE_COLS),
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
).to(DEVICE)

print(model)


# ==================== 4. 定义损失函数与优化器 ====================

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ==================== 5. 验证函数 ====================

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(DEVICE)                  # [B, T, F]
            y_batch = y_batch.to(DEVICE).unsqueeze(1)     # [B, 1]

            y_pred = model(x_batch)                       # [B, 1]
            loss = criterion(y_pred, y_batch)

            bs = y_batch.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    mse = total_loss / total_count
    rmse = sqrt(mse)
    return mse, rmse


# ==================== 6. 训练循环（早停） ====================

best_val_rmse = float("inf")
best_state_dict = None
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    epoch_count = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(DEVICE)                  # [B, T, F]
        y_batch = y_batch.to(DEVICE).unsqueeze(1)     # [B, 1]

        optimizer.zero_grad()
        y_pred = model(x_batch)                       # [B, 1]
        loss = criterion(y_pred, y_batch)

        loss.backward()
        optimizer.step()

        bs = y_batch.size(0)
        epoch_loss += loss.item() * bs
        epoch_count += bs

    train_mse = epoch_loss / epoch_count
    train_rmse = sqrt(train_mse)

    val_mse, val_rmse = evaluate(model, val_loader)

    # 早停判断
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_state_dict = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    print(
        f"Epoch [{epoch:02d}/{NUM_EPOCHS}] "
        f"Train MSE: {train_mse:.6f}, RMSE: {train_rmse:.6f} | "
        f"Val MSE: {val_mse:.6f}, RMSE: {val_rmse:.6f}"
    )

    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered at epoch {epoch}")
        break


# ==================== 7. 保存最优模型 ====================

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    out_path = "lstm_density_best.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Best model saved to {out_path}, best Val RMSE = {best_val_rmse:.6f}")
else:
    print("Warning: best_state_dict is None, model not saved.")
