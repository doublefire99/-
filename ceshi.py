import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

MODEL_PATH = "lstm_density_best.pt"

CESHI_DIR  = "processed_files_with_density_ceshi"   
OUT_DIR    = "nn_vs_baseline_20days_out"            

PRED_START_DAY = 32     # 预测起始 day_index
PRED_DAYS = 20          # 只预测 20 天（032-051）

STATS_START_DAY = 32
STATS_END_DAY   = 365

SEQ_LEN     = 200
HIDDEN_SIZE = 64
NUM_LAYERS  = 3
DROPOUT     = 0.3

INFER_BATCH = 1024     

USE_SEMILOGY = False

torch.backends.cudnn.enabled = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================================

COL_GPS = "GPS Time (sec)"
COL_DAY = "day_index"
COL_NEW = "Density_New (kg/m^3)"
COL_PRE = "Density_pre (kg/m³)"  
ALT_COL_PRE = "Density_pre (kg/m^3)"  

COL_LOG_TRUE    = "log_rho_true"
COL_LOG_PRE     = "log_rho_pre"
COL_LOG_TRUE_N  = "log_rho_true_norm"
COL_LOG_PRE_N   = "log_rho_pre_norm"
COL_F107_N      = "F10.7_OBS_norm"
COL_F107A_N     = "F10.7_OBS_CENTER81_norm"
COL_AP_N        = "AP_norm"

FEATURE_COLS_NORM = [COL_LOG_PRE_N, COL_F107_N, COL_F107A_N, COL_AP_N]
TARGET_COL_NORM   = COL_LOG_TRUE_N

REQUIRED_COLS = [
    COL_GPS, COL_DAY, COL_NEW, COL_PRE,
    COL_LOG_TRUE, COL_LOG_PRE, COL_LOG_TRUE_N, COL_LOG_PRE_N,
    COL_F107_N, COL_F107A_N, COL_AP_N
]

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
        out, _ = self.lstm(x)       # [B, T, H]
        last_out = out[:, -1, :]    # [B, H]
        y_hat = self.fc(last_out)   # [B, 1]
        return y_hat


def ceshi_path(day: int) -> str:
    return os.path.join(CESHI_DIR, f"ceshi{day:03d}.csv")


def ensure_cols(df: pd.DataFrame, fname: str) -> pd.DataFrame:
    
    if COL_PRE not in df.columns and ALT_COL_PRE in df.columns:
        df = df.rename(columns={ALT_COL_PRE: COL_PRE})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{fname} 缺少列：{missing}")
    return df


def compute_mu_sigma_log_true(start_day: int, end_day: int):
    """
    用 ceshi 文件夹中的 log_rho_true 计算 mean/std（ddof=0）
    用于把网络输出的 pred_log_rho_true_norm 反标准化回 log_rho_true
    """
    vals = []
    for day in range(start_day, end_day + 1):
        p = ceshi_path(day)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df = ensure_cols(df, os.path.basename(p))
        x = pd.to_numeric(df[COL_LOG_TRUE], errors="coerce").astype(float).to_numpy()
        x = x[np.isfinite(x)]
        if x.size:
            vals.append(x)

    if not vals:
        return np.nan, np.nan
    z = np.concatenate(vals, axis=0)
    mu = float(np.mean(z))
    sd = float(np.std(z, ddof=0))
    if not np.isfinite(mu) or not np.isfinite(sd) or sd == 0.0:
        return np.nan, np.nan
    return mu, sd


@torch.no_grad()
def predict_norm_full(df: pd.DataFrame, model: nn.Module) -> np.ndarray:
    """
    输入：按时间排序的 df
    输出：长度 N 的 pred_norm（前 SEQ_LEN-1 为 NaN，之后为预测 log_rho_true_norm）
    """
    df = df.sort_values(by=[COL_DAY, COL_GPS]).reset_index(drop=True)
    N = len(df)
    pred = np.full((N,), np.nan, dtype=float)
    if N < SEQ_LEN:
        return pred

    feats = df[FEATURE_COLS_NORM].astype(np.float32).to_numpy()  # [N, 4]
    num_seq = N - SEQ_LEN + 1

    s = 0
    while s < num_seq:
        e = min(s + INFER_BATCH, num_seq)
        bs = e - s
        x_batch = np.zeros((bs, SEQ_LEN, feats.shape[1]), dtype=np.float32)
        for i in range(bs):
            idx = s + i
            x_batch[i] = feats[idx: idx + SEQ_LEN]

        x_t = torch.from_numpy(x_batch).to(DEVICE)
        y_hat = model(x_t).squeeze(1).detach().cpu().numpy()  # [bs]
        last_idx = np.arange(s, e) + (SEQ_LEN - 1)
        pred[last_idx] = y_hat
        s = e

    return pred


def metrics_density(y_true: np.ndarray, y_pred: np.ndarray):
    """
    在密度域计算：RMSE, MAE, Bias, MAPE（true>0）
    """
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[m]
    yp = y_pred[m]
    if yt.size == 0:
        return {"count": 0, "rmse": np.nan, "mae": np.nan, "bias": np.nan, "mape": np.nan}

    err = yp - yt
    rmse = float(math.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))

    m2 = (yt != 0) & np.isfinite(yt)
    if np.any(m2):
        mape = float(np.mean(np.abs(err[m2] / yt[m2])) * 100.0)  # %
    else:
        mape = np.nan

    return {"count": int(yt.size), "rmse": rmse, "mae": mae, "bias": bias, "mape": mape}


def improvement_pct(baseline_val: float, nn_val: float) -> float:
    """提升(%)：越大越好；若 baseline 为 0 或 NaN，则返回 NaN"""
    if not np.isfinite(baseline_val) or baseline_val == 0 or not np.isfinite(nn_val):
        return float("nan")
    return float((baseline_val - nn_val) / baseline_val * 100.0)


def plot_timeseries_density(day: int, gps: np.ndarray, true_den: np.ndarray, nn_den: np.ndarray, base_den: np.ndarray, out_dir: str):
    plt.figure()
    if USE_SEMILOGY:
        plt.semilogy(gps, true_den, label="True Density")
        plt.semilogy(gps, nn_den, label="NN Pred Density")
        plt.semilogy(gps, base_den, label="Baseline Density_pre")
    else:
        plt.plot(gps, true_den, label="True Density")
        plt.plot(gps, nn_den, label="NN Pred Density")
        plt.plot(gps, base_den, label="Baseline Density_pre")

    plt.xlabel("GPS Time (sec)")
    plt.ylabel("Density (kg/m^3)")
    plt.title(f"Day {day:03d} Density: True vs NN vs Baseline")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, f"day{day:03d}_density_timeseries.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def plot_error_hist(day: int, rel_err_nn: np.ndarray, rel_err_base: np.ndarray, out_dir: str):
    """
    画相对误差直方图（%）：(pred-true)/true * 100
    """
    plt.figure()
    plt.hist(rel_err_base, bins=80, alpha=0.6, label="Baseline rel error (%)")
    plt.hist(rel_err_nn, bins=80, alpha=0.6, label="NN rel error (%)")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Count")
    plt.title(f"Day {day:03d} Relative Error Histogram")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, f"day{day:03d}_rel_error_hist.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def plot_overall_error_hist(rel_err_nn_all: np.ndarray, rel_err_base_all: np.ndarray, out_dir: str, day0: int, day1: int):
    plt.figure()
    plt.hist(rel_err_base_all, bins=100, alpha=0.6, label="Baseline rel error (%)")
    plt.hist(rel_err_nn_all, bins=100, alpha=0.6, label="NN rel error (%)")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Count")
    plt.title(f"OVERALL Relative Error Histogram ({day0:03d}-{day1:03d})")
    plt.legend()
    plt.tight_layout()
    p = os.path.join(out_dir, f"overall_rel_error_hist_{day0:03d}_{day1:03d}.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def main():
    if not os.path.isdir(CESHI_DIR):
        raise FileNotFoundError(f"找不到 ceshi 文件夹：{CESHI_DIR}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")

    os.makedirs(OUT_DIR, exist_ok=True)
    plots_dir = os.path.join(OUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print("Using device:", DEVICE)

    # 1) 加载模型
    model = LSTMModel(
        input_size=len(FEATURE_COLS_NORM),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    print(f"[OK] Loaded model: {MODEL_PATH}")

    # 2) 计算 log_rho_true 的 mean/std，用于反标准化（log 空间）
    mu_log, sd_log = compute_mu_sigma_log_true(STATS_START_DAY, STATS_END_DAY)
    if not np.isfinite(mu_log) or not np.isfinite(sd_log):
        raise RuntimeError("无法计算 log_rho_true 的 mean/std，请确认 ceshi 文件包含 log_rho_true。")
    print(f"[OK] log_rho_true mean/std (from ceshi {STATS_START_DAY:03d}-{STATS_END_DAY:03d}): {mu_log:.6g} / {sd_log:.6g}")

    # 3) 预测 20 天 + 对比 baseline + 画图 + 汇总指标
    days = list(range(PRED_START_DAY, PRED_START_DAY + PRED_DAYS))
    summary = []
    rel_err_nn_all = []
    rel_err_base_all = []

    for day in days:
        p = ceshi_path(day)
        if not os.path.exists(p):
            print(f"[WARN] missing: {os.path.basename(p)}")
            continue

        df = pd.read_csv(p)
        df = ensure_cols(df, os.path.basename(p))
        df = df.sort_values(by=[COL_DAY, COL_GPS]).reset_index(drop=True)

        # NN 预测（norm 的 log）
        pred_norm = predict_norm_full(df, model)  # [N], 前 SEQ_LEN-1 为 NaN

        # 反标准化回 log10 密度，再还原到密度
        pred_log = pred_norm * sd_log + mu_log
        pred_den = np.power(10.0, pred_log)

        # 真实密度、baseline 密度
        true_den = pd.to_numeric(df[COL_NEW], errors="coerce").astype(float).to_numpy()
        base_den = pd.to_numeric(df[COL_PRE], errors="coerce").astype(float).to_numpy()

        # 只在 NN 有预测的位置做公平对比
        valid = np.isfinite(pred_den) & np.isfinite(true_den) & np.isfinite(base_den)
        # 同时要求 true_den > 0，避免相对误差发散
        valid = valid & (true_den > 0)

        if np.sum(valid) == 0:
            print(f"[SKIP] day {day:03d}: valid points = 0")
            continue

        yt = true_den[valid]
        ynn = pred_den[valid]
        yb = base_den[valid]

        m_nn = metrics_density(yt, ynn)
        m_b  = metrics_density(yt, yb)

        # 提升百分比（越大越好）
        imp_rmse = improvement_pct(m_b["rmse"], m_nn["rmse"])
        imp_mae  = improvement_pct(m_b["mae"],  m_nn["mae"])
        imp_mape = improvement_pct(m_b["mape"], m_nn["mape"])

        summary.append({
            "day_index": day,
            "count": m_nn["count"],
            "baseline_rmse": m_b["rmse"],
            "nn_rmse": m_nn["rmse"],
            "rmse_improve_%": imp_rmse,
            "baseline_mae": m_b["mae"],
            "nn_mae": m_nn["mae"],
            "mae_improve_%": imp_mae,
            "baseline_mape_%": m_b["mape"],
            "nn_mape_%": m_nn["mape"],
            "mape_improve_%": imp_mape,
            "baseline_bias": m_b["bias"],
            "nn_bias": m_nn["bias"],
        })

        # 保存带预测列的文件（方便你检查）
        df_out = df.copy()
        df_out["pred_log_rho_true_norm"] = pred_norm
        df_out["pred_log_rho_true"] = pred_log
        df_out["pred_Density_NN (kg/m^3)"] = pred_den
        df_out["abs_error_NN"] = pred_den - true_den
        df_out["abs_error_baseline"] = base_den - true_den

        rel_nn = np.full_like(true_den, np.nan, dtype=float)
        rel_b  = np.full_like(true_den, np.nan, dtype=float)
        rel_nn[valid] = (pred_den[valid] - true_den[valid]) / true_den[valid] * 100.0
        rel_b[valid]  = (base_den[valid] - true_den[valid]) / true_den[valid] * 100.0
        df_out["rel_error_NN_%"] = rel_nn
        df_out["rel_error_baseline_%"] = rel_b

        out_csv = os.path.join(OUT_DIR, f"pred_compare_day{day:03d}.csv")
        df_out.to_csv(out_csv, index=False, encoding="utf-8")

        # 画图：时间序列（三条曲线）
        gps = pd.to_numeric(df[COL_GPS], errors="coerce").astype(float).to_numpy()
        # 只画 valid 点对应的时间（更干净）
        gps_v = gps[valid]
        true_v = true_den[valid]
        nn_v = pred_den[valid]
        base_v = base_den[valid]

        p_ts = plot_timeseries_density(day, gps_v, true_v, nn_v, base_v, plots_dir)

        # 误差直方图：相对误差（%），NN vs baseline提醒对比“提升”
        rel_err_nn = rel_nn[valid]
        rel_err_b  = rel_b[valid]
        p_hist = plot_error_hist(day, rel_err_nn, rel_err_b, plots_dir)

        rel_err_nn_all.append(rel_err_nn)
        rel_err_base_all.append(rel_err_b)

        print(f"[OK] day {day:03d}: saved")
        print(f"  - {out_csv}")
        print(f"  - {p_ts}")
        print(f"  - {p_hist}")
        print(f"  RMSE baseline={m_b['rmse']:.4g}, NN={m_nn['rmse']:.4g}, improve={imp_rmse:.2f}%")

    # 汇总指标输出
    if summary:
        df_sum = pd.DataFrame(summary).sort_values("day_index")
        sum_path = os.path.join(OUT_DIR, "metrics_summary_20days_density.csv")
        df_sum.to_csv(sum_path, index=False, encoding="utf-8")
        print(f"\n[OK] saved summary: {sum_path}")

        # overall 汇总（拼接所有天）
        nn_all = np.concatenate(rel_err_nn_all, axis=0) if rel_err_nn_all else np.array([])
        b_all  = np.concatenate(rel_err_base_all, axis=0) if rel_err_base_all else np.array([])
        if nn_all.size and b_all.size:
            p_overall = plot_overall_error_hist(nn_all, b_all, plots_dir, days[0], days[-1])
            print(f"[OK] overall histogram: {p_overall}")

        # 总体提升（基于 summary 的加权合并更严格，这里给一个直观版本：对全体 valid 点再算一次）
        # 如果你想严格做“全体 valid 点”整体指标，我也可以再给你加一段直接累积 yt/ynn/yb 的写法。

    print(f"\n[DONE] 输出目录：{OUT_DIR}")
    print(f"      图像目录：{os.path.join(OUT_DIR, 'plots')}")


if __name__ == "__main__":
    main()
