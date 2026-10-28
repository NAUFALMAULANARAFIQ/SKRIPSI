"""
Prediksi return JISDOR dengan LSTM: model multivariat (4 fitur) vs univariat (JISDOR saja).

Alur (mengikuti usulan skripsi):
  1. Gabung 4 data berdasarkan tanggal, hapus tanggal yang tidak lengkap (listwise deletion)
  2. Hitung return harian (%) tiap variabel
  3. Standardisasi Z-score (mean/std dari data latih saja)
  4. Sliding window, target = return JISDOR hari berikutnya (t+1)
  5. Grid search 36 skenario (unit x learning rate x batch size x sequence length),
     dipilih berdasarkan RMSE validasi terendah
  6. Uji akhir pada data 2025 + pembanding sederhana (naive dan arah mayoritas)

Pembagian data: latih 2018-2023, validasi 2024, uji 2025 (berdasarkan tanggal target).

Cara pakai:
  python lstm_jisdor.py --prep-only     # cek data saja, tanpa training
  python lstm_jisdor.py --quick         # uji cepat (grid kecil), untuk memastikan program jalan
  python lstm_jisdor.py                 # eksperimen penuh

Kebutuhan: pip install numpy pandas openpyxl tensorflow
"""
import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------------------
FILE_JISDOR = "data/jisdor.xlsx"
FILE_USD = "data/usd_index.csv"
FILE_IHSG = "data/ihsg.csv"
FILE_WTI = "data/wti.csv"

TRAIN_END = "2023-12-31"   # latih: 2018-2023
VAL_END = "2024-12-31"     # validasi: 2024, uji: 2025

UNITS = [16, 32, 64]
LEARNING_RATES = [0.001, 0.0005]
BATCH_SIZES = [16, 32]
SEQ_LENGTHS = [10, 20, 30]

EPOCHS = 100        # batas maksimum epoch
PATIENCE = 10       # early stopping pada val_loss
SEED = 42
N_FINAL_SEEDS = 5   # uji akhir diulang beberapa seed karena LSTM bersifat acak

RET_COLS = ["jisdor_return_pct", "usd_index_return_pct", "ihsg_return_pct", "wti_return_pct"]
FEATURE_SETS = {
    "multivariat": RET_COLS,
    "univariat": ["jisdor_return_pct"],
}
TARGET = "jisdor_return_pct"


# ----------------------------------------------------------------------------------------
# 1. MEMBACA DATA
# ----------------------------------------------------------------------------------------
def load_jisdor(path):
    raw = pd.read_excel(path, header=None)
    df = raw.iloc[:, 1:3].copy()
    df.columns = ["tanggal", "JISDOR"]
    df["JISDOR"] = pd.to_numeric(df["JISDOR"], errors="coerce")
    df = df.dropna(subset=["JISDOR"]).copy()  # membuang judul dan baris header Excel
    tgl = pd.to_datetime(df["tanggal"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if tgl.isna().any():  # cadangan bila format tanggal Excel berbeda
        tgl = tgl.fillna(pd.to_datetime(df["tanggal"], errors="coerce"))
    df["Date"] = tgl
    return df.dropna(subset=["Date"])[["Date", "JISDOR"]]


def _parse_usd_index(s):
    """Nilai di file FRED kamu tertulis '1.096.445' (titik ribuan), artinya 109,6445."""
    s = str(s).strip()
    if s in ("", ".", "nan", "None"):
        return np.nan
    if s.count(".") >= 2:
        return float(s.replace(".", "")) / 10000.0
    return float(s)


def load_usd(path):
    df = pd.read_csv(path, sep=";", dtype=str)
    out = pd.DataFrame({
        "Date": pd.to_datetime(df.iloc[:, 0], format="%d/%m/%Y"),
        "USD_Index": df.iloc[:, 1].map(_parse_usd_index),
    })
    return out.dropna()


def load_wti(path):
    df = pd.read_csv(path, sep=";", dtype=str)
    out = pd.DataFrame({
        "Date": pd.to_datetime(df.iloc[:, 0], format="%d/%m/%Y"),
        "WTI": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
    })
    n_bad = int((out["WTI"] <= 0).sum())
    if n_bad:
        print(f"  [peringatan] WTI: {n_bad} harga <= 0 (mis. 20/04/2020 = -36,98) dianggap tidak valid "
              "dan ikut dihapus lewat listwise deletion, karena return tidak terdefinisi pada harga negatif.")
        out.loc[out["WTI"] <= 0, "WTI"] = np.nan
    return out.dropna()


def load_ihsg(path):
    df = pd.read_csv(path, sep=";", dtype=str, encoding="utf-8-sig")
    nilai = df["Terakhir"].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    out = pd.DataFrame({
        "Date": pd.to_datetime(df["Date"], format="%d/%m/%Y"),
        "IHSG": pd.to_numeric(nilai, errors="coerce"),
    })
    return out.dropna()


def build_dataset(data_dir):
    print("Membaca data ...")
    jis = load_jisdor(os.path.join(data_dir, FILE_JISDOR))
    usd = load_usd(os.path.join(data_dir, FILE_USD))
    ihsg = load_ihsg(os.path.join(data_dir, FILE_IHSG))
    wti = load_wti(os.path.join(data_dir, FILE_WTI))
    print(f"  JISDOR={len(jis)}  USD Index={len(usd)}  IHSG={len(ihsg)}  WTI={len(wti)} baris (setelah nilai kosong dibuang)")

    df = jis.merge(usd, on="Date").merge(ihsg, on="Date").merge(wti, on="Date")  # inner join = listwise deletion
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  Setelah digabung dan listwise deletion: {len(df)} baris "
          f"({df['Date'].min().date()} s.d. {df['Date'].max().date()})")

    ret = pd.DataFrame({"Date": df["Date"]})
    ret["jisdor_return_pct"] = df["JISDOR"].pct_change() * 100
    ret["usd_index_return_pct"] = df["USD_Index"].pct_change() * 100
    ret["ihsg_return_pct"] = df["IHSG"].pct_change() * 100
    ret["wti_return_pct"] = df["WTI"].pct_change() * 100
    ret = ret.dropna().reset_index(drop=True)
    return df, ret


def describe_returns(ret):
    print("\nStatistik return (%):")
    print(ret[RET_COLS].describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
    for c in RET_COLS:
        top = ret.loc[ret[c].abs().nlargest(3).index, ["Date", c]]
        print(f"  Return ekstrem {c}: " + ", ".join(f"{d.date()} ({v:.2f})" for d, v in zip(top["Date"], top[c])))


# ----------------------------------------------------------------------------------------
# 2. SEQUENCE, METRIK, MODEL
# ----------------------------------------------------------------------------------------
def make_windows(feat_std, target_std, seq_len):
    """Jendela baris [i-L, i-1] untuk memprediksi target di baris i (t+1)."""
    n = len(feat_std)
    swv = np.lib.stride_tricks.sliding_window_view(feat_std, (seq_len, feat_std.shape[1]))[:, 0]
    X = swv[: n - seq_len].astype("float32")
    idx = np.arange(seq_len, n)
    y = target_std[idx].astype("float32")
    return X, y, idx


def metrics(y, p):
    err = y - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = y != 0  # return aktual tepat nol tidak punya arah
    if np.all(p == 0):
        da = float("nan")
    else:
        da = float(np.mean(np.sign(y[nz]) == np.sign(p[nz])) * 100)
    return {"MAE": mae, "RMSE": rmse, "DA": da}


def train_and_predict(Xtr, ytr, Xva, yva, Xte, units, lr, batch, seed, epochs, patience):
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers

    keras.utils.set_random_seed(seed)
    model = keras.Sequential([
        layers.Input(shape=Xtr.shape[1:]),
        layers.LSTM(units, return_sequences=False),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="linear"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch,
                     shuffle=True, callbacks=[stop], verbose=0)
    pva = model.predict(Xva, verbose=0).ravel()
    pte = model.predict(Xte, verbose=0).ravel()
    n_ep = len(hist.history["loss"])
    keras.backend.clear_session()
    return pva, pte, n_ep


# ----------------------------------------------------------------------------------------
# 3. EKSPERIMEN PER MODE
# ----------------------------------------------------------------------------------------
def run_mode(mode, cols, ret, mu, sd, grid, args):
    print(f"\n{'=' * 78}\nMODE: {mode.upper()}  (fitur: {', '.join(cols)})\n{'=' * 78}")
    dates = ret["Date"].values
    feat_std = ((ret[cols] - mu[cols]) / sd[cols]).values
    target_std = ((ret[TARGET] - mu[TARGET]) / sd[TARGET]).values
    y_raw = ret[TARGET].values
    t_end, v_end = np.datetime64(TRAIN_END), np.datetime64(VAL_END)

    def split(seq_len):
        X, y, idx = make_windows(feat_std, target_std, seq_len)
        d = dates[idx]
        tr, va, te = d <= t_end, (d > t_end) & (d <= v_end), d > v_end
        return X, y, idx, tr, va, te

    # --- grid search ---
    rows = []
    t0 = time.time()
    for k, (u, lr, b, L) in enumerate(grid, start=1):
        X, y, idx, tr, va, te = split(L)
        pva, _, n_ep = train_and_predict(X[tr], y[tr], X[va], y[va], X[te], u, lr, b, SEED, args.epochs, PATIENCE)
        m = metrics(y_raw[idx][va], pva * sd[TARGET] + mu[TARGET])
        rows.append({"skenario": k, "unit": u, "learning_rate": lr, "batch_size": b, "seq_len": L,
                     "epoch_berjalan": n_ep, "val_MAE": m["MAE"], "val_RMSE": m["RMSE"], "val_DA": m["DA"]})
        print(f"  [{k:>2}/{len(grid)}] unit={u:<2} lr={lr:<6} batch={b:<2} seq={L:<2} | "
              f"val RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  DA={m['DA']:.1f}%  ({n_ep} epoch)")
    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(os.path.join(args.out_dir, f"hasil_grid_{mode}.csv"), index=False)
    best = grid_df.loc[grid_df["val_RMSE"].idxmin()]
    print(f"\n  Konfigurasi terbaik (RMSE validasi terendah): skenario {int(best['skenario'])} -> "
          f"unit={int(best['unit'])}, lr={best['learning_rate']}, batch={int(best['batch_size'])}, "
          f"seq={int(best['seq_len'])}  (waktu grid {time.time() - t0:.0f} dtk)")

    # --- uji akhir: ulang beberapa seed pada konfigurasi terbaik ---
    L = int(best["seq_len"])
    X, y, idx, tr, va, te = split(L)
    y_te = y_raw[idx][te]
    finals, pred_first = [], None
    for s in range(args.seeds):
        _, pte, n_ep = train_and_predict(X[tr], y[tr], X[va], y[va], X[te], int(best["unit"]),
                                         float(best["learning_rate"]), int(best["batch_size"]),
                                         SEED + s, args.epochs, PATIENCE)
        p_te = pte * sd[TARGET] + mu[TARGET]
        m = metrics(y_te, p_te)
        finals.append({"mode": mode, "seed": SEED + s, **m})
        if pred_first is None:
            pred_first = p_te
        print(f"  Uji 2025 seed {SEED + s}: MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  DA={m['DA']:.1f}%")
    pd.DataFrame({"Date": dates[idx][te], "aktual_return_pct": y_te, "prediksi_return_pct": pred_first}) \
        .to_csv(os.path.join(args.out_dir, f"prediksi_uji_{mode}.csv"), index=False)
    return pd.DataFrame(finals), best, (idx, te, y_te)


def baselines(ret, y_te, idx_te):
    """Pembanding sederhana pada data uji yang sama."""
    y_raw = ret[TARGET].values
    train_mask = ret["Date"] <= pd.Timestamp(TRAIN_END)
    tr_y = y_raw[train_mask.values]
    major = 1.0 if (tr_y > 0).sum() >= (tr_y < 0).sum() else -1.0
    naive = y_raw[idx_te - 1]                      # return hari sebelumnya
    return {
        "Naive (return = hari sebelumnya)": metrics(y_te, naive),
        "Nol (return = 0)": metrics(y_te, np.zeros_like(y_te)),
        f"Arah mayoritas data latih ({'naik' if major > 0 else 'turun'})":
            {**metrics(y_te, np.full_like(y_te, 1e-9 * major)), "MAE": np.nan, "RMSE": np.nan},
    }


# ----------------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="folder berisi 4 file data")
    ap.add_argument("--out-dir", default="hasil", help="folder keluaran")
    ap.add_argument("--prep-only", action="store_true", help="hanya preprocessing, tanpa training")
    ap.add_argument("--quick", action="store_true", help="grid kecil + epoch pendek untuk uji program")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=N_FINAL_SEEDS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    raw, ret = build_dataset(args.data_dir)
    n_tr = int((ret["Date"] <= TRAIN_END).sum())
    n_va = int(((ret["Date"] > TRAIN_END) & (ret["Date"] <= VAL_END)).sum())
    n_te = int((ret["Date"] > VAL_END).sum())
    print(f"  Baris return: {len(ret)}  (latih={n_tr}, validasi={n_va}, uji={n_te})")
    describe_returns(ret)
    raw.to_csv(os.path.join(args.out_dir, "data_gabungan_mentah.csv"), index=False)
    ret.to_csv(os.path.join(args.out_dir, "data_return.csv"), index=False)
    if args.prep_only:
        print(f"\nSelesai. Cek file di folder '{args.out_dir}'.")
        return

    # Standardisasi: mean/std HANYA dari data latih
    train = ret[ret["Date"] <= TRAIN_END]
    mu, sd = train[RET_COLS].mean(), train[RET_COLS].std(ddof=0)
    print("\nParameter standardisasi (data latih):")
    print(pd.DataFrame({"mean": mu, "std": sd}).round(4).to_string())

    grid = list(itertools.product(UNITS, LEARNING_RATES, BATCH_SIZES, SEQ_LENGTHS))
    if args.quick:
        grid, args.epochs, args.seeds = [(16, 0.001, 32, 10), (16, 0.001, 32, 20)], 5, 2
    print(f"\nJumlah skenario per mode: {len(grid)}")

    all_final, info = [], None
    for mode, cols in FEATURE_SETS.items():
        fin, best, info = run_mode(mode, cols, ret, mu, sd, grid, args)
        all_final.append(fin)
    final = pd.concat(all_final, ignore_index=True)
    final.to_csv(os.path.join(args.out_dir, "hasil_uji_akhir_per_seed.csv"), index=False)

    # Ringkasan. Baris target uji (tahun 2025) selalu sama untuk semua seq_len, jadi pembanding sebanding.
    idx, te, y_te = info
    summary = []
    for mode in FEATURE_SETS:
        f = final[final["mode"] == mode]
        summary.append({"model": f"LSTM {mode}",
                        **{f"{c}": f"{f[c].mean():.4f} ± {f[c].std(ddof=0):.4f}" if c != 'DA'
                           else f"{f[c].mean():.2f} ± {f[c].std(ddof=0):.2f}" for c in ["MAE", "RMSE", "DA"]}})
    for name, m in baselines(ret, y_te, idx[te]).items():
        summary.append({"model": name,
                        "MAE": "-" if np.isnan(m["MAE"]) else f"{m['MAE']:.4f}",
                        "RMSE": "-" if np.isnan(m["RMSE"]) else f"{m['RMSE']:.4f}",
                        "DA": "-" if np.isnan(m["DA"]) else f"{m['DA']:.2f}"})
    summary = pd.DataFrame(summary)
    summary.to_csv(os.path.join(args.out_dir, "ringkasan_uji_2025.csv"), index=False)
    print(f"\n{'=' * 78}\nRINGKASAN UJI AKHIR 2025 (MAE/RMSE dalam % return; DA dalam %; LSTM = rata-rata ± std "
          f"dari {args.seeds} seed)\n{'=' * 78}")
    print(summary.to_string(index=False))
    print(f"\nSemua keluaran tersimpan di folder '{args.out_dir}'.")


if __name__ == "__main__":
    main()
