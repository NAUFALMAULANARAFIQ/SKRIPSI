import csv
import json
import logging
import io
from pathlib import Path
from itertools import product
import joblib
import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ============================================================
# 0. KONFIGURASI UMUM
# ============================================================
SEED = 42
np.random.seed(SEED)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
parser = argparse.ArgumentParser(description="Pelatihan LSTM dengan parser CSV per sumber.")
parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "hasil")
parser.add_argument("--check-only", action="store_true", help="Periksa preprocessing dan sequence tanpa TensorFlow.")
parser.add_argument("--open-report", action="store_true", help="Buka laporan HTML di browser setelah selesai.")
args = parser.parse_args()
OUTPUT_DIR = args.output_dir
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "preprocessing.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(
            LOG_FILE,
            mode="w",
            encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger(__name__)
# Visual disimpan sebagai PNG dan dikumpulkan dalam laporan HTML lokal.
import html
import webbrowser
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
VIS_DIR = OUTPUT_DIR / "visual"
VIS_DIR.mkdir(exist_ok=True)
HISTORY_DIR = OUTPUT_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)
FIGURES = []
RAW_TABLES = []
COLORS = ["#176B87", "#C66B35", "#438C69", "#8C5698"]

def save_figure(fig, filename, title):
    fig.tight_layout()
    fig.savefig(VIS_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    FIGURES.append((filename, title))

def plot_series(frame, title, filename, unit="Nilai"):
    fig, axes = plt.subplots(len(frame.columns), 1, figsize=(13, 10), sharex=True, squeeze=False)
    for i, col in enumerate(frame.columns):
        ax = axes[i, 0]
        ax.plot(frame.index, frame[col], color=COLORS[i % 4], linewidth=0.8)
        ax.set_title(col, loc="left", fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(alpha=0.18)
    axes[-1, 0].set_xlabel("Tanggal observasi")
    fig.suptitle(title, fontsize=15)
    save_figure(fig, filename, title)

def plot_missing(frame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].imshow(frame.isna().T, aspect="auto", interpolation="nearest",
                   cmap=ListedColormap(["#176B87", "#E7B36A"]), vmin=0, vmax=1)
    axes[0].set_yticks(range(len(frame.columns)), frame.columns)
    positions = np.linspace(0, len(frame)-1, min(6, len(frame))).astype(int)
    axes[0].set_xticks(positions, frame.index[positions].strftime("%Y-%m"))
    axes[0].set_title("Biru: tersedia | Kuning: kosong")
    counts = frame.isna().sum()
    axes[1].bar(counts.index, counts, color=COLORS)
    axes[1].bar_label(axes[1].containers[0])
    axes[1].set_ylabel("Jumlah nilai kosong")
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Missing value pada gabungan tanggal semua sumber")
    save_figure(fig, "02_missing_value.png", "Missing value sebelum listwise deletion")

def plot_preprocessing(raw, clean, ret, zscore):
    plot_series(raw, "Data harga sebelum penghapusan missing value", "01_harga_awal.png")
    plot_missing(raw)
    plot_series(clean, "Data harga setelah listwise deletion", "03_harga_bersih.png")
    plot_series(ret, "Return antarobservasi lengkap yang berurutan", "04_return.png", "Return (%)")
    plot_series(zscore, "Z-score | parameter scaler hanya dari training", "05_zscore.png", "Z-score")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, col in zip(axes.flat, ret.columns):
        ax.hist(ret[col], bins=60, color="#176B87", alpha=0.8)
        ax.set_title(col)
        ax.set_xlabel("Return (%) | seluruh rentang, tanpa pemotongan outlier")
        ax.set_ylabel("Frekuensi")
    fig.suptitle("Distribusi return")
    save_figure(fig, "06_distribusi.png", "Distribusi return; nilai ekstrem tetap ditampilkan")
    fig, ax = plt.subplots(figsize=(13, 4))
    for label, mask, color in [("Training", train_mask, COLORS[0]),
                               ("Validasi", val_mask, COLORS[1]), ("Uji", test_mask, COLORS[2])]:
        ax.plot(ret.index[mask], ret.loc[mask, TARGET], label=f"{label} (n={int(mask.sum())})", color=color, lw=0.8)
    ax.set_ylabel("Return JISDOR (%)")
    ax.legend()
    ax.set_title("Pembagian berdasarkan tanggal target, sebelum sequence")
    save_figure(fig, "07_pembagian.png", "Training, validasi, dan uji")
    corr = ret.loc[train_mask].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), [x.replace("_return_pct", "") for x in corr], rotation=30)
    ax.set_yticks(range(len(corr)), [x.replace("_return_pct", "") for x in corr])
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr.iloc[i,j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Korelasi Pearson")
    ax.set_title("Korelasi return pada data training")
    save_figure(fig, "08_korelasi.png", "Korelasi training, bukan bukti sebab akibat")

def plot_history(history, scenario):
    frame = pd.DataFrame(history)
    frame.index = np.arange(1, len(frame)+1)
    frame.index.name = "epoch"
    frame.to_csv(HISTORY_DIR / f"skenario_{scenario:02d}.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame.index, frame["loss"], label="Training", color=COLORS[0])
    ax.plot(frame.index, frame["val_loss"], label="Validasi", color=COLORS[1])
    epoch_best = int(frame["val_loss"].idxmin())
    ax.axvline(epoch_best, color="#555555", ls="--", label=f"Epoch validasi terbaik: {epoch_best}")
    ax.set(xlabel="Epoch", ylabel="MSE pada skala Z-score", title=f"Kurva loss | skenario {scenario}")
    ax.legend()
    ax.grid(alpha=0.15)
    save_figure(fig, f"loss_skenario_{scenario:02d}.png", f"Training dan validation loss: skenario {scenario}")
    return epoch_best

def plot_ranking(frame):
    fig, ax = plt.subplots(figsize=(12, 8))
    ordered = frame.sort_values("val_RMSE", ascending=False)
    bars = ax.barh(ordered["scenario"].astype(str), ordered["val_RMSE"], color=COLORS[0])
    ax.bar_label(bars, fmt="%.4f", fontsize=7, padding=2)
    ax.set(xlabel="RMSE validasi (poin persentase), lebih kecil lebih baik", ylabel="Skenario",
           title="Peringkat skenario berdasarkan validasi")
    ax.margins(x=0.12)
    save_figure(fig, "09_peringkat.png", "Pemilihan model hanya berdasarkan RMSE validasi")

def prediction_diagnostics(dates, actual, predicted, split):
    residual = actual - predicted
    pd.DataFrame({"aktual_return_pct": actual, "prediksi_return_pct": predicted,
                  "residual_poin_persentase": residual}, index=pd.Index(dates, name="Date")).to_csv(
                      OUTPUT_DIR / f"prediksi_{split}.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0,0].plot(dates, actual, label="Aktual", color=COLORS[0], lw=0.9)
    axes[0,0].plot(dates, predicted, label="Prediksi", color=COLORS[1], lw=0.9)
    axes[0,0].legend()
    axes[0,0].set(title="Aktual dan prediksi", ylabel="Return (%)")
    axes[0,1].scatter(actual, predicted, s=14, alpha=0.6, color=COLORS[0])
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    axes[0,1].plot([low, high], [low, high], "--", color=COLORS[1], label="Prediksi sempurna")
    axes[0,1].set(xlabel="Aktual (%)", ylabel="Prediksi (%)", title="Kesesuaian prediksi")
    axes[1,0].plot(dates, residual, color=COLORS[2], lw=0.8)
    axes[1,0].axhline(0, color="gray", ls="--")
    axes[1,0].set(title="Residual = aktual - prediksi", ylabel="Poin persentase")
    axes[1,1].hist(residual, bins=35, color=COLORS[0], alpha=0.8)
    axes[1,1].set(title="Distribusi residual", xlabel="Poin persentase", ylabel="Frekuensi")
    for ax in (axes[0,0], axes[1,0]):
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Model terpilih | {split} | prediksi satu langkah dengan histori aktual")
    save_figure(fig, f"10_diagnostik_{split}.png", f"Diagnostik model terpilih: {split}")
    labels = [-1, 0, 1]
    matrix = np.array([[np.sum((np.sign(actual)==a) & (np.sign(predicted)==b)) for b in labels] for a in labels])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), ["Turun", "Tetap", "Naik"])
    ax.set_yticks(range(3), ["Turun", "Tetap", "Naik"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i,j]), ha="center", va="center",
                    color="white" if matrix[i,j] > matrix.max()/2 else "black")
    ax.set(xlabel="Arah prediksi", ylabel="Arah aktual", title=f"Ketepatan arah | {split}")
    save_figure(fig, f"11_arah_{split}.png", f"Confusion matrix arah: {split}; tanda nol merupakan kelas tersendiri")

def baseline_comparison(dates, actual, predicted, split):
    training = returns.loc[train_mask, TARGET]
    majority = float(np.sign(training).mode().iloc[0])
    # Arah mayoritas ditentukan hanya dari training; besaran memakai rerata absolut training.
    candidates = {"LSTM": predicted, "Return nol": np.zeros(len(actual)),
                  "Rerata training": np.full(len(actual), training.mean()),
                  "Return sebelumnya": returns[TARGET].shift(1).loc[dates].to_numpy(),
                  "Arah mayoritas training": np.full(len(actual), majority * training.abs().mean())}
    rows = []
    for label, pred in candidates.items():
        rows.append({"model": label, "MAE": mean_absolute_error(actual, pred),
                     "RMSE": np.sqrt(mean_squared_error(actual, pred)), "R2": r2_score(actual, pred),
                     "DA": np.mean(np.sign(actual)==np.sign(pred))*100})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / f"baseline_{split}.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric in zip(axes, ["RMSE", "DA"]):
        bars = ax.barh(frame["model"], frame[metric], color=[COLORS[1]]+[COLORS[0]]*4)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.set_xlabel("Poin persentase" if metric == "RMSE" else "Persen arah tepat")
        ax.set_title(metric + (" | lebih kecil lebih baik" if metric == "RMSE" else " | lebih besar lebih baik"))
        ax.margins(x=0.20)
    fig.suptitle(f"Perbandingan baseline pada tanggal yang sama | {split}")
    save_figure(fig, f"12_baseline_{split}.png", f"LSTM dan baseline: {split}")
    return frame

def write_report(status, tables=None):
    pieces = ['<!doctype html><html lang="id"><meta charset="utf-8"><title>Laporan LSTM</title>',
              '<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#203342;background:#f6f8fa}h1{color:#176B87}section{background:white;padding:24px;margin:22px 0;border-radius:10px}img{width:100%;height:auto}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border:1px solid #ddd}small{color:#596978}</style>',
              '<h1>Laporan visual LSTM JISDOR</h1><p>'+html.escape(status)+'</p>',
              '<p>Harga awal berarti data setelah format CSV, tanggal, dan angka berhasil dibaca, sebelum penghapusan missing value. Cuplikan teks sumber ditampilkan di bawah. Return dihitung antarobservasi lengkap; jarak kalender dapat berbeda. Nilai WTI negatif dipertahankan. Grafik bukan bukti kausalitas.</p>',
              '<p>Model memprediksi return satu langkah memakai histori aktual. Evaluasi ini bukan prediksi rekursif setahun penuh. MAE/RMSE memakai poin persentase; loss memakai MSE Z-score. DA membandingkan tanda negatif, nol, dan positif.</p>']
    for name, raw in RAW_TABLES:
        pieces.append('<section><h2>Cuplikan sumber: '+html.escape(name)+'</h2><pre>'+html.escape(raw)+'</pre></section>')
    for title, table in (tables or []):
        pieces.append('<section><h2>'+html.escape(title)+'</h2>'+table.to_html(index=False, float_format=lambda x:f"{x:.5f}")+'</section>')
    for filename, title in FIGURES:
        pieces.append('<section><h2>'+html.escape(title)+'</h2><img loading="lazy" src="visual/'+filename+'" alt="'+html.escape(title)+'"></section>')
    pieces.append('</html>')
    path = OUTPUT_DIR / "laporan_visual.html"
    path.write_text("\n".join(pieces), encoding="utf-8")
    return path

# ============================================================
# 1. FUNGSI LOGGING
# ============================================================
def log_df(title, df, n=5):
    log.info("=" * 70)
    log.info(title)
    log.info("Bentuk data: %s", df.shape)
    if df.empty:
        log.warning("Data kosong.")
        return
    if isinstance(df.index, pd.DatetimeIndex):
        log.info(
            "Periode: %s s.d. %s",
            df.index.min().date(),
            df.index.max().date()
        )
    log.info(
        "Nilai kosong per kolom: %s",
        df.isna().sum().to_dict()
    )
    log.info("\n%s", df.head(n).to_string())
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        desc = numeric.describe().loc[
            ["mean", "std", "min", "max"]
        ]
        log.info(
            "Statistik:\n%s",
            desc.round(4).to_string()
        )
# ============================================================
# 2. KONFIGURASI DATA
# ============================================================
FILES = {
    "jisdor": (
        "data/jisdor.csv",
        "Kurs"
    ),
    "usd_index": (
        "data/usd_index.csv",
        "DTWEXBGS"
    ),
    "ihsg": (
        "data/ihsg.csv",
        "Terakhir"
    ),
    "wti": (
        "data/wti.csv",
        "DCOILWTICO"
    ),
}
# ============================================================
# 3. FUNGSI PEMBACAAN DAN PEMBERSIHAN CSV
# ============================================================
# Header dan isi file unggahan menggunakan pemisah yang berbeda.
# Format ini disengaja untuk empat CSV yang disertakan pengguna.
CSV_FORMATS = {
    "jisdor": {"sep": ";", "date": "%m/%d/%Y %I:%M:%S %p", "decimal": "."},
    "usd_index": {"sep": ",", "date": "%Y-%m-%d", "decimal": "."},
    "ihsg": {"sep": ";", "date": "%d/%m/%Y", "decimal": ","},
    "wti": {"sep": ";", "date": "%d/%m/%Y", "decimal": "."},
}


def load_series(path, col, name):
    path = Path(path)
    cfg = CSV_FORMATS[name]
    with path.open("r", encoding="utf-8-sig") as file:
        lines = [(i, line.rstrip("\r\n")) for i, line in enumerate(file, 1) if line.strip()]
    RAW_TABLES.append((name, "\n".join(line for _, line in lines[:6])))
    if not lines:
        raise ValueError(f"File kosong: {path}")

    # Seluruh header unggahan memakai koma. IHSG memiliki satu nama kosong
    # di tengah header, sedangkan JISDOR memiliki satu nama kosong di akhir.
    headers = [cell.strip() for cell in next(csv.reader([lines[0][1]], delimiter=","))]
    headers = [cell for cell in headers if cell]
    if "Date" not in headers or col not in headers:
        raise ValueError(f"Header tidak sesuai pada {path}: {headers}")
    rows = []
    for line_no, line in lines[1:]:
        row = next(csv.reader([line], delimiter=cfg["sep"]))
        row = [cell.strip() for cell in row]
        while len(row) > len(headers) and row[-1] == "":
            row.pop()
        if len(row) != len(headers):
            raise ValueError(
                f"{path}, baris {line_no}: diharapkan {len(headers)} kolom, "
                f"ditemukan {len(row)}. Periksa pemisah {cfg['sep']!r}."
            )
        rows.append(row)
    df = pd.DataFrame(rows, columns=headers)
    dates = pd.to_datetime(df["Date"], format=cfg["date"], errors="coerce")
    if dates.isna().any():
        examples = df.loc[dates.isna(), "Date"].head().tolist()
        raise ValueError(f"Tanggal tidak sesuai format pada {path}: {examples}")
    raw = df[col].astype("string").str.strip()
    missing = raw.isin(["", ".", "-", "NA", "N/A", "NaN", "nan", "None"])
    normalized = raw.mask(missing)
    if cfg["decimal"] == ",":
        normalized = normalized.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    values = pd.to_numeric(normalized, errors="coerce")
    invalid = values.isna() & ~missing
    if invalid.any():
        raise ValueError(f"Angka tidak valid pada {path}: {raw[invalid].head().tolist()}")
    result = pd.DataFrame({"Date": dates, name: values.astype("float64")})
    if result["Date"].duplicated().any():
        raise ValueError(f"Tanggal duplikat pada {path}; periksa sumber sebelum menggabungkan.")
    result = result.dropna(subset=[name]).sort_values("Date").set_index("Date")
    if result.empty:
        raise ValueError(f"Tidak ada data valid: {path}")
    if not np.isfinite(result[name].to_numpy()).all():
        raise ValueError(f"Angka nonfinite pada {path}")
    log.info("%s: %d baris valid, %d nilai kosong, periode %s s.d. %s",
             name, len(result), int(missing.sum()), result.index.min().date(), result.index.max().date())
    log.info("Data bersih:\n%s", result.head().to_string())
    return result[[name]]

# ============================================================
# 4. PENGUMPULAN DATA
# ============================================================
series = [
    load_series(args.data_dir / Path(path).name, col, name)
    for name, (path, col) in FILES.items()
]
# ============================================================
# 5. PREPROCESSING DAN PENANGANAN MISSING VALUE
# ============================================================
for (name, (path, col)), s in zip(
    FILES.items(),
    series
):
    log.info(
        "Sumber %-10s: %d baris (%s s.d. %s)",
        name,
        len(s),
        s.index.min().date(),
        s.index.max().date()
    )
# Outer join mempertahankan seluruh tanggal dari semua sumber.
# Dropna kemudian mempertahankan hanya tanggal yang lengkap.
gabungan = pd.concat(
    series,
    axis=1,
    join="outer",
    sort=True
).sort_index()
log_df(
    "DATA GABUNGAN SEBELUM PENANGANAN MISSING VALUE",
    gabungan
)
data = gabungan.dropna().copy()
log.info(
    "Jumlah baris setelah outer join: %d",
    len(gabungan)
)
log.info(
    "Jumlah baris dihapus karena tidak lengkap: %d",
    len(gabungan) - len(data)
)
log_df(
    "DATA SETELAH LISTWISE DELETION (HARGA MENTAH)",
    data
)
if data.empty:
    raise ValueError(
        "Tidak ada tanggal yang lengkap pada seluruh sumber data. "
        "Periksa periode dan format tanggal setiap file CSV."
    )
data.to_csv(
    OUTPUT_DIR / "log_1_setelah_missing_value.csv"
)
# ============================================================
# 6. TRANSFORMASI RETURN
# ============================================================
# Return dihitung antarobservasi lengkap yang berurutan, bukan selalu
# antartanggal kalender berurutan. Tidak ada imputasi/forward fill.
returns = data.pct_change(fill_method=None).dropna() * 100
if not np.isfinite(returns.to_numpy()).all():
    raise ValueError("Return mengandung inf/NaN; periksa harga nol pada periode sebelumnya.")
if (data["wti"] <= 0).any():
    log.warning("WTI memiliki harga <= 0. Return persentase dipertahankan sesuai metode awal; tinjau interpretasinya untuk penelitian.")
returns.columns = [
    f"{col}_return_pct"
    for col in returns.columns
]
FEATURES = list(returns.columns)
TARGET = "jisdor_return_pct"
if TARGET not in FEATURES:
    raise ValueError(
        f"Target '{TARGET}' tidak ditemukan. "
        f"Fitur tersedia: {FEATURES}"
    )
target_idx = FEATURES.index(TARGET)
log_df(
    "DATA SETELAH TRANSFORMASI RETURN (%)",
    returns
)
returns.to_csv(
    OUTPUT_DIR / "log_2_return.csv"
)
if returns.empty:
    raise ValueError(
        "Data return kosong. Periksa data harga mentah."
    )
# ============================================================
# 7. PEMBAGIAN DATA BERDASARKAN PERIODE
# ============================================================
train_mask = returns.index <= "2023-12-31"
val_mask = (
    (returns.index >= "2024-01-01")
    & (returns.index <= "2024-12-31")
)
test_mask = returns.index >= "2025-01-01"
n_train = int(train_mask.sum())
n_val = int(val_mask.sum())
n_test = int(test_mask.sum())
log.info("=" * 70)
log.info(
    "PEMBAGIAN DATA | train: %d | validasi: %d | uji: %d",
    n_train,
    n_val,
    n_test
)
total = len(returns)
log.info(
    "Proporsi | train: %.1f%% | validasi: %.1f%% | uji: %.1f%%",
    100 * n_train / total,
    100 * n_val / total,
    100 * n_test / total
)
if n_train == 0:
    raise ValueError(
        "Data training kosong. "
        "Tidak ada data return sampai 31 Desember 2023."
    )
if n_val == 0:
    raise ValueError(
        "Data validasi kosong. "
        "Pastikan data mencakup tahun 2024."
    )
if n_test == 0:
    raise ValueError(
        "Data pengujian kosong. "
        "Pastikan data mencakup tahun 2025 atau sesudahnya."
    )
# ============================================================
# 8. STANDARDISASI Z-SCORE
# ============================================================
scaler = StandardScaler()
# Scaler hanya dipelajari dari data training.
scaler.fit(
    returns.loc[train_mask]
)
log.info(
    "Mean data training: %s",
    dict(zip(FEATURES, scaler.mean_.round(5)))
)
log.info(
    "Std data training: %s",
    dict(zip(FEATURES, scaler.scale_.round(5)))
)
scaled = scaler.transform(returns)
scaled_df = pd.DataFrame(
    scaled,
    index=returns.index,
    columns=FEATURES
)
log_df(
    "DATA SETELAH STANDARDISASI Z-SCORE",
    scaled_df
)
log.info(
    "Mean training setelah standardisasi: %s",
    scaled_df.loc[train_mask].mean().round(3).to_dict()
)
log.info(
    "Std training setelah standardisasi: %s",
    scaled_df.loc[train_mask].std().round(3).to_dict()
)
scaled_df.to_csv(
    OUTPUT_DIR / "log_3_standardisasi.csv"
)
# ============================================================
# 9. PEMBENTUKAN SEQUENCE (SLIDING WINDOW)
# ============================================================
plot_preprocessing(gabungan, data, returns, scaled_df)
write_report("Preprocessing selesai; pelatihan belum selesai.")

def make_sequences(arr, seq_len, target_col):
    """
    X:
        (samples, seq_len, jumlah_fitur)
    y:
        Return JISDOR pada satu periode setelah window.
    """
    X, y, idx = [], [], []
    for i in range(len(arr) - seq_len):
        X.append(
            arr[i:i + seq_len]
        )
        y.append(
            arr[i + seq_len, target_col]
        )
        # Posisi tanggal target yang diprediksi
        idx.append(
            i + seq_len
        )
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32),
        np.asarray(idx, dtype=int)
    )
def split_by_target_date(X, y, idx):
    """
    Membagi sampel berdasarkan tanggal target,
    bukan tanggal awal window.
    """
    dates = returns.index[idx]
    conditions = {
        "train": dates <= "2023-12-31",
        "val": (
            (dates >= "2024-01-01")
            & (dates <= "2024-12-31")
        ),
        "test": dates >= "2025-01-01",
    }
    parts = {}
    for name, condition in conditions.items():
        parts[name] = (
            X[condition],
            y[condition]
        )
    return parts
def log_sequence(seq_len, parts, X_all):
    log.info("=" * 70)
    log.info(
        "SEQUENCE (sequence length = %d)",
        seq_len
    )
    log.info(
        "Total sampel: %d | Bentuk X: %s",
        len(X_all),
        X_all.shape
    )
    for name, (Xp, yp) in parts.items():
        log.info(
            "%-5s -> X: %s | y: %s",
            name,
            Xp.shape,
            yp.shape
        )
    log.info(
        "Format input LSTM = (samples, timestep, features)"
    )
    if len(parts["train"][0]) == 0:
        log.warning(
            "Tidak ada sampel training untuk sequence length %d",
            seq_len
        )
        return
    X1 = parts["train"][0][0]
    y1 = parts["train"][1][0]
    index_labels = [
        (
            f"t-{seq_len - 1 - i}"
            if i < seq_len - 1
            else "t"
        )
        for i in range(seq_len)
    ]
    contoh = pd.DataFrame(
        X1,
        columns=FEATURES,
        index=index_labels
    )
    log.info(
        "Contoh X pertama (distandardisasi):\n%s",
        contoh.round(4).to_string()
    )
    log.info(
        "Contoh Y pertama (target t+1, distandardisasi): %.4f",
        y1
    )
# ============================================================
# 10. ARSITEKTUR MODEL LSTM
# ============================================================
def build_model(seq_len, n_features, units, lr):
    model = Sequential([
        Input(
            shape=(seq_len, n_features)
        ),
        LSTM(
            units,
            return_sequences=False
        ),
        Dense(
            16,
            activation="relu"
        ),
        Dense(
            1,
            activation="linear"
        ),
    ])
    model.compile(
        optimizer=Adam(
            learning_rate=lr
        ),
        loss="mse"
    )
    return model
# ============================================================
# 11. METRIK EVALUASI
# ============================================================
mu_t = scaler.mean_[target_idx]
sd_t = scaler.scale_[target_idx]
def inverse(z):
    """
    Mengembalikan nilai target dari skala Z-score
    ke skala return persentase.
    """
    return z * sd_t + mu_t
def evaluate(y_true_z, y_pred_z):
    """
    Menghitung metrik pada skala return (%).
    """
    y_true = inverse(
        np.asarray(y_true_z).ravel()
    )
    y_pred = inverse(
        np.asarray(y_pred_z).ravel()
    )
    metrics = {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "DA": float(
            np.mean(
                np.sign(y_true) == np.sign(y_pred)
            ) * 100
        ),
    }
    # R2 tidak terdefinisi dengan baik jika hanya satu sampel.
    if len(y_true) >= 2:
        metrics["R2"] = float(
            r2_score(y_true, y_pred)
        )
    else:
        metrics["R2"] = float("nan")
    return metrics
# ============================================================
# 12. SKENARIO PENGUJIAN
# ============================================================
UNITS = [16, 32, 64]
LRS = [
    0.001,
    0.0005
]
BATCHES = [16, 32]
SEQ_LENS = [10, 20, 30]
EPOCHS = 100
PATIENCE = 10
# ============================================================
# 13. MENYIAPKAN SEQUENCE SATU KALI PER SEQUENCE LENGTH
# ============================================================
sequence_cache = {}
for seq_len in SEQ_LENS:
    X_all, y_all, idx_all = make_sequences(
        scaled,
        seq_len,
        target_idx
    )
    parts = split_by_target_date(
        X_all,
        y_all,
        idx_all
    )
    sequence_cache[seq_len] = parts
    log_sequence(
        seq_len,
        parts,
        X_all
    )
    for subset_name in ["train", "val", "test"]:
        X_part, y_part = parts[subset_name]
        if len(X_part) == 0:
            raise ValueError(
                f"Tidak ada sampel {subset_name} "
                f"untuk sequence length {seq_len}. "
                "Periksa rentang tanggal dan jumlah data."
            )
# ============================================================
# 14. PELATIHAN SELURUH SKENARIO
# Total: 3 unit x 2 learning rate x 2 batch x 3 sequence
#       = 36 skenario.
# ============================================================
if args.check_only:
    log.info("PEMERIKSAAN BERHASIL: data, standardisasi, dan seluruh sequence siap.")
    report = write_report("Pemeriksaan preprocessing selesai. Mode --check-only tidak melatih model.")
    if args.open_report:
        webbrowser.open(report.resolve().as_uri())
    raise SystemExit(0)

# TensorFlow hanya diperlukan saat benar-benar melatih model.
import tensorflow as tf
from keras import Sequential
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

tf.random.set_seed(SEED)

results = []
best = {
    "rmse": np.inf,
    "config": None
}
scenario_no = 0
# PERBAIKAN:
# Tidak ada lagi nested loop pelatihan di dalam loop pelatihan.
# Setiap kombinasi hyperparameter hanya diuji satu kali.
for units, lr, batch, seq_len in product(
    UNITS,
    LRS,
    BATCHES,
    SEQ_LENS
):
    scenario_no += 1
    log.info("=" * 70)
    log.info(
        "SKENARIO %d | units=%d | lr=%s | batch=%d | seq_len=%d",
        scenario_no,
        units,
        lr,
        batch,
        seq_len
    )
    parts = sequence_cache[seq_len]
    Xtr, ytr = parts["train"]
    Xva, yva = parts["val"]
    # Membersihkan state model sebelum membangun model berikutnya
    tf.keras.backend.clear_session()
    model = build_model(
        seq_len=seq_len,
        n_features=len(FEATURES),
        units=units,
        lr=lr
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )
    history = model.fit(
        Xtr,
        ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=batch,
        verbose=0,
        callbacks=[early_stopping]
    )
    best_epoch = plot_history(history.history, scenario_no)
    write_report(f"Pelatihan berlangsung; {scenario_no} skenario sudah selesai.")
    # Prediksi data validasi
    yva_pred = model.predict(
        Xva,
        verbose=0
    )
    val_metrics = evaluate(
        yva,
        yva_pred
    )
    row = {
        "scenario": scenario_no,
        "units": units,
        "lr": lr,
        "batch": batch,
        "seq_len": seq_len,
        "epochs_max": EPOCHS,
        "epochs_actual": len(history.history["loss"]),
        "best_epoch": best_epoch,
        **{
            f"val_{key}": value
            for key, value in val_metrics.items()
        }
    }
    results.append(row)
    print(
        f"\nSkenario {scenario_no}/36"
    )
    print(
        json.dumps(
            row,
            indent=2,
            ensure_ascii=False,
            allow_nan=True
        )
    )
    log.info(
        "Metrik validasi: %s",
        val_metrics
    )
    # Model terbaik dipilih hanya berdasarkan RMSE validasi.
    if (
        np.isfinite(val_metrics["RMSE"])
        and val_metrics["RMSE"] < best["rmse"]
    ):
        best = {
            "rmse": val_metrics["RMSE"],
            "config": row.copy()
        }
        pd.DataFrame(history.history).rename_axis("epoch_zero_based").to_csv(OUTPUT_DIR / "history_terbaik.csv")
        model.save(
            OUTPUT_DIR / "model_terbaik.keras"
        )
        joblib.dump(
            scaler,
            OUTPUT_DIR / "scaler.pkl"
        )
        with open(
            OUTPUT_DIR / "config_terbaik.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                row,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=True
            )
        log.info(
            "Model terbaik sementara disimpan. RMSE validasi: %.6f",
            best["rmse"]
        )
# ============================================================
# 15. MENYIMPAN HASIL SELURUH SKENARIO
# ============================================================
if not results:
    raise RuntimeError(
        "Tidak ada hasil pelatihan yang berhasil diperoleh."
    )
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "val_RMSE",
    ascending=True
).reset_index(drop=True)
results_df.to_csv(
    OUTPUT_DIR / "hasil_skenario_lstm.csv",
    index=False
)
if best["config"] is None:
    raise RuntimeError(
        "Model terbaik tidak ditemukan. "
        "Periksa data validasi dan metrik evaluasi."
    )
print("\n" + "=" * 70)
print("KONFIGURASI MODEL TERBAIK")
print(
    json.dumps(
        best["config"],
        indent=2,
        ensure_ascii=False
    )
)
log.info(
    "Konfigurasi model terbaik: %s",
    best["config"]
)
# ============================================================
# 16. EVALUASI AKHIR PADA DATA UJI
# Model dipilih berdasarkan validasi, lalu diuji satu kali
# pada periode pengujian 2025 dan sesudahnya.
# ============================================================
best_seq_len = int(
    best["config"]["seq_len"]
)
Xte, yte = sequence_cache[best_seq_len]["test"]
best_model = tf.keras.models.load_model(
    OUTPUT_DIR / "model_terbaik.keras"
)
yte_pred = best_model.predict(
    Xte,
    verbose=0
)
test_metrics = evaluate(
    yte,
    yte_pred
)
print("\n" + "=" * 70)
print("HASIL EVALUASI DATA UJI")
print(
    json.dumps(
        test_metrics,
        indent=2,
        ensure_ascii=False,
        allow_nan=True
    )
)
log.info(
    "Hasil evaluasi data uji: %s",
    test_metrics
)
with open(
    OUTPUT_DIR / "hasil_evaluasi_test.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        test_metrics,
        file,
        indent=4,
        ensure_ascii=False,
        allow_nan=True
    )
log.info("=" * 70)
log.info("SELESAI")
log.info("Jumlah skenario: %d", len(results))
log.info("Model terbaik: model_terbaik.keras")
log.info("Scaler: scaler.pkl")
log.info("Konfigurasi: config_terbaik.json")
log.info("Hasil skenario: hasil_skenario_lstm.csv")
log.info("Evaluasi test: hasil_evaluasi_test.json")
# ============================================================
# 17. VISUAL EVALUASI MODEL TERPILIH DAN BASELINE
# ============================================================
plot_ranking(results_df)
_, _, all_idx = make_sequences(scaled, best_seq_len, target_idx)
all_dates = returns.index[all_idx]
report_tables = [("Peringkat skenario validasi", results_df)]
for split, date_mask in {
    "train": all_dates <= "2023-12-31",
    "val": (all_dates >= "2024-01-01") & (all_dates <= "2024-12-31"),
    "test": all_dates >= "2025-01-01",
}.items():
    Xpart, ypart = sequence_cache[best_seq_len][split]
    pred_z = yte_pred if split == "test" else best_model.predict(Xpart, verbose=0)
    actual, predicted = inverse(ypart), inverse(np.asarray(pred_z).ravel())
    dates = all_dates[date_mask]
    prediction_diagnostics(dates, actual, predicted, split)
    comparison = baseline_comparison(dates, actual, predicted, split)
    report_tables.append((f"Model terpilih dan baseline: {split}", comparison))
report = write_report("Pelatihan dan evaluasi selesai. Training adalah evaluasi in-sample. Model dipilih berdasarkan validasi; uji tidak dipakai untuk memilih model.", report_tables)
log.info("Laporan visual: %s", report.resolve())
if args.open_report:
    webbrowser.open(report.resolve().as_uri())
