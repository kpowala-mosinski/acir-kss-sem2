import os
import wave
import numpy as np
import matplotlib.pyplot as plt

ORIG_WAV = "przy_jedzeniuk.wav"  # <-- sprawdź dokładną nazwę swojego pliku WAV
M_VALUES = [2, 3, 4]


def read_wav_norm(path):
    """Wczytuje 16-bitowy WAV mono i normalizuje amplitudę do [-1, 1]."""
    with wave.open(path, "rb") as w:
        fs = w.getframerate()
        y = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return fs, y.astype(np.float64) / 32768.0


def calc_metrics(y_orig, y_rec):
    """Oblicza SNR [dB] oraz RMSE na wspólnej długości sygnałów."""
    n = min(len(y_orig), len(y_rec))
    err = y_orig[:n] - y_rec[:n]
    signal_power = np.sum(y_orig[:n] ** 2)
    noise_power = np.sum(err ** 2)
    snr = 10.0 * np.log10(signal_power / (noise_power + 1e-12))
    rmse = np.sqrt(np.mean(err ** 2))
    return snr, rmse, err


def find_voiced_window(y, win_len=300):
    """Znajduje fragment mowy o najwyższej energii (ok. 27 ms) do zbliżenia."""
    energy = np.convolve(y ** 2, np.ones(win_len) / win_len, mode="same")
    center = int(np.argmax(energy))
    start = max(0, center - win_len // 2)
    end = min(len(y), start + win_len)
    return slice(start, end)



base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, "wykresy_proj2")
os.makedirs(out_dir, exist_ok=True)

orig_path = os.path.join(base_dir, ORIG_WAV)
orig_size = os.path.getsize(orig_path)
fs, y_orig = read_wav_norm(orig_path)

results = {}
print(f"{'m':>3} | {'Rozmiar [B]':>11} | {'Kompresja':>9} | {'Redukcja':>8} | {'SNR [dB]':>8} | {'RMSE':>8}")
print("-" * 62)

for m in M_VALUES:
    bin_path = os.path.join(base_dir, f"skompresowany_m{m}.bin")
    rec_path = os.path.join(base_dir, f"zrekonstruowany_m{m}.wav")

    _, y_rec = read_wav_norm(rec_path)
    comp_size = os.path.getsize(bin_path)
    ratio = orig_size / comp_size
    reduction = (1.0 - comp_size / orig_size) * 100.0
    snr, rmse, err = calc_metrics(y_orig, y_rec)

    results[m] = {
        "y": y_rec,
        "err": err,
        "size": comp_size,
        "ratio": ratio,
        "reduction": reduction,
        "snr": snr,
        "rmse": rmse,
    }
    print(f"{m:>3} | {comp_size:>11d} | {ratio:>6.2f}:1 | {reduction:>7.1f}% | {snr:>8.2f} | {rmse:>8.5f}")

# Wspólna długość i oś czasu
n_common = min(len(y_orig), *(len(results[m]["y"]) for m in M_VALUES))
y_ref = y_orig[:n_common]
t = np.arange(n_common) / fs
zoom = find_voiced_window(y_ref, win_len=300)

# 1. Porównanie pełnych przebiegów czasowych
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, sharey=True)
axes[0].plot(t, y_ref, lw=0.4, color="black")
axes[0].set_title("Oryginał (WAV 16-bit, 11025 Hz)")
colors = {2: "tab:red", 3: "tab:orange", 4: "tab:green"}

for idx, m in enumerate(M_VALUES, start=1):
    r = results[m]
    axes[idx].plot(t, r["y"][:n_common], lw=0.4, color=colors[m])
    axes[idx].set_title(f"Rekonstrukcja m = {m} bitów (kompresja {r['ratio']:.2f}:1, SNR = {r['snr']:.2f} dB)")

for ax in axes:
    ax.set_ylabel("Amplituda")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Czas [s]")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "1_przebiegi_calosc.png"), dpi=200)

# 2. Zbliżenie fragmentu dźwięcznego (oryginał vs rekonstrukcje)
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, sharey=True)
t_zoom_ms = (t[zoom] - t[zoom.start]) * 1000.0
for idx, m in enumerate(M_VALUES):
    r = results[m]
    axes[idx].plot(t_zoom_ms, y_ref[zoom], lw=1.2, color="black", alpha=0.75, label="Oryginał")
    axes[idx].plot(t_zoom_ms, r["y"][zoom], lw=1.0, linestyle="--", color=colors[m], label=f"Rekonstrukcja m={m}")
    axes[idx].set_title(f"Zbliżenie sygnału — m = {m} bity (SNR = {r['snr']:.2f} dB)")
    axes[idx].set_ylabel("Amplituda")
    axes[idx].legend(loc="upper right")
    axes[idx].grid(alpha=0.3)
axes[-1].set_xlabel("Czas względny [ms]")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "2_porownanie_zoom.png"), dpi=200)

# 3. Błąd rekonstrukcji y(k) - y_hat(k) we wspólnej skali Y
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, sharey=True)
for idx, m in enumerate(M_VALUES):
    r = results[m]
    axes[idx].plot(t, r["err"][:n_common], lw=0.4, color=colors[m])
    axes[idx].set_title(f"Błąd rekonstrukcji dla m = {m} (RMSE = {r['rmse']:.5f})")
    axes[idx].set_ylabel("Błąd")
    axes[idx].grid(alpha=0.3)
axes[-1].set_xlabel("Czas [s]")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "3_bledy_rekonstrukcji.png"), dpi=200)

# 4. Spektrogramy (dobrze pokazują szum kwantyzacji w wyższych częstotliwościach)
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
panels = [("Oryginał", y_ref)] + [(f"Rekonstrukcja m = {m}", results[m]["y"][:n_common]) for m in M_VALUES]
for ax, (title, sig) in zip(axes.flat, panels):
    ax.specgram(sig, NFFT=256, Fs=fs, noverlap=128, cmap="inferno", vmin=-110, vmax=-20)
    ax.set_title(title)
    ax.set_ylabel("Częstotliwość [Hz]")
    ax.set_xlabel("Czas [s]")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "4_spektrogramy.png"), dpi=200)

# 5. Kompromis: stopień kompresji vs jakość (SNR)
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ratios = [results[m]["ratio"] for m in M_VALUES]
snrs = [results[m]["snr"] for m in M_VALUES]

l1 = ax1.plot(M_VALUES, ratios, "o-", color="tab:blue", lw=2, label="Stopień kompresji [:1]")
ax1.set_xlabel("Liczba bitów kwantyzacji błędu (m)")
ax1.set_ylabel("Stopień kompresji [:1]", color="tab:blue")
ax1.set_xticks(M_VALUES)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
l2 = ax2.plot(M_VALUES, snrs, "s--", color="tab:red", lw=2, label="SNR [dB]")
ax2.set_ylabel("SNR [dB]", color="tab:red")

lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper center")
ax1.set_title("Kompromis między stopniem kompresji a jakością rekonstrukcji")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "5_kompresja_vs_snr.png"), dpi=200)

print(f"\nZapisano 5 wykresów w folderze: {out_dir}")


