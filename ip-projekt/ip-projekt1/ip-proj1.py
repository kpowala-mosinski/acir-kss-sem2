# ip projekt 1 Karolina Grodz 188664, Kajetan Powala-Mosinski 193054, acir kss1
import os
import sys
import wave
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

R = 4 # rzad modelu ar
LAMBDA = 0.999 # wspolczynnik zampominania ew-ls
M = 20 # okno estymatora wariancji bledu predykcji
MU = 3.0 # prog detekcji
MAX_BURST = 4 # max dlugosc serii zaklocen
DELTA = 100.0 # inicjalizacja macierzy kowariancji
P_TRACE_MAX = 1e4 # gorne ograniczenie sladu macierzy p
WARMUP = 500 # liczba probek na zbieznosc estymatorow

# odczytanie pliku wav
def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        fs = w.getframerate()
        y = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return fs, y.astype(np.float64) / 32768.0

# zapisywanie pliku wav
def write_wav(path, fs, y):
    data = np.clip(np.round(y * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data.tobytes())

# obliczanie wskaznika wielokrokowego
def multistep_gain(theta, h):
    psi = [1.0]
    for i in range(1, h):
        psi.append(sum(theta[j] * psi[i - 1 - j] for j in range(min(i, R)))) # sum_{i=0}^{h-1} psi_i^2
    return sum(p * p for p in psi)

# wyznaczenie dlugosci serii zaklocen od y(k)
def burst_length(x, y, k, theta, sigma2):
    hist = list(x[k - R:k])
    for m in range(MAX_BURST + 1):
        if k + m >= len(y):
            return m
        pred = theta @ hist[:-R - 1:-1]
        if m > 0 and abs(y[k + m] - pred) <= MU * np.sqrt(sigma2 * multistep_gain(theta, m + 1)):
            return m
        hist.append(pred)
    return 0

def remove_clicks(y):
    n = len(y)
    x = y.copy()
    theta = np.zeros(R) # [a_1, ..., a_R]
    P = DELTA * np.eye(R)
    e2_window = deque(maxlen=M)

    flags = np.zeros(n, dtype=bool)
    thetas = np.zeros((n, R))
    errors = np.zeros(n)
    thresholds = np.full(n, np.nan)

    k = R
    while k < n:
        phi = x[k - 1:k - R - 1:-1] if k > R else x[k - 1::-1]   # [x(k-1) ... x(k-R)]
        e = y[k] - theta @ phi
        sigma2 = sum(e2_window) / len(e2_window) if e2_window else e * e
        thr = MU * np.sqrt(sigma2)
        errors[k], thresholds[k], thetas[k] = e, thr, theta

        if k >= WARMUP and abs(e) > thr:
            m = burst_length(x, y, k, theta, sigma2)
            if m > 0: # interpolacja liniowa miedzy x(k-1) a y(k+m), dla m = 1: x(k) = 0.5 * [x(k-1) + y(k+1)]
                left = x[k - 1]
                right = y[k + m] if k + m < n else left
                x[k:k + m] = left + (right - left) * np.arange(1, m + 1) / (m + 1)
                flags[k:k + m] = True
                thresholds[k:k + m] = thr
                thetas[k:k + m] = theta
                for i in range(k + 1, k + m):
                    errors[i] = y[i] - theta @ x[i - 1:i - R - 1:-1]
                k += m
                continue

        Pphi = P @ phi
        den = LAMBDA + phi @ Pphi
        theta = theta + Pphi * (e / den)
        P = (P - np.outer(Pphi, Pphi) / den) / LAMBDA
        P = 0.5 * (P + P.T)
        tr = np.trace(P)
        if tr > P_TRACE_MAX:
            P *= P_TRACE_MAX / tr
        e2_window.append(e * e)
        thetas[k] = theta
        k += 1

    return x, flags, thetas, errors, thresholds

# wykresy
def plot_results(fs, y, y_clean, flags, thetas, errors, thresholds, out_dir, show):
    os.makedirs(out_dir, exist_ok=True)
    t = np.arange(len(y)) / fs
    idx = np.flatnonzero(flags)

    c = np.argmax(np.abs(y - y_clean))
    zoom = slice(max(c - 100, 0), min(c + 100, len(y)))

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True, sharey=True)
    ax[0].plot(t, y, lw=0.4)
    ax[0].set_title("Sygnał przed odszumianiem")
    ax[1].plot(t, y_clean, lw=0.4, color="tab:green")
    ax[1].set_title("Sygnał po odszumianiu")
    ax[1].set_xlabel("t [s]")
    for a in ax:
        a.set_ylabel("amplituda")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sygnal_przed_po.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[zoom], y[zoom], ".-", lw=0.8, label="przed")
    ax.plot(t[zoom], y_clean[zoom], ".-", lw=0.8, label="po")
    z = idx[(idx >= zoom.start) & (idx < zoom.stop)]
    ax.plot(t[z], y[z], "rx", label="próbki zakwestionowane")
    ax.set_title("Powiększenie fragmentu z zakłóceniem")
    ax.set_xlabel("t [s]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "zoom.png"), dpi=150)

    fig, ax = plt.subplots(R, 1, figsize=(12, 8), sharex=True)
    for i in range(R):
        ax[i].plot(t, thetas[:, i], lw=0.5)
        ax[i].set_ylabel(f"$a_{i + 1}$")
        ax[i].grid(alpha=0.3)
    ax[0].set_title(f"Przebieg parametrów modelu AR({R}) - EW-LS, $\\lambda$ = {LAMBDA}")
    ax[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "parametry_ar.png"), dpi=150)

    fig, ax = plt.subplots(2, 1, figsize=(12, 7))
    ax[0].plot(t, np.abs(errors), lw=0.4, label="|e(k)|")
    ax[0].plot(t, thresholds, lw=0.6, color="tab:red", label=f"{MU:g}$\\sigma_e(k)$")
    ax[0].set_title("Moduł błędu predykcji i próg detekcji")
    ax[0].set_xlabel("t [s]")
    ax[0].legend(loc="upper right")
    ax[1].plot(t[zoom], np.abs(errors[zoom]), ".-", lw=0.8, label="|e(k)|")
    ax[1].plot(t[zoom], thresholds[zoom], color="tab:red", label=f"{MU:g}$\\sigma_e(k)$")
    ax[1].plot(t[z], np.abs(errors[z]), "kx", label="próbki zakwestionowane")
    ax[1].set_title("Powiększenie")
    ax[1].set_xlabel("t [s]")
    ax[1].legend(loc="upper right")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "prog_detekcji.png"), dpi=150)

    if show:
        plt.show()

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    here = os.path.dirname(os.path.abspath(__file__))
    in_path = args[0] if args else os.path.join(here, "02.wav")
    out_path = os.path.splitext(in_path)[0] + "_odszumiony.wav"

    fs, y = read_wav(in_path)
    y_clean, flags, thetas, errors, thresholds = remove_clicks(y)
    write_wav(out_path, fs, y_clean)

    edges = np.diff(np.concatenate(([0], flags.astype(np.int8), [0])))
    lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    print(f"Plik: {in_path}, fs = {fs} Hz, {len(y)} próbek ({len(y) / fs:.1f} s)")
    print(f"Zakwestionowane próbki: {flags.sum()} w {len(lengths)} seriach")
    for m in range(1, MAX_BURST + 1):
        print(f"  serie o długości {m}: {(lengths == m).sum()}")
    print(f"Zapisano: {out_path}")

    plot_results(fs, y, y_clean, flags, thetas, errors, thresholds,
                 os.path.join(here, "wykresy"), "--show" in sys.argv)

if __name__ == "__main__":
    main()