import os
import struct
import wave
import numpy as np


def load_compressed_binary(bin_path):
    """
    Odczytuje nagłówek oraz wszystkie zakodowane segmenty z pliku binarnego
    (big-endian, zgodnie z formatem zapisanym przez BitArray w nadajniku).
    """
    with open(bin_path, "rb") as f:
        # 1. Odczyt nagłówka (10 bajtów: uint8, uint8, uint16, uint16, uint32)
        m, r, n_seg, fs, num_segments = struct.unpack(">BBHHI", f.read(10))

        # Wagi do zamiany m bitów na liczbę całkowitą (np. dla m=3: [4, 2, 1])
        bit_weights = 1 << np.arange(m - 1, -1, -1, dtype=np.int64)
        float_fmt = f">{r + 1}f"
        float_bytes = (r + 1) * 4
        err_bytes = (n_seg * m + 7) // 8

        # 2. Odczyt kolejnych segmentów
        segments = []
        for _ in range(num_segments):
            vals = struct.unpack(float_fmt, f.read(float_bytes))
            a = np.array(vals[:r], dtype=np.float64)
            e_max = vals[r]

            # Rozpakowanie dokładnie m-bitowych indeksów błędów z surowych bajtów
            raw_err = np.frombuffer(f.read(err_bytes), dtype=np.uint8)
            bits = np.unpackbits(raw_err)[: n_seg * m].reshape(n_seg, m)
            e_q = bits.astype(np.int64) @ bit_weights

            segments.append({
                "a": a,
                "e_max": e_max,
                "e_q": e_q,
            })

    return m, r, n_seg, fs, segments


def dequantize_errors(e_q, e_max, m):
    """
    Odtwarza przybliżone wartości błędu resztowego z przedziału [-e_max, e_max].
    """
    if e_max == 0.0:
        return np.zeros(len(e_q), dtype=np.float64)

    levels = 2**m
    return (e_q / (levels - 1)) * (2.0 * e_max) - e_max


def reconstruct_segment(a, e_deq, history):
    """
    Rekonstruuje jeden segment N próbek za pomocą filtru 1 / A(z^-1):
    y(k) = -[a_1*y(k-1) + ... + a_r*y(k-r)] + e(k)
    """
    r = len(a)
    n_seg = len(e_deq)

    y_full = np.zeros(r + n_seg, dtype=np.float64)
    y_full[:r] = history

    for k in range(n_seg):
        # Pobieramy r poprzednich próbek: [y(k-1), y(k-2), ..., y(k-r)]
        past = y_full[k : k + r][::-1]
        y_full[r + k] = -np.dot(a, past) + e_deq[k]

    return y_full[r:]


def receiver(bin_path):
    """
    Dekoduje cały plik binarny i skleja segmenty w jeden ciągły sygnał mowy.
    """
    m, r, n_seg, fs, segments = load_compressed_binary(bin_path)
    step = n_seg - r
    total_len = n_seg + (len(segments) - 1) * step

    y_out = np.zeros(total_len, dtype=np.float64)
    history = np.zeros(r, dtype=np.float64)

    # Wagi do płynnego sklejenia (crossfade) 10 próbek na zakładce
    fade_in = np.linspace(0.0, 1.0, r)
    fade_out = 1.0 - fade_in

    for idx, seg in enumerate(segments):
        e_deq = dequantize_errors(seg["e_q"], seg["e_max"], m)
        y_rec = reconstruct_segment(seg["a"], e_deq, history)

        start = idx * step
        if idx == 0:
            y_out[:n_seg] = y_rec
        else:
            # Płynne połączenie na zakładce r=10 próbek
            y_out[start : start + r] = (
                y_out[start : start + r] * fade_out + y_rec[:r] * fade_in
            )
            # Dopisanie pozostałych N - R = 246 nowych próbek
            y_out[start + r : start + n_seg] = y_rec[r:]

        # Historia dla kolejnego segmentu (próbki bezpośrednio przed następną zakładką)
        history = y_rec[n_seg - 2 * r : n_seg - r]

    return fs, y_out


def write_wav_11025(path, fs, y, normalized=False):
    """
    Zapisuje zrekonstruowany sygnał jako 16-bitowy WAV mono.
    Ustaw normalized=True TYLKO jeśli w nadajniku dzieliłaś próbki przez 32768.0.
    """
    if normalized:
        y = y * 32768.0

    data = np.clip(np.round(y), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes(data.tobytes())


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for m in [2, 3, 4]:
        in_bin = os.path.join(base_dir, f"skompresowany_m{m}.bin")
        out_wav = os.path.join(base_dir, f"zrekonstruowany_m{m}.wav")

        fs, y_rec = receiver(in_bin)
        # UWAGA: jeśli w czyt_wav w nadajniku nie dzieliłaś przez 32768.0, zostaw normalized=False
        write_wav_11025(out_wav, fs, y_rec, normalized=False)
        print(f"Zapisano zrekonstruowany plik: {out_wav} ({len(y_rec) / fs:.2f} s)")