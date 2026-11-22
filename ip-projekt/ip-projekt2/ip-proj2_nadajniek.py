import wave
import numpy as np
import os
from bitstring import BitArray

N = 256
R = 10 
STEP = N - R

######## NADAJNIK ############
def czyt_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "Format zly"
        fs = w.getframerate()
        assert fs == 11025, f"Błędne fs: {fs} Hz"
        y = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return y.astype(np.float64)

def okno_wagowe(n_samples):
    k = np.arange(1, n_samples + 1)
    return 0.5 * (1 - np.cos((2 * np.pi / (n_samples + 1)) * k))

def calculate_autocorrelation(x, r):
    N = len(x)
    R = np.zeros(r + 1)
    for lag in range(r + 1):
        R[lag] = np.sum(x[:N - lag] * x[lag:])
    return R

def levinson_durbin(R, r):
    a = np.zeros(r + 1)
    a[0] = 1.0
    E = R[0]
    for i in range(1, r + 1):
        lambda_val = 0.0
        for j in range(1, i):
            lambda_val += a[j] * R[i - j]
        k = -(R[i] + lambda_val) / (E + 1e-12) 

        a_new = np.copy(a)
        a_new[i] = k
        for j in range(1, i):
            a_new[j] = a[j] + k * a[i - j]
        a = a_new
        E = E * (1 - k**2)
    return a[1:]

def calculate_residual_error(y_segment, history, a):
    r = len(a)
    N = len(y_segment)
    e = np.zeros(N)
    y_full = np.concatenate((history, y_segment))
    
    for k in range(N):
        past_samples = y_full[k : k + r][::-1] 
        e[k] = y_segment[k] + np.sum(a * past_samples)
    return e

def quantize_errors(e, m):
    e_max = np.max(np.abs(e))
    
    if e_max == 0:
        return 0.0, np.zeros_like(e, dtype=int)
    levels = 2**m
    e_norm = (e + e_max) / (2 * e_max)
    e_quantized_indices = np.floor(e_norm * (levels - 1) + 0.5).astype(int)
    e_quantized_indices = np.clip(e_quantized_indices, 0, levels - 1)
    
    return e_max, e_quantized_indices

def save_compressed_binary(bin_path, transmitted_data, m, fs=11025):
    num_segments = len(transmitted_data)
    
    with open(bin_path, "wb") as f:
        header = BitArray()
        header.append(BitArray(uint=m, length=8))
        header.append(BitArray(uint=R, length=8))
        header.append(BitArray(uint=N, length=16))
        header.append(BitArray(uint=fs, length=16))
        header.append(BitArray(uint=num_segments, length=32))
        header.tofile(f)
        for seg in transmitted_data:
            seg_bits = BitArray()
            for coeff in seg['a']:
                seg_bits.append(BitArray(float=float(coeff), length=32))
            seg_bits.append(BitArray(float=float(seg['e_max']), length=32))
            for val in seg['e_q']:
                seg_bits.append(BitArray(uint=int(val), length=m))
            seg_bits.tofile(f)

def report_compression(wav_path, bin_path, m):
    orig_size = os.path.getsize(wav_path)
    comp_size = os.path.getsize(bin_path)
    ratio = orig_size / comp_size
    reduction = (1.0 - comp_size / orig_size) * 100.0
    
    print(f"--- Wyniki dla kwantyzacji m = {m} bitów ---")
    print(f"Rozmiar oryginału (WAV): {orig_size} B")
    print(f"Rozmiar skompresowany:   {comp_size} B")
    print(f"Stopień kompresji:       {ratio:.2f} : 1 (oszczędność miejsca: {reduction:.1f}%)\n")


def transmitter_skeleton(y, m=2):
    window = okno_wagowe(N)
    zeros_pad = np.zeros(R)
    transmitted_data = []
    for i in range(0, len(y) - N + 1, STEP):
        segment = y[i : i + N]
        if i == 0:
            history = np.zeros(R)
        else:
            history = y[i - R : i]
        windowed_segment = segment * window
        padded_segment = np.concatenate((zeros_pad, windowed_segment, zeros_pad))
        R_vec = calculate_autocorrelation(padded_segment, R)
        a_coeffs = levinson_durbin(R_vec, R)
        e_residual = calculate_residual_error(segment, history, a_coeffs)
        e_max, e_quantized = quantize_errors(e_residual, m)
        transmitted_data.append({
            'a': a_coeffs,
            'e_max': e_max,
            'e_q': e_quantized
        })

    return transmitted_data



base_dir = os.path.dirname(os.path.abspath(__file__))
input_wav = os.path.join(base_dir, "przy_jedzeniuk.wav")  # sprawdź dokładną nazwę pliku
y = czyt_wav(input_wav)

for m in [2, 3, 4]:
    out_bin = os.path.join(base_dir, f"skompresowany_m{m}.bin")
    data = transmitter_skeleton(y, m=m)
    save_compressed_binary(out_bin, data, m=m)
    report_compression(input_wav, out_bin, m=m)