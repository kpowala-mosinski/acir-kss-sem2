import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def K1(P):
    return 4 * (P / 100)**2 + 45
def K2(P):
    return -4 * ((P - 400) / 100)**2 + 90
def K3(P):
    return 15 * (P / 100) + 40
def KS(P):
    return K1(P[0]) + K2(P[1]) + K3(P[2])

def wizualizacja_kosztow():
    P = np.linspace(0, 500, 500)
    plt.figure(figsize=(10, 6))
    plt.plot(P, K1(P), label='K1(P1)', linewidth=2, color='blue')
    plt.plot(P, K2(P), label='K2(P2)', linewidth=2, color='red')
    plt.plot(P, K3(P), label='K3(P3)', linewidth=2, color='green')
    plt.xlim(0, 500)
    plt.ylim(0, 150)
    plt.title('Koszty produkcji energii w czasie dt=1')
    plt.xlabel('Moc P [MW]')
    plt.ylabel('Koszt K(P)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.show()

def funkcja_kary(P, P_ZAP):
    h = P[0] + P[1] + P[2] - P_ZAP
    bolans_mocy = 1e6 * (h**2)
    g = [
        10 - P[0], P[0] - 400,
        10 - P[1], P[1] - 350,
        10 - P[2], P[2] - 380 
    ]
    zakres_mocy = 1e6 * sum(max(0, warunek)**2 for warunek in g)
    
    return KS(P) + bolans_mocy + zakres_mocy

def zloty_podzial(f_1d, a=-500, b=500, tol=1e-4):
    zloty_stosunek = (np.sqrt(5) - 1) / 2
    c = b - zloty_stosunek * (b - a)
    d = a + zloty_stosunek * (b - a)
    
    while abs(b - a) > tol:
        if f_1d(c) < f_1d(d):
            b = d
            d = c
            c = b - zloty_stosunek * (b - a)
        else:
            a = c
            c = d
            d = a + zloty_stosunek * (b - a)
            
    return (b + a) / 2

def powell(f, x0, tol_x=1e-4, tol_f=1e-4, max_iter=200):
    x = np.array(x0, dtype=float)
    n = len(x)
    kierunki = np.array([
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
        [1.0, 1.0, 1.0]
    ])
    for iteracja in range(max_iter):
        x_start = x.copy()
        f_start = f(x_start)
        for i in range(n):
            d = kierunki[i]
            f_kierunkowa = lambda alpha: f(x + alpha * d)
            opt_alpha = zloty_podzial(f_kierunkowa)
            x = x + opt_alpha * d
        nowy_kierunek = x - x_start
        for i in range(n - 1):
            kierunki[i] = kierunki[i + 1]
        kierunki[-1] = nowy_kierunek
        f_kierunkowa = lambda alpha: f(x + alpha * nowy_kierunek)
        opt_alpha = zloty_podzial(f_kierunkowa)
        x = x + opt_alpha * nowy_kierunek

        if np.linalg.norm(x - x_start) < tol_x and abs(f(x) - f_start) < tol_f:
            return x, True
            
    return x, False

def aktywne_ograniczenia(P):
    aktywne = []
    margines = 1e-1
    if abs(P[0] - 10) < margines: aktywne.append("P1_min")
    if abs(P[0] - 400) < margines: aktywne.append("P1_max")
    if abs(P[1] - 10) < margines: aktywne.append("P2_min")
    if abs(P[1] - 350) < margines: aktywne.append("P2_max")
    if abs(P[2] - 10) < margines: aktywne.append("P3_min")
    if abs(P[2] - 380) < margines: aktywne.append("P3_max")
    return ", ".join(aktywne) if aktywne else "Brak"


print("=== PODPUNKT 6.1 ===")
wizualizacja_kosztow()

print("=== PODPUNKT 6.2 ===")
warianty = [
        {"nazwa": "a) PZAP = 200", "P_ZAP": 200, "start1": [60, 60, 80], "start2": [100, 50, 50]},
        {"nazwa": "b) PZAP = 400", "P_ZAP": 400, "start1": [130, 130, 140], "start2": [200, 100, 100]},
        {"nazwa": "c) PZAP = 800", "P_ZAP": 800, "start1": [260, 260, 280], "start2": [300, 250, 250]}
    ]
wyniki = []
for w in warianty:
    f_celu = lambda P: funkcja_kary(P, w["P_ZAP"])
    
    for idx, start in enumerate([w["start1"], w["start2"]]):
        P_opt, sukces = powell(f_celu, start)
        
        # Wymuszamy zaokrąglenie dla czystości logów
        P_opt = np.round(P_opt, 4)
        K_min = round(KS(P_opt), 2)
        
        wyniki.append({
            "Wariant": w["nazwa"],
            "Punkt startowy": str(start),
            "P1*": P_opt[0],
            "P2*": P_opt[1],
            "P3*": P_opt[2],
            "Kmin": K_min,
            "Aktywne ograniczenia": aktywne_ograniczenia(P_opt),
            "Zbieżność (tol_x/f)": "Tak" if sukces else "Nie"
        })
print("=== TABELA 6.2 ===")
df_wyniki = pd.DataFrame(wyniki)
print(df_wyniki.to_string(index=False))
