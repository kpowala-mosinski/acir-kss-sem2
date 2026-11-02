import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def h1_A(x):
    return x[0]*x[1] + 1.0
def h2_A(x):
    return x[1] - x[0]**2
def h1_B(x):
    return x[0]**2 + x[1]**2 - 100.0
def h2_B(x):
    return x[1] - x[0] - 1.0
def f_kary(x, h1, h2, beta1, beta2):
    return beta1 * (h1(x)**2) + beta2 * (h2(x)**2)

def rysuj_krzywe(h1, h2, x1_zakres, x2_zakres, tytul, etykieta_h1, etykieta_h2, trajektorie=None):
    X1, X2 = np.meshgrid(x1_zakres, x2_zakres)
    Z1 = np.zeros_like(X1)
    Z2 = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_punkt = [X1[i, j], X2[i, j]]
            Z1[i, j] = h1(x_punkt)
            Z2[i, j] = h2(x_punkt)
            
    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2)
    plt.contour(X1, X2, Z2, levels=[0], colors='red', linewidths=2)
    plt.plot([], [], color='blue', linewidth=2, label=etykieta_h1)
    plt.plot([], [], color='red', linewidth=2, label=etykieta_h2)
    if trajektorie:
        for traj, nazwa, kolor in trajektorie:
            plt.plot(traj[:, 0], traj[:, 1], color=kolor, marker='.', linestyle='-', label=nazwa, alpha=0.8)
            plt.plot(traj[0, 0], traj[0, 1], color=kolor, marker='o')
            plt.plot(traj[-1, 0], traj[-1, 1], color=kolor, marker='x', markersize=8)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(tytul)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()

def format_pt(x):
        return f"[{x[0]:.4f}, {x[1]:.4f}]"

def czy_rozwiazanie(x, h1, h2, tol=1e-2):
    if abs(h1(x)) < tol and abs(h2(x)) < tol:
        return "Tak"
    return "Nie (lokalne min.)"

### Hooke-Jeeves ###
def faza_probna(f, x, step_size):
    x_nowe = x.copy()
    for i in range(len(x)):
        x_test = x_nowe.copy()
        x_test[i] += step_size
        if f(x_test) < f(x_nowe):
            x_nowe = x_test
        else:
            x_test[i] -= 2 * step_size
            if f(x_test) < f(x_nowe):
                x_nowe = x_test
                
    return x_nowe

def hooke_jeeves(f, x0, step_size, tol_x, tol_f, max_iter):
    x_baza = np.array(x0, dtype=float)
    x_biezace = x_baza.copy()
    trajektoria = [x_baza.copy()]
    for i in range(max_iter):
        x_nowe = faza_probna(f, x_biezace, step_size)
        if f(x_nowe) < f(x_baza):
            x_skok = x_nowe + (x_nowe - x_baza)
            x_baza = x_nowe.copy()
            x_biezace = x_skok.copy()            
            trajektoria.append(x_baza.copy())
            
        else:
            if np.array_equal(x_biezace, x_baza):
                step_size /= 2.0
                if step_size < tol_x:
                    print(f"Hooke-Jeeves => znaleziono ekstremum w {i} krokach")
                    break
            else:
                x_biezace = x_baza.copy()
    return x_baza, np.array(trajektoria)

### Davidon ###
def estymacja_gradientu(f, x):
   h = 1e-5
   grad = np.zeros_like(x)
   for i in range(len(x)):
       x_plus = x.copy()
       x_minus = x.copy()
       x_plus[i] += h
       x_minus[i] -= h
       grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
   return grad

def wielkosc_kroku(f, x, d, step_size, tol_x):
  f_old = f(x)
  norma_d = np.linalg.norm(d)
  d_jedn = d / norma_d
  krok = step_size
  x_new = x + krok * d_jedn

  while f(x_new) >= f_old and krok > tol_x:
    krok = krok / 2.0
    x_new = x + krok * d_jedn
  if f(x_new) >= f_old:
     return 0, x
  
  return krok, x_new

def davidon(f, x0, step_size, tol_x, tol_f, max_iter):
    x = np.array(x0, dtype=float)
    trajektoria = [x.copy()]
    n = len(x)
    H = np.eye(n)
    g = estymacja_gradientu(f, x)
    for i in range(max_iter):
        d = -np.dot(H, g)
        krok, x_new = wielkosc_kroku(f, x, d, step_size, tol_x)
        if krok == 0:
            H = np.eye(n)
            d = -g
            krok, x_new = wielkosc_kroku(f, x, d, step_size, tol_x)
            if krok == 0:
                print(f"Davidon => utknął w {i} iteracji.")
                break
        zmiana_x = np.linalg.norm(x_new - x)
        zmiana_f = abs(f(x_new) - f(x))
        if zmiana_x < tol_x or zmiana_f < tol_f:
            x = x_new
            trajektoria.append(x.copy())
            print(f"Davidon => znaleziono ekstremum w {i} krokach")
            break
        g_new = estymacja_gradientu(f, x_new)
        p = x_new - x
        q = g_new - g
        mianownik1 = np.dot(p, q)
        H_razy_q = np.dot(H, q)
        mianownik2 = np.dot(q, H_razy_q)
        if mianownik1 > 1e-10 and mianownik2 > 1e-10:
            czlon1 = np.outer(p, p) / mianownik1
            czlon2 = np.outer(H_razy_q, H_razy_q) / mianownik2
            H = H + czlon1 - czlon2
        x = x_new
        g = g_new
        trajektoria.append(x.copy())
        
    return x, np.array(trajektoria)

### wykres 2d ###
def poziomice_2d(f_kary, h1, h2, x1_zakres, x2_zakres, poziomy_zwykle, poziomy_grube, tytul, trajektorie):
    X1, X2 = np.meshgrid(x1_zakres, x2_zakres)
    Z_K = np.zeros_like(X1)
    Z1 = np.zeros_like(X1)
    Z2 = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_pt = [X1[i, j], X2[i, j]]
            Z_K[i, j] = f_kary(x_pt)
            Z1[i, j] = h1(x_pt)
            Z2[i, j] = h2(x_pt)
            
    plt.figure(figsize=(10, 8))
    
    # Zwykłe poziomice (szare, cienkie)
    cp = plt.contour(X1, X2, Z_K, levels=poziomy_zwykle, colors='gray', alpha=0.5, linewidths=1)
    plt.clabel(cp, inline=True, fontsize=8)
    
    # Wybrane, grube poziomice (czarne, wyraźne)
    cp_grube = plt.contour(X1, X2, Z_K, levels=poziomy_grube, colors='black', linewidths=2)
    plt.clabel(cp_grube, inline=True, fontsize=10)
    
    # Krzywe h1=0 i h2=0
    plt.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2)
    plt.contour(X1, X2, Z2, levels=[0], colors='red', linewidths=2)
    
    # Sztuczna legenda
    plt.plot([], [], color='blue', linewidth=2, label='h1(x)=0')
    plt.plot([], [], color='red', linewidth=2, label='h2(x)=0')
    plt.plot([], [], color='black', linewidth=2, label='Wyróżnione poziomice K(x)')
    
    # Rysowanie trajektorii
    for traj, nazwa, kolor in trajektorie:
        plt.plot(traj[:, 0], traj[:, 1], color=kolor, marker='.', linestyle='-', label=nazwa, alpha=0.8)
        plt.plot(traj[0, 0], traj[0, 1], color=kolor, marker='o')
        plt.plot(traj[-1, 0], traj[-1, 1], color=kolor, marker='x', markersize=8)
        
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.title(tytul)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.show()

### wykres 3d ###
def rysuj_3d_K(K_func, x1_zakres, x2_zakres, z_max, tytul):
    X1, X2 = np.meshgrid(x1_zakres, x2_zakres)
    Z = np.zeros_like(X1)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            Z[i, j] = K_func([X1[i, j], X2[i, j]])
            
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    Z_clipped = np.clip(Z, 0, z_max)
    surf = ax.plot_surface(X1, X2, Z_clipped, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_zlim(0, z_max)
    ax.set_title(tytul)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('K(x)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

######################### main #################################
print("=== PODPUNKT 4.1 ===")
beta1 = 0.5
beta2 = 0.5
step_size_A = 1.0
tol_x_A = 1e-4
tol_f_A = 1e-4
max_iter = 30000
x0_A1 = [-2.0, 2.0]
x0_A2 = [0.6, -0.5]
step_size_B = 0.01
tol_x_B = 1e-4
tol_f_B = 1e-4
max_iter = 30000
x0_B = [0.0, 0.0]

f_celu_A = lambda x: f_kary(x, h1_A, h2_A, beta1, beta2)
# print(f"\nStart z punktu {x0_A1}:")
# print(f"Początkowa wartość funkcji kary K(x0): {f_celu_A(x0_A1):.4f}") #
x_opt_A1, traj_A1 = hooke_jeeves(f_celu_A, x0_A1, step_size_A, tol_x_A, tol_f_A, max_iter)
x_opt_Dav_A1, traj_Dav_A1 = davidon(f_celu_A, x0_A1, step_size_A, tol_x_A, tol_f_A, max_iter)
# print(f"Hooke-Jeeves x*: [{x_opt_A1[0]:.4f}, {x_opt_A1[1]:.4f}]")
# print(f"Davidon x*:      [{x_opt_Dav_A1[0]:.4f}, {x_opt_Dav_A1[1]:.4f}]")

# print(f"\nStart z punktu {x0_A2}:")
# print(f"Początkowa wartość funkcji kary K(x0): {f_celu_A(x0_A2):.4f}") #
x_opt_A2, traj_A2 = hooke_jeeves(f_celu_A, x0_A2, step_size_A, tol_x_A, tol_f_A, max_iter)
x_opt_Dav_A2, traj_Dav_A2 = davidon(f_celu_A, x0_A2, step_size_A, tol_x_A, tol_f_A, max_iter)
# print(f"Hooke-Jeeves x*: [{x_opt_A2[0]:.4f}, {x_opt_A2[1]:.4f}]")
# print(f"Davidon x*:      [{x_opt_Dav_A2[0]:.4f}, {x_opt_Dav_A2[1]:.4f}]")

trajektorie_A = [
        (traj_A1, "H-J [-2, 2]", "green"),
        (traj_A2, "H-J [0.6, -0.5]", "orange"),
        (traj_Dav_A1, "Davidon [-2, 2]", "purple"),
        (traj_Dav_A2, "Davidon [0.6, -0.5]", "magenta")
    ]

x1_A = np.linspace(-3, 3, 400)
x2_A = np.linspace(-4, 4, 400)
rysuj_krzywe(h1_A, h2_A, x1_A, x2_A, 
             "Problem A (Krzywe i trajektoria)", 
             "h1(x)=0 (x1*x2 = -1)", 
             "h2(x)=0 (x2 = x1^2)",
             trajektorie=trajektorie_A)

f_celu_B = lambda x: f_kary(x, h1_B, h2_B, beta1, beta2)
# print(f"\nStart z punktu {x0_B}:")
# print(f"Początkowa wartość funkcji kary K(x0): {f_celu_B(x0_B):.4f}")
x_opt_B, traj_B = hooke_jeeves(f_celu_B, x0_B, step_size_B, tol_x_B, tol_f_B, max_iter)
x_opt_Dav_B, traj_Dav_B = davidon(f_celu_B, x0_B, step_size_B, tol_x_B, tol_f_B, max_iter)
# print(f"Hooke-Jeeves x*: [{x_opt_B[0]:.4f}, {x_opt_B[1]:.4f}]")
# print(f"Davidon x*:      [{x_opt_Dav_B[0]:.4f}, {x_opt_Dav_B[1]:.4f}]")
trajektorie_B = [
        (traj_B, "Hooke-Jeeves [0, 0]", "green"),
        (traj_Dav_B, "Davidon [0, 0]", "purple")
    ]
x1_B = np.linspace(-18, 18, 400)
x2_B = np.linspace(-12, 12, 400)
rysuj_krzywe(h1_B, h2_B, x1_B, x2_B, 
                "Problem B: Okrąg i Prosta", 
                "h1(x)=0 (x1^2+x2^2 = 100)", 
                "h2(x)=0 (x2 = x1+1)", 
                trajektorie=trajektorie_B)
### tabela ###
print("=== TABELA PODPUNKT 4.1 ===")
dane_tabela = [
    {
        "Problem": "A", "Algorytm": "Hooke-Jeeves", "Punkt startowy": "[-2.0, 2.0]", 
        "x*": format_pt(x_opt_A1), 
        "h1(x*)": round(h1_A(x_opt_A1), 4), "h2(x*)": round(h2_A(x_opt_A1), 4), 
        "Czy rozwiązanie?": czy_rozwiazanie(x_opt_A1, h1_A, h2_A)
    },
    {
        "Problem": "A", "Algorytm": "Davidon", "Punkt startowy": "[-2.0, 2.0]", 
        "x*": format_pt(x_opt_Dav_A1), 
        "h1(x*)": round(h1_A(x_opt_Dav_A1), 4), "h2(x*)": round(h2_A(x_opt_Dav_A1), 4), 
        "Czy rozwiązanie?": czy_rozwiazanie(x_opt_Dav_A1, h1_A, h2_A)
    },
    {
        "Problem": "A", "Algorytm": "Hooke-Jeeves", "Punkt startowy": "[0.6, -0.5]", 
        "x*": format_pt(x_opt_A2), 
        "h1(x*)": round(h1_A(x_opt_A2), 4), "h2(x*)": round(h2_A(x_opt_A2), 4), 
        "Czy rozwiązanie?": czy_rozwiazanie(x_opt_A2, h1_A, h2_A)
    },
    {
        "Problem": "A", "Algorytm": "Davidon", "Punkt startowy": "[0.6, -0.5]", 
        "x*": format_pt(x_opt_Dav_A2), 
        "h1(x*)": round(h1_A(x_opt_Dav_A2), 4), "h2(x*)": round(h2_A(x_opt_Dav_A2), 4), 
        "Czy rozwiązanie?": czy_rozwiazanie(x_opt_Dav_A2, h1_A, h2_A)
    },
    {
        "Problem": "B", "Algorytm": "Hooke-Jeeves", "Punkt startowy": "[0.0, 0.0]", 
        "x*": format_pt(x_opt_B), 
        "h1(x*)": round(h1_B(x_opt_B), 4), "h2(x*)": round(h2_B(x_opt_B), 4), 
        "Czy rozwiązanie?": czy_rozwiazanie(x_opt_B, h1_B, h2_B)
    },
    {
        "Problem": "B", "Algorytm": "Davidon", "Punkt startowy": "[0.0, 0.0]", 
        "x*": format_pt(x_opt_Dav_B), 
        "h1(x*)": round(h1_B(x_opt_Dav_B), 4), "h2(x*)": round(h2_B(x_opt_Dav_B), 4), 
        "Czy rozwiązanie?": czy_rozwiazanie(x_opt_Dav_B, h1_B, h2_B)
    }
]
df_wyniki = pd.DataFrame(dane_tabela)
print(df_wyniki.to_string(index=False))

print("=== PODPUNKT 4.2 ===")
beta1 = 0.5
beta2 = 0.5
step_size_A = 1.0
tol_x_A = 1e-4
tol_f_A = 1e-4
max_iter = 30000
x0_A1 = [-2.0, 2.0]
x0_A2 = [0.6, -0.5]
step_size_B = 0.01
tol_x_B = 1e-4
tol_f_B = 1e-4
max_iter = 30000
x0_B = [0.0, 0.0]
poziomy_zwykle_A = np.linspace(0.1, 12.1, 10)
poziomy_grube_A = [0.5, 1.0, 3.0]
poziomice_2d(f_celu_A, h1_A, h2_A, x1_A, x2_A, poziomy_zwykle_A, poziomy_grube_A, 
                    "Problem A: Mapa topograficzna K(x)", trajektorie_A)
rysuj_3d_K(f_celu_A, x1_A, x2_A, z_max=12.0, tytul="Problem A: Powierzchnia 3D K(x)")
poziomy_zwykle_B = np.linspace(10, 3210, 10)
poziomy_grube_B = [50, 250, 750]
poziomice_2d(f_celu_B, h1_B, h2_B, x1_B, x2_B, poziomy_zwykle_B, poziomy_grube_B, 
                    "Problem B: Mapa topograficzna K(x)", trajektorie_B)
rysuj_3d_K(f_celu_B, x1_B, x2_B, z_max=5000.0, tytul="Problem B: Powierzchnia 3D K(x)")
dane_4_2 = [
    {
        "Problem": "A (Hiperbola/Parabola)", "Algorytm": "Hooke-Jeeves", "Punkt startowy": "[-2.0, 2.0]", 
        "x*": format_pt(x_opt_A1), "K(x*)": round(f_celu_A(x_opt_A1), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_A1, h1_A, h2_A)
    },
    {
        "Problem": "A (Hiperbola/Parabola)", "Algorytm": "Davidon", "Punkt startowy": "[-2.0, 2.0]", 
        "x*": format_pt(x_opt_Dav_A1), "K(x*)": round(f_celu_A(x_opt_Dav_A1), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_Dav_A1, h1_A, h2_A)
    },
    {
        "Problem": "A (Hiperbola/Parabola)", "Algorytm": "Hooke-Jeeves", "Punkt startowy": "[0.6, -0.5]", 
        "x*": format_pt(x_opt_A2), "K(x*)": round(f_celu_A(x_opt_A2), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_A2, h1_A, h2_A)
    },
    {
        "Problem": "A (Hiperbola/Parabola)", "Algorytm": "Davidon", "Punkt startowy": "[0.6, -0.5]", 
        "x*": format_pt(x_opt_Dav_A2), "K(x*)": round(f_celu_A(x_opt_Dav_A2), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_Dav_A2, h1_A, h2_A)
    },
    {
        "Problem": "B (Okrąg/Prosta)", "Algorytm": "Hooke-Jeeves", "Punkt startowy": "[0.0, 0.0]", 
        "x*": format_pt(x_opt_B), "K(x*)": round(f_celu_B(x_opt_B), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_B, h1_B, h2_B)
    },
    {
        "Problem": "B (Okrąg/Prosta)", "Algorytm": "Davidon", "Punkt startowy": "[0.0, 0.0]", 
        "x*": format_pt(x_opt_Dav_B), "K(x*)": round(f_celu_B(x_opt_Dav_B), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_Dav_B, h1_B, h2_B)
    }
]
print("=== TABELA PODPUNKT 4.2 ===")
print(pd.DataFrame(dane_4_2).to_string(index=False))

print("=== PODPUNKT 4.3 ===")
beta1_v2 = 0.01
beta2_v2 = 10.0
step_size_B = 0.01
tol_x_B = 1e-4
tol_f_B = 1e-4
max_iter = 30000
x0_B1_v2 = [0.0, 0.0]
x0_B2_v2 = [0.5, 1.0]
f_celu_B_v2 = lambda x: f_kary(x, h1_B, h2_B, beta1_v2, beta2_v2)
print(f"\nStart z punktu {x0_B1_v2}:")
x_opt_HJ_v2, traj_HJ_v2 = hooke_jeeves(f_celu_B_v2, x0_B1_v2, step_size_B, tol_x_B, tol_f_B, max_iter)
x_opt_Dav_v2, traj_Dav_v2 = davidon(f_celu_B_v2, x0_B1_v2, step_size_B, tol_x_B, tol_f_B, max_iter)
print(f"\nStart z punktu {x0_B2_v2}:")
x_opt_Dav_v2_pt2, traj_Dav_v2_pt2 = davidon(f_celu_B_v2, x0_B2_v2, step_size_B, tol_x_B, tol_f_B, max_iter)
trajektorie_4_3 = [
    (traj_HJ_v2, "H-J [0, 0]", "green"),
    (traj_Dav_v2, "Davidon [0, 0]", "purple"),
    (traj_Dav_v2_pt2, "Davidon [0.5, 1]", "magenta")
]
poziomy_zwykle_4_3 = np.linspace(10, 730, 10)
poziomy_grube_4_3 = [50, 200, 500] 
poziomice_2d(f_celu_B_v2, h1_B, h2_B, x1_B, x2_B, poziomy_zwykle_4_3, poziomy_grube_4_3, 
                    "Problem B (wagi: b1=0.01, b2=10)", trajektorie_4_3)
rysuj_3d_K(f_celu_B_v2, x1_B, x2_B, z_max=800.0, tytul="Problem B (wagi: b1=0.01, b2=10) - 3D")
dane_4_3 = [
    {
        "Algorytm": "Hooke-Jeeves", "Punkt startowy": "[0.0, 0.0]", 
        "x*": format_pt(x_opt_HJ_v2), "K(x*)": round(f_celu_B_v2(x_opt_HJ_v2), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_HJ_v2, h1_B, h2_B)
    },
    {
        "Algorytm": "Davidon", "Punkt startowy": "[0.0, 0.0]", 
        "x*": format_pt(x_opt_Dav_v2), "K(x*)": round(f_celu_B_v2(x_opt_Dav_v2), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_Dav_v2, h1_B, h2_B)
    },
    {
        "Algorytm": "Davidon", "Punkt startowy": "[0.5, 1.0]", 
        "x*": format_pt(x_opt_Dav_v2_pt2), "K(x*)": round(f_celu_B_v2(x_opt_Dav_v2_pt2), 4),
        "Rozwiązanie równań?": czy_rozwiazanie(x_opt_Dav_v2_pt2, h1_B, h2_B)
    }
]
print("=== TABELA PODPUNKT 4.3 ===")
print(pd.DataFrame(dane_4_3).to_string(index=False))
