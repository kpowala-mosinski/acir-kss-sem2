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




######################### main #############################
print("=== PODPUNKT 4.1: Hooke-Jeeves dla Problemu A ===")

beta1 = 0.5
beta2 = 0.5
step_size_A = 1.0
tol_x_A = 1e-4
tol_f_A = 1e-4
max_iter = 30000
x0_A1 = [-2.0, 2.0]
x0_A2 = [0.6, -0.5]

f_celu_A = lambda x: f_kary(x, h1_A, h2_A, beta1, beta2)
print(f"\nStart z punktu {x0_A1}:")
print(f"Początkowa wartość funkcji kary K(x0): {f_celu_A(x0_A1):.4f}") #
x_opt_A1, traj_A1 = hooke_jeeves(f_celu_A, x0_A1, step_size_A, tol_x_A, tol_f_A, max_iter)
print(f"Znaleziony punkt x*: [{x_opt_A1[0]:.4f}, {x_opt_A1[1]:.4f}]")
print(f"h1(x*) = {h1_A(x_opt_A1):.6f} (idealnie = 0)")
print(f"h2(x*) = {h2_A(x_opt_A1):.6f} (idealnie = 0)")

x1_A = np.linspace(-3, 3, 400)
x2_A = np.linspace(-4, 4, 400)
rysuj_krzywe(h1_A, h2_A, x1_A, x2_A, 
             "Problem A (Krzywe i trajektoria)", 
             "h1(x)=0 (x1*x2 = -1)", 
             "h2(x)=0 (x2 = x1^2)",
             trajektorie=[(traj_A1, "Hooke-Jeeves start [-2, 2]", "green")])

# Przedziały z instrukcji dla Problemu B
# x1_B = np.linspace(-18, 18, 400) #[cite: 10]
# x2_B = np.linspace(-12, 12, 400) #[cite: 10]
# rysuj_krzywe(h1_B, h2_B, x1_B, x2_B, 
#                 "Problem B: Okrąg i Prosta", 
#                 "h1(x)=0 (x1^2+x2^2 = 100)", 
#                 "h2(x)=0 (x2 = x1+1)")
