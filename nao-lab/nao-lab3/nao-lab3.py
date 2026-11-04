import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

###### FUNKCJE ######
def f_gold(x1, x2):
   czesc1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
   czesc2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
   return czesc1 * czesc2

def ograniczenia(x1, x2):
   g1 = -0.5 - x1
   g2 = x1 - 0.4
   g3 = -1.3 - x2
   g4 = x2 + 0.5
   return g1, g2, g3, g4

def f_kary(x1, x2, alfa):
   suma = sum(np.maximum(0, g)**2 for g in ograniczenia(x1, x2))
   return alfa * suma

def p_ograniczona(x1, x2, alfa):
   return -f_gold(x1, x2) + f_kary(x1, x2, alfa)

###### HOOKE-JEEVES ######
def probny_krok(f, x, f_x, krok):
   # ruch próbny - kolejno wzdłuż każdej osi w obie strony
   x_nowy = x.copy()
   f_nowy = f_x
   for i in range(len(x)):
      for kierunek in (1, -1):
         x_test = x_nowy.copy()
         x_test[i] += kierunek * krok
         f_test = f(x_test[0], x_test[1])
         if f_test < f_nowy:
            x_nowy, f_nowy = x_test, f_test
            break
   return x_nowy, f_nowy

def hooke_jeeves(f, x0, step_size, tol_x, tol_f, max_iter, granica=1e6):
   x_baza = np.array(x0, dtype=float)
   f_baza = f(x_baza[0], x_baza[1])
   krok = step_size
   trajektoria = [x_baza.copy()]
   # punkt, wokół którego wykonywany jest ruch próbny (baza lub punkt po ruchu roboczym)
   x_start, f_start = x_baza, f_baza
   po_ruchu_roboczym = False

   for i in range(max_iter):
      x_nowy, f_nowy = probny_krok(f, x_start, f_start, krok)

      if f_nowy < f_baza:
         zmiana_x = np.linalg.norm(x_nowy - x_baza)
         zmiana_f = abs(f_nowy - f_baza)
         # ruch roboczy - przesunięcie w kierunku poprawy
         x_wzorzec = x_nowy + (x_nowy - x_baza)
         x_baza, f_baza = x_nowy, f_nowy
         trajektoria.append(x_baza.copy())

         if not np.isfinite(f_baza) or np.linalg.norm(x_baza) > granica:
            print(f"hooke_jeeves => rozbieżność po {i} iteracjach")
            return x_baza, np.array(trajektoria), "rozbieżność (P -> -inf)"
         if zmiana_x < tol_x or zmiana_f < tol_f:
            print(f"hooke_jeeves => znaleziono minimum w {i} iteracjach")
            return x_baza, np.array(trajektoria), "zbieżność"

         x_start, f_start = x_wzorzec, f(x_wzorzec[0], x_wzorzec[1])
         po_ruchu_roboczym = True
      elif po_ruchu_roboczym:
         # ruch roboczy nieudany - powrót do bazy
         x_start, f_start = x_baza, f_baza
         po_ruchu_roboczym = False
      else:
         krok = krok / 2
         if krok < tol_x:
            print(f"hooke_jeeves => znaleziono minimum w {i} iteracjach")
            return x_baza, np.array(trajektoria), "zbieżność"

   print(f"hooke_jeeves => przekroczono max_iter")
   return x_baza, np.array(trajektoria), "przekroczono max_iter"

###### WYKRESY ######
X1_ZAKRES = (-2, 2)
X2_ZAKRES = (-3.5, 1.5)

def siatka(n):
   x1 = np.linspace(*X1_ZAKRES, n)
   x2 = np.linspace(*X2_ZAKRES, n)
   return np.meshgrid(x1, x2)

def rysuj_obszar():
   prostokat = patches.Rectangle((-0.5, -1.3), 0.4 - (-0.5), -0.5 - (-1.3),
                                 linewidth=2, edgecolor='red', facecolor='none',
                                 linestyle='--', label='Obszar dopuszczalny Φ')
   plt.gca().add_patch(prostokat)

def wykres_2d(f, poziomy, tytul, trajektoria=None):
   X1, X2 = siatka(500)
   Z = f(X1, X2)

   plt.figure(figsize=(8, 6))
   poziomice = plt.contour(X1, X2, Z, levels=poziomy, cmap='viridis', linewidths=0.8)
   plt.clabel(poziomice, fontsize=7)
   plt.colorbar(poziomice, label='Wartość funkcji')
   rysuj_obszar()
   if trajektoria:
      for traj, nazwa, kolor in trajektoria:
         plt.plot(traj[:, 0], traj[:, 1], color=kolor, marker='.', linestyle='-', label=nazwa, alpha=0.8)
         plt.plot(traj[0, 0], traj[0, 1], color=kolor, marker='o')  # Kropka na starcie
         plt.plot(traj[-1, 0], traj[-1, 1], color=kolor, marker='x', markersize=8)  # X na końcu
   plt.xlim(*X1_ZAKRES)
   plt.ylim(*X2_ZAKRES)
   plt.legend()
   plt.xlabel('x1')
   plt.ylabel('x2')
   plt.title(tytul)
   plt.grid(True, linestyle='--', alpha=0.3)
   plt.show()

def wykres_3d(f, z_min, z_max, tytul):
   X1, X2 = siatka(300)
   Z = f(X1, X2)
   Z_przyciete = np.clip(Z, z_min, z_max)
   fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(111, projection='3d')
   powierzchnia = ax.plot_surface(X1, X2, Z_przyciete,
                                  cmap='viridis',
                                  linewidth=0,
                                  antialiased=True,
                                  rstride=3, cstride=3)

   ax.set_zlim(z_min, z_max)
   ax.view_init(elev=30, azim=120)
   ax.set_xlabel('x1')
   ax.set_ylabel('x2')
   ax.set_zlabel('wartość funkcji')
   plt.title(tytul)
   fig.colorbar(powierzchnia, ax=ax, label='Wartość funkcji', shrink=0.7)
   plt.show()

###### 3.1 Wizualizacja zewnętrznej funkcji kary ######
K_10 = lambda x1, x2: f_kary(x1, x2, 10)
K_1e5 = lambda x1, x2: f_kary(x1, x2, 1e5)

wykres_2d(K_10, np.linspace(0.1, 49.1, 10), 'K(α, x) dla α = 10')
wykres_2d(K_1e5, np.linspace(10, 40010, 10), 'K(α, x) dla α = 1e5')
wykres_3d(K_10, 0, 40000, 'K(α, x) dla α = 10')
wykres_3d(K_1e5, 0, 40000, 'K(α, x) dla α = 1e5')

###### 3.2 Wizualizacja funkcji faktycznie minimalizowanej ######
P_10 = lambda x1, x2: p_ograniczona(x1, x2, 10)
P_1e5 = lambda x1, x2: p_ograniczona(x1, x2, 1e5)

poziomy_10 = [-40000, -30000, -20000, -10000, -3000, -1000, -300, -100, -33, -10]
poziomy_1e5 = [-2000, -1000, -500, -300, -100, -40, -10, 111, 1000, 2000]

wykres_2d(P_10, poziomy_10, 'P(α, x) dla α = 10')
wykres_3d(P_10, -40000, 0, 'P(α, x) dla α = 10')
wykres_2d(P_1e5, poziomy_1e5, 'P(α, x) dla α = 1e5')
wykres_3d(P_1e5, -3000, 3000, 'P(α, x) dla α = 1e5')

###### 3.3 Wpływ współczynnika kary i kroku początkowego ######
alfa = 1e4
P_1e4 = lambda x1, x2: p_ograniczona(x1, x2, alfa)

wykres_2d(P_1e4, poziomy_1e5, 'P(α, x) dla α = 1e4')
wykres_3d(P_1e4, -3000, 3000, 'P(α, x) dla α = 1e4')
wykres_3d(P_1e4, -40000, 40000, 'P(α, x) dla α = 1e4 (szerszy zakres)')

# Oczekiwania na podstawie kształtu P:
# - start [0.4, -0.6] z małym krokiem -> lokalne minimum P przy górnej granicy Φ; przy α = 1e4 kara jest
#   za słaba, więc minimum P leży nieco poza Φ (g4(x*) > 0),
# - start [0.4, -0.6] z krokiem 100 -> pierwszy ruch próbny wychodzi daleko poza Φ, gdzie -f_Gold (stopień 8)
#   dominuje nad karą kwadratową, więc P -> -inf i algorytm ucieka (rozbieżność),
# - start [0.0, -1.2] z małym krokiem -> P maleje w stronę dolnej granicy Φ, a kara α = 1e4 nie zatrzymuje
#   wzrostu f_Gold poza Φ -> algorytm przekracza granicę x2 = -1.3 i ucieka do -inf.
tol_x = 1e-4
tol_f = 1e-4
max_iter = 30000
zestawy = [
   ([0.4, -0.6], 0.01, "red"),
   ([0.4, -0.6], 100, "blue"),
   ([0.0, -1.2], 0.01, "green"),
]

def format_pt(x):
   return f"[{x[0]:.4g}, {x[1]:.4g}]"

wyniki = []
trajektorie = []
for x0, step_size, kolor in zestawy:
   x_hj, trajektoria_hj, powod = hooke_jeeves(P_1e4, x0, step_size, tol_x, tol_f, max_iter)
   if powod == "zbieżność":
      wynik_x = format_pt(x_hj)
      wynik_p = f"{P_1e4(x_hj[0], x_hj[1]):.4g}"
      trajektorie.append((trajektoria_hj, f"HJ x0={x0}, krok={step_size}", kolor))
   else:
      wynik_x = powod
      wynik_p = "-"
   wyniki.append({
      "Punkt startowy": str(x0),
      "step_size": step_size,
      "x* / Komentarz": wynik_x,
      "P(x*)": wynik_p,
   })

wykres_2d(P_1e4, poziomy_1e5, 'Trajektorie Hooke\'a-Jeevesa, P(α, x) dla α = 1e4', trajektoria=trajektorie)

df_hj = pd.DataFrame(wyniki)
print("\nTabela dla podpunktu 3.3 (metoda Hooke'a-Jeevesa, alfa = 1e4)")
print(df_hj.to_string(index=False))
