import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

##### GRADIENT SPRZĘŻONY ######
def estymacja_gradientu(f, x):
   h = 1e-5
   grad = np.zeros(2)
   grad[0] = (f(x[0] + h, x[1]) - f(x[0] - h, x[1])) / (2 * h)
   grad[1] = (f(x[0], x[1] + h) - f(x[0], x[1] - h)) / (2 * h)
   return grad

def wielkosc_kroku(f, x, d, step_size, tol_x):
  f_old = f(x[0], x[1])
  norma_d = np.linalg.norm(d)
  d_jedn = d / norma_d
  krok = step_size
  x_new = x + krok * d_jedn

  while f(x_new[0], x_new[1]) >= f_old and krok > tol_x:
    krok = krok / 2
    x_new = x + krok * d_jedn
  if f(x_new[0], x_new[1]) >= f_old:
     return 0, x
  
  return krok, x_new

def g_s(f, x0, step_size, tol_x, tol_f, max_iter):
   x = np.array(x0, dtype=float)
   trajektoria = [x.copy()] 
   g_stary = estymacja_gradientu(f,x)
   d = -g_stary

   for i in range(max_iter):
      krok, x_new = wielkosc_kroku(f, x, d, step_size, tol_x)

      if krok == 0:
         d = -g_stary
         krok, x_new = wielkosc_kroku(f, x, d, step_size, tol_x)

      zmiana_x = np.linalg.norm(x_new - x)
      zmiana_f = abs(f(x_new[0], x_new[1]) - f(x[0], x[1]))
      if zmiana_x < tol_x or zmiana_f < tol_f:  # waruneki zatrzymania
         x = x_new
         trajektoria.append(x.copy())
         print(f"g_s => znaleziono ekstremum w {i} krokach")
         break
      g_new = estymacja_gradientu(f, x_new)
      beta = np.dot(g_new, g_new) / np.dot(g_stary, g_stary)
      d = -g_new + beta * d

      x = x_new
      g_stary = g_new
      trajektoria.append(x.copy())

   return x, np.array(trajektoria)

###### SIMPLEX #######
def sortowanie_wierzcholkow(f, simplex):
   wierzcholki = np.array([f(p[0], p[1]) for p in simplex])
   indeksy = np.argsort(wierzcholki)
   return simplex[indeksy], wierzcholki[indeksy]

def krok_simplex(f, simplex, wierzcholki):
   alfa = 1.0
   gamma = 2.0
   beta = 0.5
   sigma = 0.5
   x_best, x_good, x_bad = simplex[0], simplex[1], simplex[2]
   f_best, f_good, f_bad = wierzcholki[0], wierzcholki[1], wierzcholki[2]

   xc = (x_best + x_good) / 2.0
   xr = xc + alfa*(xc - x_bad)
   f_xr = f(xr[0], xr[1])

   if f_xr < f_best:
      xe = xc + gamma*(xr-xc)
      if f(xe[0], xe[1]) < f_xr:
         simplex[2] = xe
      else:
         simplex[2] = xr
   elif f_xr < f_good:
      simplex[2] = xr
   else:
      if f_xr < f_bad:
         xk = xc + beta*(xr - xc)
      else: 
         xk = xc + beta*(x_bad - xc)

      if f(xk[0], xk[1]) < min(f_bad, f_xr):
         simplex[2] = xk
      else:
         simplex[1] = x_best + sigma*(x_good - x_best)
         simplex[2] = x_best + sigma*(x_bad - x_best)
   return simplex

def simplex(f, x0, step_size, tol_x, tol_f, max_iter):
   x0 = np.array(x0, dtype=float)
   simplex = np.array([
        x0,
        x0 + [step_size, 0.0],
        x0 + [0.0, step_size]
        ])
   trajektoria = []

   for i in range(max_iter):
      simplex, wierzcholki = sortowanie_wierzcholkow(f, simplex)
      x_best, x_good, x_bad = simplex[0], simplex[1], simplex[2]
      f_best, f_good, f_bad = wierzcholki[0], wierzcholki[1], wierzcholki[2]
      trajektoria.append(x_best.copy())

      rozmiar_x = max(np.linalg.norm(x_good - x_best), np.linalg.norm(x_bad - x_best))
      roznica_f = abs(f_bad - f_best)
      if rozmiar_x < tol_x or roznica_f < tol_f:
         print (f"simplex => znaleziono ekstremum w {i} krokach")
         break

      simplex = krok_simplex(f, simplex, wierzcholki)
   return simplex[0], np.array(trajektoria)


def f_zastepcza(x1, x2):
   czesc1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
   czesc2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
   return -(czesc1 * czesc2) 

def f_kary(x1, x2, alfa):
  g1 = -0.5 - x1
  g2 = x1 - 0.4
  g3 = -1.3 - x2
  g4 = x2 + 0.5
  suma = (max(0,g1)**2 + max(0,g2)**2 + max(0,g3)**2 + max(0,g4)**2)
  return alfa * suma

def p_ograniczona(x1, x2, alfa):
  return f_zastepcza(x1, x2) + f_kary(x1, x2, alfa)

def wykres_2d(trajektoria, obszar=False):
   x1_zakres = np.linspace(-1, 3, 400)
   x2_zakres = np.linspace(-1.6, 1, 400)
   X1, X2 = np.meshgrid(x1_zakres, x2_zakres)
   Z = f_zastepcza(X1, X2)

   plt.figure(figsize=(8, 6))
   wlasne_poziomy = [
        -1200, -1079, -958, -838, -597, -235, -50, -25, -10, -6
        ]
   poziomice = plt.contour(X1, X2, Z, levels=wlasne_poziomy, cmap='viridis', linewidths=0.7)
   plt.colorbar(poziomice, label='Wartość funkcji f(x)')
   if obszar:
      szerokosc = 0.4 - (-0.5)
      wysokosc = -0.5 - (-1.3)
      prostokat = patches.Rectangle((-0.5, -1.3), szerokosc, wysokosc, 
                                    linewidth=2, edgecolor='red', facecolor='none', 
                                    linestyle='--', label='Obszar dopuszczalny')
      plt.gca().add_patch(prostokat)
   if trajektoria:
        for traj, nazwa, kolor in trajektoria:
            plt.plot(traj[:, 0], traj[:, 1], color=kolor, marker='.', linestyle='-', label=nazwa, alpha=0.8)
            plt.plot(traj[0, 0], traj[0, 1], color=kolor, marker='o')  # Kropka na starcie
            plt.plot(traj[-1, 0], traj[-1, 1], color=kolor, marker='x', markersize=8)  # X na końcu
        plt.legend()
   plt.xlabel('x1')
   plt.ylabel('x2')
   plt.title('dwuwymiarowy wykres funkcji zastępczej')
   plt.grid(True, linestyle='--', alpha=0.3)
   plt.show()

def wykres_3d():
   x1_zakres = np.linspace(-1, 3, 300)
   x2_zakres = np.linspace(-1.6, 1, 300)
   X1, X2 = np.meshgrid(x1_zakres, x2_zakres)
   Z = f_zastepcza(X1, X2)
   Z_przyciete = np.clip(Z, -4000, 0)
   fig = plt.figure(figsize=(8, 6))
   ax = fig.add_subplot(111, projection='3d')
   powierzchnia = ax.plot_surface(X1, X2, Z_przyciete, 
                                  cmap='viridis', 
                                  linewidth=0, 
                                  antialiased=True,
                                  rstride=3, cstride=3)

   ax.set_zlim(-4000, 0)
   ax.view_init(elev=30, azim=120)  
   ax.set_xlabel('x1')
   ax.set_ylabel('x2')
   ax.set_zlabel('f_zastepcza(x1, x2)')
   plt.title('trójwymiarowy wykres funkcji zastępczej')
   fig.colorbar(powierzchnia, ax=ax, label='Wartość funkcji f(x)', shrink=0.7)
   plt.show()


wykres_3d()

#lokalne
step_size = 0.1
tol_x = 1e-3
tol_f = 1e-3
max_iter = 30000
x0 = [0.5, 0.0]

x_gs, trajektoria_gs = g_s(f_zastepcza, x0, step_size, tol_x, tol_f, max_iter)
# print(f"x_gs = {x_gs}")
# print(f"trajektoria_gs = {trajektoria_gs}")
x_simplex, trajektoria_simplex = simplex(f_zastepcza, x0, step_size, tol_x, tol_f, max_iter)
# print(f"x_simplex = {x_simplex}")
# print(f"trajektoria_simplex = {trajektoria_simplex}")
wykres_2d(trajektoria=[(trajektoria_gs, "g_s (lokalne)", "red"),(trajektoria_simplex, "simplex (lokalne)", "blue")])

#globalne
step_size = 0.1
tol_x = 1e-3
tol_f = 1e-3
max_iter = 30
xa = [0.0, -1.0]
x_gs_a, trajektoria_gs_a = g_s(f_zastepcza, xa, step_size, tol_x, tol_f, max_iter)
# print(f"x_gs_a = {x_gs_a}")
# print(f"trajektoria_gs_a = {trajektoria_gs_a}")
x_simplex_a, trajektoria_simplex_a = simplex(f_zastepcza, xa, step_size, tol_x, tol_f, max_iter)
# print(f"x_simplex_a = {x_simplex_a}")
# print(f"trajektoria_simplex_a = {trajektoria_simplex_a}")
wykres_2d(trajektoria=[(trajektoria_gs_a, "g_s (globalne)", "red"),(trajektoria_simplex_a, "simplex (globalne)", "blue")])
xb = [-0.1, -1.1]
x_gs_b, trajektoria_gs_b = g_s(f_zastepcza, xb, step_size, tol_x, tol_f, max_iter)
# print(f"x_gs_b = {x_gs_b}")
# print(f"trajektoria_gs_b = {trajektoria_gs_b}")
x_simplex_b, trajektoria_simplex_b = simplex(f_zastepcza, xb, step_size, tol_x, tol_f, max_iter)
# print(f"x_simplex_b = {x_simplex_b}")
# print(f"trajektoria_simplex_b = {trajektoria_simplex_b}")
wykres_2d(trajektoria=[(trajektoria_gs_b, "g_s (globalne)", "red"),(trajektoria_simplex_b, "simplex (globalne)", "blue")])

#tabela1
def format_pt(x):
   if isinstance(x, str):
      return x
   return f"[{x[0]:.4f}, {x[1]:.4f}]"
dane_nieograniczone = [
   {
      "Punkt startowy": "[0.5, 0.0] (x0)", 
      "Metoda": "g_s", 
      "x* / Komentarz": format_pt(x_gs), 
      "f_Gold(x*)": round(-f_zastepcza(x_gs[0], x_gs[1]), 4)
   },
   {
      "Punkt startowy": "[0.5, 0.0] (x0)", 
      "Metoda": "simplex", 
      "x* / Komentarz": format_pt(x_simplex), 
      "f_Gold(x*)": round(-f_zastepcza(x_simplex[0], x_simplex[1]), 4)
   },
   {
      "Punkt startowy": "[0.0, -1.0] (xA)", 
      "Metoda": "g_s", 
      "x* / Komentarz": "Brak maksimum", 
      "f_Gold(x*)": "-"
   },
   {
      "Punkt startowy": "[0.0, -1.0] (xA)", 
      "Metoda": "simplex", 
      "x* / Komentarz": "Brak maksimum", 
      "f_Gold(x*)": "-"
   },
   {
      "Punkt startowy": "[-0.1, -1.1] (xB)", 
      "Metoda": "g_s", 
      "x* / Komentarz": "Brak maksimum", 
      "f_Gold(x*)": "-"
   },
   {
      "Punkt startowy": "[-0.1, -1.1] (xB)", 
      "Metoda": "simplex", 
      "x* / Komentarz": "Brak maksimum", 
      "f_Gold(x*)": "-"
   }
]
df_nieograniczone = pd.DataFrame(dane_nieograniczone)
print("\nTabela dla podpunktu 2.1 (Obszar nieograniczony)")
print(df_nieograniczone.to_string(index=False))

#ograniczone
step_size = 1e-2
tol_x = 1e-4
tol_f = 1e-4
max_iter = 30000
alfa = 100000
x0 = [0.0, -0.8]
f_ograniczona = lambda x1, x2: p_ograniczona(x1, x2, alfa)
x_gs_ograniczona, trajektoria_gs_ograniczona = g_s(f_ograniczona, x0, step_size, tol_x, tol_f, max_iter)
# print(f"x_gs = {x_gs_ograniczona}")
# print(f"trajektoria_gs = {trajektoria_gs_ograniczona}")
x_simplex_ograniczona, trajektoria_simplex_ograniczona = simplex(f_ograniczona, x0, step_size, tol_x, tol_f, max_iter)
# print(f"x_simplex = {x_simplex_ograniczona}")
# print(f"trajektoria_simplex = {trajektoria_simplex_ograniczona}")
wykres_2d(trajektoria=[(trajektoria_gs_ograniczona, "G-S (lokalne)", "red"),(trajektoria_simplex_ograniczona, "Simplex (lokalne)", "blue")], obszar=True)

#tabela2
def sprawdz_aktywne_ograniczenie(x):
   g1 = -0.5 - x[0]
   g2 = x[0] - 0.4
   g3 = -1.3 - x[1]
   g4 = x[1] + 0.5
   
   ograniczenia = [("g1(x)", g1), ("g2(x)", g2), ("g3(x)", g3), ("g4(x)", g4)]
   aktywne = max(ograniczenia, key=lambda item: item[1])
   if aktywne[1] > -1e-2:
      return aktywne[0], round(aktywne[1], 4)
   return "Brak", "-"
akt_ogr_gs, war_ogr_gs = sprawdz_aktywne_ograniczenie(x_gs_ograniczona)
akt_ogr_simp, war_ogr_simp = sprawdz_aktywne_ograniczenie(x_simplex_ograniczona)
dane_ograniczone = [
   {
      "Punkt startowy": "[0.0, -1.0]", 
      "Algorytm": "g_s", 
      "x*": format_pt(x_gs_ograniczona), 
      "Rodzaj minimum f": "Globalne w obszarze", 
      "f(x*)": round(f_ograniczona(x_gs_ograniczona[0], x_gs_ograniczona[1]), 4), 
      "Aktywne ogr.": akt_ogr_gs, 
      "Wartość g_i(x*)": war_ogr_gs
   },
   {
      "Punkt startowy": "[0.0, -1.0]", 
      "Algorytm": "simplex", 
      "x*": format_pt(x_simplex_ograniczona), 
      "Rodzaj minimum f": "Globalne w obszarze", 
      "f(x*)": round(f_ograniczona(x_simplex_ograniczona[0], x_simplex_ograniczona[1]), 4), 
      "Aktywne ogr.": akt_ogr_simp, 
      "Wartość g_i(x*)": war_ogr_simp
   }
]
df_ograniczone = pd.DataFrame(dane_ograniczone)
print("\nTabela dla podpunktu 2.2 (Obszar ograniczony)")
print(df_ograniczone.to_string(index=False))