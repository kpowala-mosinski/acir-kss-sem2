import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

###### SFORMUŁOWANIE PROBLEMU ######
# ograniczenia nierównościowe w postaci a·x <= b (numeracja 1-5)
OGR_A = np.array([
   [3.0, 1.5],    # 1: 3x1 + 1.5x2 <= 18  (maszyna M1)
   [2.0, 2.0],    # 2: 2x1 + 2x2 <= 20    (maszyna M2)
   [1.0, 0.0],    # 3: x1 <= 5
   [-1.0, 0.0],   # 4: x1 >= 0
   [0.0, -1.0],   # 5: x2 >= 0
])
OGR_B = np.array([18.0, 20.0, 5.0, 0.0, 0.0])

# warianty: z1, z2 oraz ograniczenie równościowe a·x = 0 (numer 6) lub None
WARIANTY = {
   "a": (60, 40, None),
   "b": (100, 40, None),
   "c": (40, 80, None),
   "d": (60, 60, np.array([1.0, -0.5])),   # x1 = 1/2 x2
   "e": (60, 60, np.array([1.0, -5.0])),   # x1 = 5 x2
}

def ograniczenia_g(x):
   return OGR_A @ x - OGR_B

def funkcja_kary(x, h, alfa):
   kara = np.sum(np.maximum(0, ograniczenia_g(x))**2)
   if h is not None:
      kara += np.dot(h, x)**2
   return alfa * kara

###### ROSENBROCK ######
def rosenbrock(f, x0, step_size, tol_x, tol_f, max_iter, alfa=3.0, beta=-0.5):
   x = np.array(x0, dtype=float)
   f_x = f(x)
   n = len(x)
   kierunki = np.eye(n)
   kroki = np.full(n, step_size)
   trajektoria = [x.copy()]
   iteracje = 0

   while iteracje < max_iter:
      x_etap, f_etap = x.copy(), f_x
      przesuniecia = np.zeros(n)
      sukces = np.zeros(n, dtype=bool)
      porazka = np.zeros(n, dtype=bool)

      # etap - próby wzdłuż kierunków aż w każdym kierunku wystąpi sukces i porażka
      while not np.all(sukces & porazka) and iteracje < max_iter:
         iteracje += 1
         for i in range(n):
            x_test = x + kroki[i] * kierunki[i]
            f_test = f(x_test)
            if f_test < f_x:
               x, f_x = x_test, f_test
               przesuniecia[i] += kroki[i]
               kroki[i] *= alfa
               sukces[i] = True
            else:
               kroki[i] *= beta
               porazka[i] = True
         if np.max(np.abs(kroki)) < 1e-12:
            break
      trajektoria.append(x.copy())

      zmiana_x = np.linalg.norm(x - x_etap)
      zmiana_f = abs(f_x - f_etap)
      if zmiana_x < tol_x and zmiana_f < tol_f:
         return x, np.array(trajektoria), True, True
      if np.max(np.abs(kroki)) < 1e-12:
         return x, np.array(trajektoria), zmiana_x < tol_x, zmiana_f < tol_f

      # obrót układu kierunków - ortogonalizacja Grama-Schmidta
      A = np.array([np.sum(przesuniecia[j:, None] * kierunki[j:], axis=0) for j in range(n)])
      if np.linalg.norm(A[0]) > 1e-12:
         nowe = []
         for j in range(n):
            b = A[j] - sum(np.dot(A[j], d) * d for d in nowe)
            if np.linalg.norm(b) < 1e-12:
               break
            nowe.append(b / np.linalg.norm(b))
         if len(nowe) == n:
            kierunki = np.array(nowe)

   return x, np.array(trajektoria), False, False

def optymalizuj(z, h, x0, maksymalizacja, step_size, tol_x, tol_f, max_iter):
   # zewnętrzna funkcja kary z rosnącym współczynnikiem alfa
   znak = -1 if maksymalizacja else 1
   x = np.array(x0, dtype=float)
   trajektoria = [x.copy()]
   for alfa_kary in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]:
      f = lambda x: znak * np.dot(z, x) + funkcja_kary(x, h, alfa_kary)
      x, traj, ok_x, ok_f = rosenbrock(f, x, step_size, tol_x, tol_f, max_iter)
      trajektoria.extend(traj[1:])
   return x, np.array(trajektoria), ok_x, ok_f

###### OBSZAR DOPUSZCZALNY ######
def wierzcholki_obszaru(h):
   proste_A = list(OGR_A)
   proste_B = list(OGR_B)
   if h is not None:
      proste_A.append(h)
      proste_B.append(0.0)

   punkty = []
   for i, j in itertools.combinations(range(len(proste_A)), 2):
      M = np.array([proste_A[i], proste_A[j]])
      if abs(np.linalg.det(M)) < 1e-12:
         continue
      p = np.linalg.solve(M, [proste_B[i], proste_B[j]])
      dopuszczalny = np.all(ograniczenia_g(p) <= 1e-9) and (h is None or abs(np.dot(h, p)) <= 1e-9)
      if dopuszczalny and not any(np.allclose(p, q) for q in punkty):
         punkty.append(p)

   punkty = np.array(punkty)
   srodek = punkty.mean(axis=0)
   katy = np.arctan2(punkty[:, 1] - srodek[1], punkty[:, 0] - srodek[0])
   return punkty[np.argsort(katy)]

def ksztalt_obszaru(wierzcholki):
   nazwy = {1: "punkt", 2: "odcinek", 3: "trójkąt", 4: "czworokąt", 5: "pięciokąt"}
   opis = ", ".join(f"({p[0] + 0.0:.4g}, {p[1] + 0.0:.4g})" for p in wierzcholki)
   return f"{nazwy.get(len(wierzcholki), f'{len(wierzcholki)}-kąt')}: {opis}"

def aktywne_ograniczenia(x, h, tol=1e-3):
   aktywne = [str(i + 1) for i, g in enumerate(ograniczenia_g(x)) if abs(g) < tol]
   if h is not None:
      aktywne.append("6")
   return ", ".join(aktywne) if aktywne else "brak"

###### WYKRES ######
def wykres_2d(nazwa, z, h, wierzcholki, trajektoria, x_max, x_min):
   x1_zakres = np.linspace(-1, 13, 400)
   x2_zakres = np.linspace(-1, 13, 400)
   X1, X2 = np.meshgrid(x1_zakres, x2_zakres)
   Z = z[0] * X1 + z[1] * X2

   plt.figure(figsize=(8, 7))
   poziomice = plt.contour(X1, X2, Z, levels=np.linspace(0, 900, 10), cmap='viridis', linewidths=0.8)
   plt.clabel(poziomice, fontsize=7)
   plt.colorbar(poziomice, label='Z(x) [zł]')

   # proste ograniczeń
   for a, b in zip(OGR_A, OGR_B):
      if a[1] != 0:
         plt.plot(x1_zakres, (b - a[0] * x1_zakres) / a[1], 'k:', linewidth=0.8)
      else:
         plt.axvline(b / a[0], color='k', linestyle=':', linewidth=0.8)
   if h is not None:
      plt.plot(x1_zakres, -h[0] * x1_zakres / h[1], 'm:', linewidth=0.8, label='ograniczenie równościowe')

   # obszar dopuszczalny i jego brzeg
   if len(wierzcholki) > 2:
      obszar = patches.Polygon(wierzcholki, closed=True, facecolor='orange', alpha=0.3,
                               edgecolor='red', linewidth=2, label='Obszar dopuszczalny Φ')
      plt.gca().add_patch(obszar)
   else:
      plt.plot(wierzcholki[:, 0], wierzcholki[:, 1], color='red', linewidth=3, label='Obszar dopuszczalny Φ')
   plt.plot(wierzcholki[:, 0], wierzcholki[:, 1], 'ro', markersize=4)

   plt.plot(trajektoria[:, 0], trajektoria[:, 1], color='blue', marker='.', linestyle='-',
            alpha=0.7, label='Rosenbrock')
   plt.plot(x_max[0], x_max[1], 'b*', markersize=14, label='x_max')
   plt.plot(x_min[0], x_min[1], 'gs', markersize=8, label='x_min')

   plt.xlim(-1, 13)
   plt.ylim(-1, 13)
   plt.gca().set_aspect('equal')
   plt.legend(loc='upper right', fontsize=8)
   plt.xlabel('x1 - produkcja W1 [t/dobę]')
   plt.ylabel('x2 - produkcja W2 [t/dobę]')
   plt.title(f'Wariant {nazwa}): z1 = {z[0]}, z2 = {z[1]}')
   plt.grid(True, linestyle='--', alpha=0.3)
   plt.show()

###### OBLICZENIA ######
step_size = 0.5
tol_x = 1e-6
tol_f = 1e-6
max_iter = 30000
x0 = [1.0, 1.0]

def format_pt(x):
   return f"[{round(x[0], 4) + 0.0:.4g}, {round(x[1], 4) + 0.0:.4g}]"

wyniki = []
for nazwa, (z1, z2, h) in WARIANTY.items():
   z = np.array([z1, z2], dtype=float)
   x_max, trajektoria, ok_x, ok_f = optymalizuj(z, h, x0, True, step_size, tol_x, tol_f, max_iter)
   x_min, _, _, _ = optymalizuj(z, h, x0, False, step_size, tol_x, tol_f, max_iter)
   wierzcholki = wierzcholki_obszaru(h)

   wykres_2d(nazwa, z, h, wierzcholki, trajektoria, x_max, x_min)

   wyniki.append({
      "Wariant": nazwa,
      "x0": format_pt(x0),
      "x_max": format_pt(x_max),
      "Z_max": f"{np.dot(z, x_max):.4g}",
      "tol_x / tol_f": f"{'tak' if ok_x else 'nie'} / {'tak' if ok_f else 'nie'}",
      "Aktywne ogr.": aktywne_ograniczenia(x_max, h),
      "x_min": format_pt(x_min),
      "Z_min": f"{round(np.dot(z, x_min), 2) + 0.0:.4g}",
      "Kształt Φ i wierzchołki": ksztalt_obszaru(wierzcholki),
   })

df = pd.DataFrame(wyniki)
pd.set_option('display.width', 250)
print("\nTabela dla zadania 5 (metoda Rosenbrocka)")
print(df.to_string(index=False))
