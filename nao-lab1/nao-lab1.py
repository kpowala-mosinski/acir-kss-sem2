import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parametry obliczen
step_size = 0.1
tol_x = 1e-3
tol_f = 1e-3
max_iter = 30000
x0 = np.array([0, 0])

# zakresy dla 1.1 i 1.2
x1_min, x1_max = -1, 3
x2_min, x2_max = -1.6, 1

levels_2d = [6, 14, 34, 64, 104, 154, 554, 854, 1104, 2954]
levels_traj = np.linspace(10, 1010, 10)

# dane dla 1.3
xs = np.array([-0.4, -0.6])
range_x1 = (-0.6, -0.1)
range_x2 = (-0.82, -0.34)


# 1.1 funkcja Goldsteina-Price'a
def f_gold(x1, x2):
    a = 1 + (x1 + x2 + 1) ** 2 * (19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2)
    b = 30 + (2 * x1 - 3 * x2) ** 2 * (18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2)
    return a * b

# funkcja celu dla wektora x=[x1,x2]
def f(x):
    return f_gold(x[0], x[1])

# estymacja gradientu różnicą centralną
def grad_est(fun, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = h
        g[i] = (fun(x + e) - fun(x - e)) / (2 * h)
    return g

# metoda simplex Neldera-Meada
def simplex(fun, x0, step_size=step_size, tol_x=tol_x, tol_f=tol_f, max_iter=max_iter):
    # zwraca (x*, f*, trajektoria)
    alpha, gamma, rho, sigma = 1, 2, 0.5, 0.5

    x0 = np.asarray(x0, dtype=float)
    n = x0.size

    # simplex początkowy = punkt startowy + przesuniecia o step_size wzdłuż osi
    simplex = np.vstack([x0] + [x0 + step_size * np.eye(n)[i] for i in range(n)])

    fvals = np.array([fun(p) for p in simplex])

    traj = [x0.copy()]

    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex, fvals = simplex[order], fvals[order]
        traj.append(simplex[0].copy())

        # warunki STOP: rozmiar simplexu i rozrzut wartości funkcji
        spread_x = max(np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1))
        spread_f = np.max(np.abs(fvals[1:] - fvals[0]))
        if spread_x < tol_x and spread_f < tol_f:
            break

        centroid = simplex[:-1].mean(axis=0)
        worst, f_worst = simplex[-1], fvals[-1]

        # odbicie
        xr = centroid + alpha * (centroid - worst)
        fr = fun(xr)

        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue

        # ekspansja
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = fun(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
            continue

        # kontrakcja
        if fr < f_worst:
            xc = centroid + rho * (xr - centroid)       # kontrakcja zewnętrzna
            fc = fun(xc)
            if fc <= fr:
                simplex[-1], fvals[-1] = xc, fc
                continue
        else:
            xc = centroid + rho * (worst - centroid)    # kontrakcja wewnętrzna
            fc = fun(xc)
            if fc < f_worst:
                simplex[-1], fvals[-1] = xc, fc
                continue

        # redukcja całego simplexu
        simplex[1:] = simplex[0] + sigma * (simplex[1:] - simplex[0])
        fvals[1:] = [fun(p) for p in simplex[1:]]

    best = int(np.argmin(fvals))
    return simplex[best].copy(), float(fvals[best]), np.array(traj)

# metoda gradientow sprzezonych z gradientem estymowanym
def _line_search(fun, x, d, step_size):
    # minimalizacja kierunkowa = ekspansja przedzialu + zloty podzial
    norm_d = np.linalg.norm(d)
    if norm_d == 0:
        return 0

    phi = lambda a: fun(x + a * d)

    # ekspansja - szukanie przedziału [a_lo, a_hi] zawierającego minimum
    a0, f0 = 0, phi(0)
    a1 = step_size / norm_d
    f1 = phi(a1)
    while f1 >= f0 and a1 > 1e-14:          # krok za duży - skróć
        a1 *= 0.5
        f1 = phi(a1)
    if a1 <= 1e-14:
        return 0

    a2 = 2 * a1
    f2 = phi(a2)
    for _ in range(100):                     # krok za mały - wydłuż
        if f2 >= f1:
            break
        a0, f0 = a1, f1
        a1, f1 = a2, f2
        a2 = 2 * a1
        f2 = phi(a2)

    # złoty podział na przedziale [a0, a2]
    gr = (np.sqrt(5) - 1) / 2
    lo, hi = a0, a2
    c, dd = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = phi(c), phi(dd)
    for _ in range(200):
        if hi - lo < 1e-10 * max(1, abs(hi)):
            break
        if fc < fd:
            hi, dd, fd = dd, c, fc
            c = hi - gr * (hi - lo)
            fc = phi(c)
        else:
            lo, c, fc = c, dd, fd
            dd = lo + gr * (hi - lo)
            fd = phi(dd)
    return 0.5 * (lo + hi)

# metoda gradientów sprzężonych
def conjugate_gradient(fun, x0, step_size=step_size, tol_x=tol_x, tol_f=tol_f, max_iter=max_iter):
    # zwraca (x*, f*, trajektoria)
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    fx = fun(x)
    g = grad_est(fun, x)
    d = -g
    traj = [x.copy()]

    for k in range(max_iter):
        if np.linalg.norm(d) == 0:
            break

        a = _line_search(fun, x, d, step_size)
        x_new = x + a * d
        f_new = fun(x_new)
        traj.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol_x and abs(f_new - fx) < tol_f:
            x, fx = x_new, f_new
            break

        g_new = grad_est(fun, x_new)
        gg = float(g @ g)
        beta = 0 if gg == 0 else max(0, float(g_new @ (g_new - g)) / gg)
        if (k + 1) % n == 0:                 # okresowy restart
            beta = 0
        d = -g_new + beta * d
        if float(g_new @ d) > 0:           # zabezpieczenie kierunku spadku
            d = -g_new

        x, fx, g = x_new, f_new, g_new

    return x.copy(), float(fx), np.array(traj)


def sig3(v):
    return float(f"{v:.3g}")

def accuracy(f_star):
    return abs(f_star - round(f_star))


def results_table(rows):
    df = pd.DataFrame(rows, columns=["punkt startowy", "metoda", "x1*", "x2*", "f*", "dokladnosc"])
    for col in ["x1*", "x2*", "f*", "dokladnosc"]:
        df[col] = df[col].map(sig3)
    return df


def run_pair(x0, rows, trajectories):
    label = f"[{x0[0]:.2f}, {x0[1]:.2f}]"
    for name, method in (("simplex N-M", simplex), ("G-S", conjugate_gradient)):
        x_star, f_star, traj = method(f, x0)
        rows.append([label, name, x_star[0], x_star[1], f_star, accuracy(f_star)])
        trajectories.append((label, name, traj, x_star))



# Wykresy
def grid(x1_rng, x2_rng, n=400):
    x1 = np.linspace(x1_rng[0], x1_rng[1], n)
    x2 = np.linspace(x2_rng[0], x2_rng[1], n)
    X1, X2 = np.meshgrid(x1, x2)
    return X1, X2, f_gold(X1, X2)

def plot_contour_2d():
    X1, X2, F = grid((x1_min, x1_max), (x2_min, x2_max))
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(X1, X2, F, levels=levels_2d, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("1.1 funkcja G-P")
    fig.tight_layout()


def plot_surface_3d():
    X1, X2, F = grid((x1_min, x1_max), (x2_min, x2_max), n=200)
    Fc = np.clip(F, None, 4000)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Fc, cmap="viridis", vmin=0, vmax=4000,
                    linewidth=0, antialiased=True)
    ax.set_zlim(0, 4000)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f")
    ax.set_title("1.1 funkcja G-P")
    fig.tight_layout()


def plot_trajectories(trajectories):
    X1, X2, F = grid((x1_min, x1_max), (x2_min, x2_max))
    n_runs = len(trajectories)
    fig, axes = plt.subplots(2, n_runs // 2, figsize=(12, 8))
    for ax, (label, name, traj, x_star) in zip(axes.ravel(), trajectories):
        cs = ax.contour(X1, X2, F, levels=levels_traj, cmap="viridis", linewidths=0.7)
        ax.clabel(cs, inline=True, fontsize=6, fmt="%g")
        ax.plot(traj[:, 0], traj[:, 1], "-o", color="red", markersize=2, linewidth=1,
                label="trajektoria")
        ax.plot(traj[0, 0], traj[0, 1], "s", color="black", markersize=7, label="x0")
        ax.plot(x_star[0], x_star[1], "*", color="blue", markersize=12, label="x*")
        ax.set_xlim(x1_min, x1_max)
        ax.set_ylim(x2_min, x2_max)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"x0 = {label}, {name}", fontsize=9)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("1.2 trajektorie algorytmow")
    fig.tight_layout(rect=(0, 0.04, 1, 1))


def plot_zoom_contour(trajectories):
    X1, X2, F = grid(range_x1, range_x2)
    fig, ax = plt.subplots(figsize=(8, 6))
    cs = ax.contour(X1, X2, F, levels=np.linspace(21, 39, 10), cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8, fmt="%g")
    ax.plot(xs[0], xs[1], "s", color="black", markersize=9, label="xs", zorder=2)
    colors = {"simplex N-M": "red", "G-S": "darkorange"}
    for label, name, traj, x_star in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], "-o", color=colors[name], markersize=3,
                linewidth=1, label=name, zorder=3)
        ax.plot(x_star[0], x_star[1], "*", color=colors[name], markersize=14,
                markeredgecolor="black", markeredgewidth=0.5, zorder=4)
    ax.set_xlim(*range_x1)
    ax.set_ylim(*range_x2)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("1.3 otoczenie xs - trajektorie algorytmow")
    ax.legend(fontsize=8)
    fig.tight_layout()


def plot_zoom_surface():
    X1, X2, F = grid(range_x1, range_x2, n=200)
    Fc = np.clip(F, 18, 43)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X1, X2, Fc, cmap="viridis", vmin=18, vmax=43,
                    linewidth=0, antialiased=True)
    ax.set_zlim(18, 43)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f")
    ax.set_title("1.3 otoczenie xs - wykres powierzchniowy")
    fig.tight_layout()



def main():
    pd.set_option("display.width", 120)
    # 1.1
    plot_contour_2d()
    plot_surface_3d()

    # 1.2
    start_points = [
        np.array([0, -0.6]),    # obszar min globalnego
        np.array([-0.9, -0.2]),   # obszar min przy (-0.6, -0.4)
        np.array([2.4, 0.4]),     # obszar min przy (1.8, 0.2)
        np.array([1, 0.9]),     # obszar min przy (1.2, 0.8)
    ]

    rows, trajectories = [], []
    for x0 in start_points:
        run_pair(x0, rows, trajectories)

    table = results_table(rows)
    print("\n1.2 minima globalne G-S")
    print(table.to_string(index=False))

    best = min(rows, key=lambda r: r[4])
    print(f"\nminimum globalne: x* = [{sig3(best[2])}, {sig3(best[3])}], "
          f"f* = {sig3(best[4])}")

    plot_trajectories(trajectories)

    # 1.3
    rows_xs, traj_xs = [], []
    run_pair(xs, rows_xs, traj_xs)

    table_xs = results_table(rows_xs)
    print("\n1.3 wyniki dla xs = [-0.4, -0.6]")
    print(table_xs.to_string(index=False))

    g_xs = grad_est(f, xs)
    print(f"\ncharakter xs: punkt stacjonarny (||grad f(xs)|| = "f"{sig3(np.linalg.norm(g_xs))}), f(xs) = {sig3(f(xs))} - punkt siodlowy.")

    plot_zoom_contour(traj_xs)
    plot_zoom_surface()

    plt.show()

if __name__ == "__main__":
    main()