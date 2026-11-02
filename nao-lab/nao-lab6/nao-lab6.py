import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def K1(P):
    return 4 * (P / 100)**2 + 45
def K2(P):
    return -4 * ((P - 400) / 100)**2 + 90
def K3(P):
    return 15 * (P / 100) + 40

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



print("=== PODPUNKT 6.1 ===")
wizualizacja_kosztow()
