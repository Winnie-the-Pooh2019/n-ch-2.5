import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Функция f(x, t) - источник тепла
def f(x, t):
    return 0.0  # Можно изменить при необходимости

# Явная схема
def solve_heat_equation(Nx_values):
    L = 1.0
    T = 1.0
    alpha = 4.0
    results = []

    for Nx in Nx_values:
        h = L / (Nx - 1)
        tau = min(h**2 / 9, h / 3)  # Условие устойчивости
        Nt = int(T / tau) + 1

        x = np.linspace(0, L, Nx)
        t = np.linspace(0, T, Nt)

        print(f"Nx = {Nx}, h = {h:.4f}, tau = {tau:.6f}")

        U = np.zeros((Nt, Nx))

        # Начальное условие
        U[0, :] = (1.3 * x**2 + 1.2) * np.sin(np.pi * x)

        # Граничные условия через зеркальное продолжение
        for k in range(Nt - 1):
            for i in range(Nx):
                if i == 0:
                    up = U[k, i + 1]
                    um = U[k, i + 1]  # dU/dx = 0 => U[i-1] = U[i+1]
                elif i == Nx - 1:
                    up = U[k, i - 1]
                    um = U[k, i - 1]
                else:
                    up = U[k, i + 1]
                    um = U[k, i - 1]

                U[k+1, i] = U[k, i] + tau * (
                    4 * (up - 2*U[k, i] + um) / h**2 +
                    (up - um) / (2*h) +
                    f(x[i], t[k])
                )

        results.append((x, t, U))

    return results

# Рисование графиков
def plot_results(results):
    for idx, (x, t, U) in enumerate(results):
        plt.figure(figsize=(10, 6))
        plt.imshow(U, aspect='auto', extent=[0, 1, 0, 1], origin='lower', cmap='hot')
        plt.colorbar(label='Temperature U(x, t)')
        plt.title(f'Temperature Distribution (Nx = {len(x)})')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.grid(True)
        plt.show()

        X, T_mesh = np.meshgrid(t, x)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X.T, T_mesh.T, U, cmap='viridis')
        ax.set_xlabel('Time t')
        ax.set_ylabel('Position x')
        ax.set_zlabel('Temperature U(x, t)')
        ax.set_title(f'3D View (Nx = {len(x)})')
        plt.show()

# Основная часть
Nx_values = [10, 50, 100]
results = solve_heat_equation(Nx_values)
plot_results(results)