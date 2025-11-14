import numpy as np
from mpl_toolkits.mplot3d import Axes3D


h = 1e-6

def f(x1, x2):
    return (x1 - 2) ** 4 + (x1 - 2 * x2) ** 2


def df(x1, x2, i):
    if i == 1:
        return (f(x1 + h, x2) - f(x1 - h, x2)) / (2 * h)
    elif i == 2:
        return (f(x1, x2 + h) - f(x1, x2 - h)) / (2 * h)


def ddf(x1, x2, i):
    if i == 1:
        return (f(x1 + h, x2) - 2 * f(x1, x2) + f(x1 - h, x2)) / (h ** 2)
    elif i == 2:
        return (f(x1, x2 + h) - 2 * f(x1, x2) + f(x1, x2 - h)) / (h ** 2)
  


def newton_coord(y, i, niters=50, epsilon=1e-8):

    yk = np.array(y, dtype=float)
    idx = i - 1
    for _ in range(niters):
        g = df(yk[0], yk[1], i)
        H = ddf(yk[0], yk[1], i)
        if abs(H) < 1e-14:
            # Evita divisão por zero ou passos explosivos
            break
        step = g / H
        yk[idx] -= step
        if abs(step) < epsilon or abs(g) < epsilon:
            break
    return yk


def hook_jeeves(y0, niters):
    """Demonstra uma busca por coordenadas usando passos de Newton em cada eixo."""
    y = np.array(y0, dtype=float)
    for _ in range(niters):
        y_old = y.copy()
        # Otimiza x1 com x2 fixo
        y = newton_coord(y, i=1, niters=50, epsilon=1e-8)
        y1 = y
        # Otimiza x2 com x1 fixo
        y = newton_coord(y, i=2, niters=50, epsilon=1e-8)
        y2 = y

        # Busca padrão: direção entre y1 e y2, busca unidimensional ao longo dela
        d = y2 - y1
        if np.linalg.norm(d) > 1e-12:
            t = 1.0
            def phi(tt):
                return f(y2[0] + tt * d[0], y2[1] + tt * d[1])

            for _ in range(50):
                g = (phi(t + h) - phi(t - h)) / (2 * h)
                H = (phi(t + h) - 2 * phi(t) + phi(t - h)) / (h ** 2)
                if abs(H) < 1e-14:
                    break
                step = g / H
                t -= step
                if abs(step) < 1e-8 or abs(g) < 1e-8:
                    break
            y = y2 + t * d
        else:
            y = y2
       

        if np.linalg.norm(y - y_old, ord=np.inf) < 1e-8:
            break
    return y



y1 = np.array([0.0, 3.0])
y_opt = hook_jeeves(y1, 20)
print("y* =", y_opt, "f(y*) =", f(y_opt[0], y_opt[1]))




import matplotlib.pyplot as plt

span = 3.0
n = 200
x = np.linspace(y_opt[0] - span, y_opt[0] + span, n)
y = np.linspace(y_opt[1] - span, y_opt[1] + span, n)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
ax.scatter([y_opt[0]], [y_opt[1]], [f(y_opt[0], y_opt[1])], color='r', s=80, label='otimo')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1,x2)')
ax.set_title('Superfície de f e ponto ótimo')
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.view_init(elev=30, azim=-60)
plt.legend()
plt.show()