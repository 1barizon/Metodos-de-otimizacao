import numpy as np
import matplotlib.pyplot as plt

h = 0.0000001

def f(x):
    return (x-1)**2  -10
def df(x):
    valor = (f(x+h)-f(x))/h
    return valor

def ddf(x):
    valor = (f(x+h) - 2*f(x) + f(x-h))/(h**2)
    return valor


def newton(x0, niters, epsilon):
    x1 = x0 - df(x0)/ddf(x0)
    i = 0
    while(np.abs(x1-x0) > epsilon or i < niters):
        x0  = x1
        x1 = x0 - df(x0)/ddf(x0)
        i += 1
    return float((x0 + x1) / 2)

otimo = newton(1, 30 , 0.001)

print(f"o valor max/min de f(x) = {f(otimo)} e em x = {otimo:2f}")


xs = np.linspace(-10,10,)
ys = f(xs)

plt.plot(xs, ys)
plt.plot(otimo, f(otimo), marker="o")




plt.title("funcao otimizada pelo metodo de newton")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True) 
plt.ylim(-50, 100) 