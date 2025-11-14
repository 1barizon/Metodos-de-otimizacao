# alunos: 
# Joao Victor Barizon
# Pablo Borges 


import numpy as np 


def f(x):
    return x**2 - 2*x


def secao_aurea(xl, xh, epsilon):
    phi  = 0.618
    while(abs(xh-xl) > epsilon):
        x1 = xh - phi*(xh-xl)
        x2 = xl + phi*(xh-xl)

        if f(x1) > f(x2):
            xl = x1
        else:
            xh = x2
    return (xh+xl)/2


xl = -3
xh = 5
epsilon = 0.000000001
otimo =  secao_aurea(xl, xh, epsilon)
print(f"x otimo: {otimo} -> f(x): {f(otimo)} ")