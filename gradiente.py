import torch
import matplotlib.pyplot as plt
def f(x):
    return (x[0] - 5)**2 + 2*(x[1] + 3)**2  # mínimo em (5, -3)

def g(lr, p_const, d):
    p_novo = p_const + lr * d
    return f(p_novo)

pontos = []
m_it = 100

def gradiente(p_ini):
    for _ in range(m_it):
      
        val = f(p_ini)
        try:
            print(f"valor de f({p_ini.tolist()}) = {float(val)}")
        except Exception:
            print(f"valor de f = {val}")

        # garante grad zero antes do backward
        if p_ini.grad is not None:
            p_ini.grad.zero_()

        val.backward()

        grad = p_ini.grad
        norm = torch.linalg.norm(grad)
        if norm.item() == 0:
            print("Gradiente nulo — parada")
            break

        # direção de descida (normalizada)
        d = -(grad / norm)

        # busca por lambda (linha unidimensional) via Newton em lr
        lr = torch.tensor(0.01, requires_grad=True)  # lambda inicial (escalar)
        p_const = p_ini.detach()   
        d_const = d.detach()

        for _ in range(10):
            g_val = g(lr, p_const, d_const)
            g_derivada = torch.autograd.grad(g_val, lr, create_graph=True)[0]
            g_derivada_2 = torch.autograd.grad(g_derivada, lr)[0]

            with torch.no_grad():
                lr = lr - (g_derivada / (g_derivada_2 + 1e-8))
            lr.requires_grad_(True)

        lr_optimal = lr.detach()
        # atualiza o ponto com o passo ótimo e re-cria um tensor que requer grad
        p_ini = (p_ini.detach() + (lr_optimal * d_const)).clone().requires_grad_(True)
        pontos.append(p_ini)

    return pontos

if __name__ == "__main__":
    p = torch.tensor([1.0, 1.0], requires_grad=True)
    gradiente(p)




    if len(pontos) > 0:
        xs = [pt.detach().cpu()[0].item() for pt in pontos]
        ys = [pt.detach().cpu()[1].item() for pt in pontos]
        zs = [f(pt.detach()).item() for pt in pontos]

        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        xr = xmax - xmin if xmax > xmin else 1.0
        yr = ymax - ymin if ymax > ymin else 1.0
        pad_x = 0.2 * xr
        pad_y = 0.2 * yr

        xg = torch.linspace(xmin - pad_x, xmax + pad_x, 150)
        yg = torch.linspace(ymin - pad_y, ymax + pad_y, 150)
        X, Y = torch.meshgrid(xg, yg, indexing='xy')
        Z = f(torch.stack([X, Y])).detach()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X.cpu().numpy(), Y.cpu().numpy(), Z.cpu().numpy(),
                        cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)

        ax.plot(xs, ys, zs, color='red', marker='o', markersize=3, linewidth=1.5, label='trajetória')

        minimo = torch.tensor([5.0, -3.0])
        ax.scatter(minimo[0].item(), minimo[1].item(), f(minimo).item(),
                   color='black', s=50, label='mínimo')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x, y)')
        ax.set_title('Convergência do gradiente no espaço 3D')
        ax.legend()
        plt.tight_layout()
        plt.show()

