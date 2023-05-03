import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

#with open('rg_nonl_epsi_1pz.npy', 'rb') as ss:
with open('rg_nonl_hsp_epsi_zpzz1_nl_4_nr_50.npy', 'rb') as ss:
    x_f = np.load(ss)
    rho = np.load(ss)
    xx = np.load(ss)
    yy = np.load(ss)
    T_pred = np.load(ss)
    I_pred = np.load(ss)
    Gamma_I_pred = np.load(ss)
    Gamma_T_pred = np.load(ss)


Nx_f = 60
lx=1
dx = 1 / Nx_f
x_p = np.linspace(0 , lx , Nx_f).T[:, None]


# reference solution
epsi = 1
Nx = 40
Nv = 24

lx, lv = 1, 1
dx = lx/(Nx+1)
x = np.linspace(dx, lx-dx, Nx).T[:, None]

points, weights = np.polynomial.legendre.leggauss(Nv)
points = lv * points
weights = lv * weights
v, w = np.float32(points[:, None]), np.float32(weights[:, None])

N = Nx*(Nv+1)

v_neg, v_pos = v[:int(Nv/2)], v[int(Nv/2):]

TL, TR = 1.0, 0.0

phiL = np.ones((int(Nv/2), 1))
for i in range(int(Nv/2)):
    phiL[i] = 5*np.sin(v[i+int(Nv/2)])
    #phiL[i] = 1


phiR = np.zeros((int(Nv/2), 1))



def nlf(x, epsi,dx,Nx,Nv,phiL,phiR,TL,TR,v,w):
    F = np.empty(Nx*(Nv+1))
    N = Nx*Nv
    # x = dx
    for i in range(int(Nv/2)):
        F[i] = epsi*v[i]*(x[Nv+i]- x[i])/dx + x[i] -x[Nx*Nv]**4

    for i in range(int(Nv/2), Nv):
        F[i] = epsi*v[i]*(x[i] - phiL[i-int(Nv/2)])/dx + x[i] -x[Nx*Nv]**4

    # x = 2dx:dx:1-2dx
    for l in range(2, Nx):
        for m in range(int(Nv/2)):
            F[(l-1)*Nv+m] = epsi*v[m]*(x[(l)*Nv + m] - x[(l-1)*Nv+m])/dx + x[(l-1)*Nv+m] - x[Nx*Nv+(l-1)]**4

        for m in range(int(Nv/2), Nv):
            F[(l-1)*Nv+m] = epsi*v[m]*(x[(l-1)*Nv + m] - x[(l-2)*Nv+m])/dx + x[(l-1)*Nv+m] - x[Nx*Nv+(l-1)]**4

    # x = 1-dx
    for i in range(int(Nv/2)):
        F[(Nx-1)*Nv+i] = epsi*v[i]*(phiR[i]- x[(Nx-1)*Nv+i])/dx + x[(Nx-1)*Nv+i] -x[Nx*(Nv+1)-1]**4

    for i in range(int(Nv/2), Nv):
        F[(Nx-1)*Nv+i] = epsi*v[i]*(x[(Nx-1)*Nv+i]- x[(Nx-2)*Nv+i])/dx + x[(Nx-1)*Nv+i] -x[Nx*(Nv+1)-1]**4

    # second pde
    # x=dx
    # tmp = x[:Nv][:,None]
    # print('ss', w.shape, x[:Nv].shape, tmp.shape )
    # adfaf
    F[N] = epsi**2 * (x[N+1]-2*x[N] + TL)/dx**2 - x[N]**4 + np.sum(w*x[:Nv][:,None])/2

    for i in range(1, Nx-1):
        F[N+i] = epsi**2 * (x[N+i+1]-2*x[N+i]+x[N+i-1])/dx**2 - x[N+i]**4 + np.sum(w*x[(i)*Nv:(i+1)*Nv][:,None])/2

    F[N+Nx-1] = epsi**2 * (TR - 2*x[N+Nx-1]+x[N+Nx-2])/dx**2 - x[N+Nx-1]**4 + np.sum(w*x[(Nx-1)*Nv:Nx*Nv][:,None])/2

    return F

x0 = np.zeros((N, 1))[:,None]
#x0 = -np.ones((N, 1))
#x0 = np.random.rand(N,1)

res = fsolve(nlf, x0, args=(epsi,dx,Nx,Nv,phiL,phiR,TL,TR,v,w), xtol=1e-5)
I = res[:Nx*Nv]
T_ref = res[Nx*Nv:]

I_ref = I.reshape(Nx, Nv)

rho_ref = np.zeros((Nx, 1))
for i in range(Nx):
    rho_ref[i] = np.sum(w*I_ref[i, :][:,None])/2

xxr, vvr = np.meshgrid(x,v)

mappable = plt.cm.ScalarMappable(cmap=plt.cm.inferno)
plt.rcParams.update({'font.size': 15})

fig = plt.figure(1)
mappable.set_array(I_pred)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, I_pred.T, cmap=mappable.cmap, norm=mappable.norm)
plt.title(r'$I(x,v)$ prediction')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('v')


fig = plt.figure(2)
mappable.set_array(I_ref)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xxr, vvr, I_ref.T, cmap=mappable.cmap, norm=mappable.norm)
plt.title(r'$I(x,v)$ reference')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('v')


fig = plt.figure(3)
plt.plot(x, rho_ref, 'r-o', x_f, rho, 'b-*')
plt.xlabel('x')
plt.title(r'$\rho(x)$')
plt.legend(['reference', 'prediction'])

fig = plt.figure(4)
plt.plot(x, T_ref, 'r-o', x_f, T_pred, 'b-*', x_f, Gamma_T_pred, 'c')
plt.xlabel('x')
plt.title(r'$T(x)$')
plt.legend(['reference', 'prediction'])



plt.show()








