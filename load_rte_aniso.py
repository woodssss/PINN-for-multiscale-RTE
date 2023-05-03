import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg

#with open('rg_aniso_hsp_epsi_1pz_gc_zp5.npy', 'rb') as ss:
with open('rg_aniso_hsp_epsi_zpzz1_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
    # np.save(ss, x_f)
    # np.save(ss, rho_real_pred)
    x_f = np.load(ss)
    rho = np.load(ss)
    xx = np.load(ss)
    yy = np.load(ss)
    f = np.load(ss)
    g = np.load(ss)
    #Gamma = np.load(ss)

print(xx.shape, f.shape)


# Now compute reference solution by FD
epsi = 0.001
gc = 0.5

Nx=80

Nv = 40

lx, lv = 1, 1
cc=1

dx = lx/(Nx+1)

x = np.linspace(dx, lx-dx, Nx).T[:, None]





points, weights = np.polynomial.legendre.leggauss(Nv)
points = lv * points
weights = lv * weights
v, w = np.float32(points[:, None]), np.float32(weights[:, None])

Dp, Dm = np.zeros((Nx, Nx)), np.zeros((Nx, Nx))

Vp, Vm = np.zeros((Nv, Nv)), np.zeros((Nv, Nv))

for i in range(Nx):
    Dp[i][i] = 1
    Dm[i][i] = -1

for i in range(Nx-1):
    Dm[i][i+1] = 1
    Dp[i+1][i] = -1

for i in range(int(Nv/2)):
    Vp[i+int(Nv/2)][i+int(Nv/2)] = v[i+int(Nv/2)]
    Vm[i][i] = v[i]

# print('s', Vp, Vm, v)
# asdas

Tp = np.kron(Dp, Vp)
Tm = np.kron(Dm, Vm)

T = (Tp + Tm)*epsi/dx

sk = np.ones((Nv,1))/2
w_mat = np.kron(sk,w.T)

# anisotropic kernel
def L1_val(v, w, weight):
    return (1+v*w) * weight


L1_mat = np.zeros((Nv, Nv))
for i in range(Nv):
    for j in range(Nv):
        L1_mat[i, j] = L1_val(v[i], v[j], w[j])

L2_vec = np.zeros((Nv, 1))
L2_mat = np.zeros((Nv,Nv))

for i in range(Nv):
    #tmp = (1 + gc) / (1 + gc ** 2 - 2 * gc * v[i] * v)
    tmp = 1+v[i]*v
    L2_vec[i] = np.sum(tmp * w)
    L2_mat[i ,i] = np.sum(tmp * w)

#L2_mat = np.diag(L2_vec[0])
# print('ss', L1_mat.shape, L2_mat.shape, L2_vec.shape)
# sdfsdf
sk = np.eye(Nx)
# L1_M = np.kron(sk, L1_mat)
L = sparse.kron(sk, (L1_mat-L2_mat)*0.5)

################################################

BC = np.zeros((Nx*Nv, 1))
ct=0
for i in range(int(Nv/2)):
    BC[i + int(Nv/2)] = epsi*v[i+int(Nv/2)]/dx*5*np.sin(v[i+int(Nv/2)])
    #BC[i + int(Nv / 2)] = epsi * v[i + int(Nv / 2)] / dx

f_ref = scipy.sparse.linalg.spsolve(T-L, BC)

f_ref = f_ref.reshape(Nx, Nv)

rho_ref = np.zeros((Nx,1))
for i in range(Nx):
    tmp = np.sum(f_ref[[i],:]*weights.T)/2
    rho_ref[i] = tmp

print(f_ref[:,-1])



xr, vr = np.meshgrid(x, v)

mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)

plt.rcParams.update({'font.size': 14})

fig = plt.figure(1)
mappable.set_array(f_ref)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xr, vr, f_ref.T, cmap=mappable.cmap, norm=mappable.norm)
plt.title('f(x,v) reference')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('v')



# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, Gamma.T)
# plt.title('Gamma prediction')

fig = plt.figure(3)
mappable.set_array(f)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xx, yy, f.T, cmap=mappable.cmap, norm=mappable.norm)
plt.title('f(x,v) prediction')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('v')

fig = plt.figure(4)
plt.plot(x_f, rho, 'r-o', x, rho_ref, 'b-*')
plt.title(r'$\rho(x)$')
plt.legend(['prediction', 'reference'])
plt.xlabel('x')
plt.show()
