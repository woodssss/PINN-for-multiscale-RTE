import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import tensorflow as tf

with open('rg_hsp_epsi_zpzz1_nl_4_nr_50.npy', 'rb') as ss:
    # np.save(ss, x_f)
    # np.save(ss, rho_real_pred)
    x_f = np.load(ss)
    rho = np.load(ss)
    xx = np.load(ss)
    yy = np.load(ss)
    f = np.load(ss)
    g = np.load(ss)

print(xx.shape, f.shape)


# Now compute reference solution by FD

CH = 3.1889

epsi = 0.001

Nx1 = 150
Nx2 = 200

Nx=Nx1+Nx2

Nv = 40

lx, lv = 1, 1
cc=1

dx1 = cc*epsi/Nx1
dx2 = (lx-cc*epsi)/Nx2

x1 = np.linspace(0, cc*epsi, Nx1).T[:, None]

x2 = np.linspace(cc*epsi+dx2, lx, Nx2).T[:, None]

x = np.concatenate([x1, x2], axis=0)



points, weights = np.polynomial.legendre.leggauss(Nv)
points = lv * points
weights = lv * weights
v, w = np.float32(points[:, None]), np.float32(weights[:, None])

Dp, Dm = np.zeros((Nx, Nx)), np.zeros((Nx, Nx))

Vp, Vm = np.zeros((Nv, Nv)), np.zeros((Nv, Nv))

for i in range(Nx1):
    Dp[i][i] = 1/dx1

for i in range(1, Nx1):
    Dp[i][i-1] = -1/dx1

for i in range(Nx1, Nx):
    Dp[i][i] = 1/dx2
    Dp[i][i-1] = -1/dx2


for i in range(Nx1):
    Dm[i][i] = - 1 / dx1
    Dm[i][i+1] = 1 / dx1

for i in range(Nx1, Nx):
    Dm[i][i] = -1 / dx2

for i in range(Nx1, Nx-1):
    Dm[i][i+1] = 1 / dx2


for i in range(int(Nv/2)):
    Vp[i+int(Nv/2)][i+int(Nv/2)] = v[i+int(Nv/2)]*epsi
    Vm[i][i] = v[i]*epsi

# print('s', Vp, Vm, v)
# asdas

Tp = sparse.kron(Dp, Vp)
Tm = sparse.kron(Dm, Vm)

T = (Tp + Tm)

sk = np.ones((Nv,1))/2
w_mat = np.kron(sk,w.T)

L = sparse.kron(np.eye(Nx), w_mat) - sparse.eye(Nx*Nv)

BC = np.zeros((Nx*Nv, 1))
ct=0
for i in range(int(Nv/2)):
    BC[i + int(Nv/2)] = epsi*v[i+int(Nv/2)]/dx1*5*np.sin(v[i+int(Nv/2)])
    #BC[i + int(Nv / 2)] = epsi * v[i + int(Nv / 2)] / dx1

f_ref_vec = scipy.sparse.linalg.spsolve(T-L, BC)

f_ref = f_ref_vec.reshape(Nx, Nv)

rho_ref = np.zeros((Nx,1))
for i in range(Nx):
    tmp = np.sum(f_ref[[i],:]*weights.T)/2
    rho_ref[i] = tmp

print(f_ref[:,-1])










Test_x = np.kron(x, np.ones((Nv,1)))
Test_v = np.kron(np.ones((Nx,1)), v)

Test = np.concatenate([Test_x, Test_v], axis=1)

mdl_name = 'rg_hsp_epsi_zpzz1_jw_1_Nx_60_nl_4_nr_50.h5'
mdl = tf.keras.models.load_model(mdl_name)

rho_test, g_test = mdl([x, Test])

rho_test, g_test = rho_test.numpy(), g_test.numpy()

f_test_pred = np.kron(rho_test, np.ones((Nv,1))) + epsi*g_test




G_file_name = 'half_space_iso_with_lx10.h5'


def my_act(x):
    return tf.nn.sigmoid(x) * np.max(5 * np.sin(1))


G_mdl = tf.keras.models.load_model(G_file_name, custom_objects={"my_act": my_act})
# G_mdl = load_model(G_file_name)

CH_pred = G_mdl(np.array([[10, 0]]))

x_G = np.concatenate([x1, x2[0][:,None]], axis=0)/epsi


sp = np.ones((Nv, 1))
sk = np.ones((Nx1+1, 1))

x_G_vec = np.kron(x_G, sp)
v_G_vec = np.kron(sk, v)

test_Gamma = np.concatenate([x_G_vec, v_G_vec], axis=1)

Gamma_test_pred_vec_1 = G_mdl(test_Gamma).numpy()

Gamma_test_pred_vec_2 = CH_pred*np.ones((Nx*Nv-(Nx1+1)*Nv, 1))

Gamma_test_pred_vec = np.concatenate([Gamma_test_pred_vec_1, Gamma_test_pred_vec_2], axis=0) - CH_pred*np.ones((Nx*Nv, 1))

Gamma_test_pred_vec = Gamma_test_pred_vec.numpy()

f_test_pred_real = f_test_pred + Gamma_test_pred_vec

print('sss', f_test_pred.shape, f_test_pred_real.shape, Gamma_test_pred_vec.shape)

plt.plot(v, Gamma_test_pred_vec[:Nv], 'r', v, f_test_pred[:Nv], 'b', v, f_test_pred_real[:Nv], 'c')
plt.show()




xr, vr = np.meshgrid(x, v)

mappable = plt.cm.ScalarMappable(cmap=plt.cm.cool)
plt.rcParams.update({'font.size': 14})

fig = plt.figure(1)
mappable.set_array(f_ref)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xr, vr, f_ref.T, cmap=mappable.cmap, norm=mappable.norm)
plt.title('f(x,v) reference')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('v')

f_test = f_test_pred_real.reshape(Nx, Nv)

fig = plt.figure(2)
mappable.set_array(f_test.T)
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xr, vr, f_ref.T-f_test.T, cmap=mappable.cmap, norm=mappable.norm)
ax.plot_surface(xr, vr, f_test_pred_real.reshape(Nx,Nv).T, cmap=mappable.cmap, norm=mappable.norm)
plt.title('f(x,v) test diff')
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
plt.plot(x_f, rho, 'r-o', x, rho_ref, 'b-*', x, CH*(1-x), 'c-^')
plt.title(r'$\rho(x)$')
plt.legend(['prediction', 'reference', 'limit system'])
plt.xlabel('x')


f_test_real = f_test_pred_real.reshape(Nx,Nv)

test_error1 = 0
er = np.square(f_ref-f_test_real)
for i in range(Nx1):
    tmp = np.sum(er[[i], :] * weights.T) / 2
    test_error1 = test_error1 + tmp

test_error1 = np.sum(test_error1)*dx1

test_error2 = 0
er = np.square(f_ref-f_test_real)
for i in range(Nx1, Nx):
    tmp = np.sum(er[[i], :] * weights.T) / 2
    test_error2 = test_error2 + tmp

test_error2 = np.sum(test_error1)*dx2

test_error = test_error1+test_error2

print('er', test_error1, test_error)


rho_test = np.zeros((Nx,1))
for i in range(Nx):
    tmp = np.sum(f_test_real[[i],:]*weights.T)/2
    rho_test[i] = tmp

plt.figure(5)
plt.plot(x, rho_test, 'r-o', x, rho_ref, 'b-*', x, CH*(1-x), 'c-^')
plt.show()
