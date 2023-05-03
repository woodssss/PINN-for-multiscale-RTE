import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#with open('rg_ori_epsi_1_jw_2_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
# with open('rg_ori_epsi_zpzz1_jw_1_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_ori_adam_epsi_zpzz1_jw_1_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_ori_epsi_zpzz1_jw_1_Nx_150_nl_4_nr_80.npy', 'rb') as ss:
#with open('rg_ori_epsi_1enz5_jw_1_Nx_150_nl_4_nr_80.npy', 'rb') as ss:
#with open('rg_ori_bl_epsi_1_jw_1_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
with open('rg_hsp_epsi_zpzz1_bd_1_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsix_bd_2_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsi1_bd_2_Nx_80_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsixab_bd_1_a_1z_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsi1_bd_1_cs_zpz1_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsi1_bd_1_cs_zp1_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsi1_bd_1_cs_1_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsix_bd_1_cc_5_nl_4_nr_50.npy', 'rb') as ss:
#with open('rg_sigma_epsix_bd_1_cc_8_nl_4_nr_50.npy', 'rb') as ss:
    x_f = np.load(ss)
    rho_pred = np.load(ss)
    x = np.load(ss)
    rho_ref = np.load(ss)
    rho_test = np.load(ss)
    xx = np.load(ss)
    vv = np.load(ss)
    f_pred = np.load(ss)
    g_pred = np.load(ss)
    xxt = np.load(ss)
    vvt = np.load(ss)
    f_ref = np.load(ss)
    f_test_pred = np.load(ss)
    epoch_vec = np.load(ss)
    emp_loss_vec = np.load(ss)
    test_error_vec = np.load(ss)

Nx = 350
Nv= 100

# Nx = 200
# Nv= 80

# Nx = 240
# Nv= 100

f_test = f_test_pred.reshape(Nx, Nv)

dx = 1/Nx

lv=1

points, weights = np.polynomial.legendre.leggauss(Nv)
points = lv * points
weights = lv * weights
v, w = np.float32(points[:, None]), np.float32(weights[:, None])

print(f_ref.shape)
rho_error = np.zeros((Nx, 1))
for i in range(Nx):
    rho_error[i] = np.sum(np.square(f_ref[[i],:]) * w.T)

f_ref_L2 = np.sum(rho_error)*dx




print('L', f_ref_L2)
# ff
#
# f_ref_L2 = 1




mappable = plt.cm.ScalarMappable(cmap=plt.cm.cool)
plt.rcParams.update({'font.size': 14})

fig = plt.figure(1)
mappable.set_array(f_ref)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xxt, vvt, f_ref.T, cmap=mappable.cmap, norm=mappable.norm)
#ax.view_init(30, 120)
plt.title('f(x,v) reference')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('v')


fig = plt.figure(2)
mappable.set_array(f_test.T)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xxt, vvt, f_test.T, cmap=mappable.cmap, norm=mappable.norm)
#ax.plot_surface(xr, vr, f_test_pred.reshape(Nx,Nv).T, cmap=mappable.cmap, norm=mappable.norm)
#ax.view_init(30, 120)
plt.title('f(x,v) prediction')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('v')

# fig = plt.figure(2)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, Gamma.T)
# plt.title('Gamma prediction')


# fig = plt.figure(3)
# mappable.set_array(f_pred)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, vv, f_pred.T, cmap=mappable.cmap, norm=mappable.norm)
# plt.title('f(x,v) prediction')
# plt.colorbar(mappable)
# plt.xlabel('x')
# plt.ylabel('v')

# fig = plt.figure(4)
# plt.plot(x, rho_ref, 'b-*')
# plt.title(r'$\rho(x)$')
# plt.xlabel('x')

# fig = plt.figure(4)
# plt.plot(x, rho_test, 'r-o', x, rho_ref, 'b-*')
# plt.title(r'$\rho(x)$')
# plt.legend(['prediction', 'reference'])
# plt.xlabel('x')

fig = plt.figure(4)
plt.plot(x, rho_test, 'r-o', x, rho_ref, 'b-*', x, 3.1889*(1-x), 'c-^')
plt.title(r'$\rho(x)$')
plt.legend(['prediction', 'reference', 'limit system'])
plt.xlabel('x')

plt.figure(5)
plt.semilogy(epoch_vec, emp_loss_vec/2, 'r-o', epoch_vec, test_error_vec/2/f_ref_L2, 'b-*')
plt.xlabel('iteration')
plt.legend(['empirical loss', r'relative $L^2$ error'])


# c1=5
# c2=8
# c3=10
# plt.figure(6)
# plt.plot(x, 1/(1+np.exp(c1*(x-0.5))), 'r-o', x, 1/(1+np.exp(c2*(x-0.5))), 'b-*', x, 1/(1+np.exp(c3*(x-0.5))), 'c')
# plt.title(r'$\sigma(x)$')
# plt.legend(['c=5', 'c=8', 'c=10'])
# plt.xlabel('x')

a=10
b=20
plt.figure(6)
plt.plot(x, (b+1+np.exp(-a*(x-0.5)))/(1+np.exp(-a*(x-0.5))), 'r-o')
plt.title(r'$\sigma(x)$')
plt.xlabel('x')


plt.show()

# print(epoch_vec[1], emp_loss_vec[0], test_error_vec[0])