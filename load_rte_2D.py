import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import scipy as scp

#with open('rg_2D_epsi_1.npy', 'rb') as ss:
#with open('rg_2D_aniso_epsi_zpzz1_h_zp5.npy', 'rb') as ss:
#with open('rg_2D_aniso_epsi_1_h_zp5.npy', 'rb') as ss:
#with open('rg_2D_bln_epsi_1.npy', 'rb') as ss:
with open('rg_2D_bl_epsi_zpzz1.npy', 'rb') as ss:
#with open('rg_2D_any_epsi_1.npy', 'rb') as ss:
#with open('rg_2D_any_epsi_zpzz1.npy', 'rb') as ss:
    x_f = np.load(ss)
    rho_pred = np.load(ss)
    xx = np.load(ss)
    yy = np.load(ss)
    g_pred = np.load(ss)
    f_pred = np.load(ss)
    x = np.load(ss)
    y = np.load(ss)
    xxt = np.load(ss)
    yyt = np.load(ss)
    rho_test = np.load(ss)
    rho_ref = np.load(ss)
    f_test_pred = np.load(ss)
    epoch_vec = np.load(ss)
    emp_loss_vec = np.load(ss)
    test_error_vec = np.load(ss)


# Nx = 100

Nx = 50
Ny = 50

# Nx = 60
# Ny = 60

# # analytic
# f_test_L2 = (np.exp(1)-np.exp(-1))**2*2*np.pi

# # aniso eps=1
# f_test_L2 = 2.08

# # aniso eps=0.001
# f_test_L2 = 1.98

# bl eps=1
f_test_L2 = 24.41


print('L', f_test_L2)

mappable = plt.cm.ScalarMappable(cmap=plt.cm.cividis)

plt.rcParams.update({'font.size': 15})

fig = plt.figure(1)
mappable.set_array(rho_ref)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xxt, yyt, rho_ref, cmap=mappable.cmap, norm=mappable.norm)
plt.title(r'$\rho(x,y)$ reference')
#plt.title(r'$\rho(x,y)$ real')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('y')

fig = plt.figure(2)
mappable.set_array(rho_test)
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xxt, yyt, rho_test.reshape(Nx, Ny).T, cmap=mappable.cmap, norm=mappable.norm)
plt.title(r'$\rho(x,y)$ prediction')
plt.colorbar(mappable)
plt.xlabel('x')
plt.ylabel('y')


plt.figure(5)
plt.semilogy(epoch_vec, emp_loss_vec, 'r-o', epoch_vec, test_error_vec/f_test_L2, 'b-*')
plt.xlabel('iteration')
plt.legend(['empirical loss', r'relative $L^2$ error'])

plt.show()
