import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys
import scipy.special as sc
from tensorflow.keras.models import load_model



class bl_util():
    # this is for linear transport equation epsi * \partial_t f + v \partial_x f = 1/epsi * L(f)
    # L(f) = 1/2 \int_-1^1 f  - f
    # the expect limit system is \partial_t rho - 1/3 \partial_xx rho = 0
    def __init__(self, model, x_f, Tset, Nx_f, Nv_f):
        self.model = model
        self.Tset = tf.convert_to_tensor(Tset, dtype=tf.float32)
        self.x_f = x_f
        self.Nx_f, self.Nv_f = Nx_f, Nv_f

    def predict(self):
        F_T, F_I = self.model([self.x_f, self.Tset])
        return F_T, F_I


    def get_rho_vec(self, rho):

        sp=tf.ones((self.Nv_f,1))

        rho_vec = self.Kron_TF(rho, sp)

        rho_vec = tf.reshape(rho_vec, [self.Nx_f * self.Nv_f, 1])

        return rho_vec

    def Kron_TF(self, A, B):
        A_shape = A.get_shape()
        B_shape = B.get_shape()

        for i in range(A_shape[0]):
            for j in range(A_shape[1]):
                if j == 0:
                    temp = tf.squeeze(A[i, j] * B)

                else:
                    temp = tf.concat([temp, tf.squeeze(A[i, j] * B)], 1)
            if i == 0:
                result = temp
            else:
                result = tf.concat([result, temp], 0)
        return result









if __name__ == "__main__":
    # input parameters
    J3_weight = 10
    Nx_f = 600
    nlr = 3
    nlg = 3
    nur = 30
    nug = 30

    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv, T = 10, 1, 1

    # define training set
    # [x_f, v_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f= 80



    x_f = np.linspace(0 , lx , Nx_f).T[:, None]

    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points = lv * points
    weights = lv * weights
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])

    Tset = np.ones((Nx_f * Nv_f, 2))  # Training set, the first column is x and the second column is v

    for i in range(Nx_f):
        Tset[i * Nv_f:(i + 1) * Nv_f, [0]] = x_f[i][0] * np.ones_like(v_f)
        Tset[i * Nv_f:(i + 1) * Nv_f, [1]] = v_f

    # [x_0, v_0] for IC
    x_0, v_0 = x_f, v_f
    rho_0 = 1+0.5*np.sin(2*np.pi*x_0)
    temp_M0 = np.exp(-15*(v_0-0.5)**2) + np.exp(-15*(v_0+0.5)**2)
    #temp_M0 = np.exp(-15 * (v_0) ** 2)
    f_0_vec=np.zeros((Nx_f*Nv_f,1))
    for i in range(Nx_f*Nv_f):
        f_0_vec[i] = rho_0[math.floor(i/Nv_f)]*temp_M0[i%Nv_f]


    # For BC, there are two BCs, for v>0 and for v<0
    nbc=80
    v_bc_pos, v_bc_neg = np.random.rand(1, nbc).T, -np.random.rand(1, nbc).T
    #v_bc_pos, v_bc_neg = np.linspace(0, lv, nbc).T[:, None], np.linspace(-lv, 0, nbc).T[:, None]
    x_bc_pos, x_bc_zeros = np.ones((nbc,1)), np.zeros((nbc,1))

    Train_BC_L = np.float32(np.concatenate((x_bc_zeros, v_bc_pos), axis=1))
    Train_BC_R = np.float32(np.concatenate((x_bc_pos, v_bc_neg), axis=1))

    fL_train = np.float32(5*np.sin(v_bc_pos))
    fR_train = 0

    # plt.plot(v_bc_pos, fL_train, 'ro')
    # plt.show()

    #print('bc', Train_BC_L, Train_BC_R)

    # define parameter for model
    dt=0.1
    f0=f_0_vec
    dtype = tf.float32
    num_ad_epochs = 10001
    num_bfgs_epochs = 8000
    # define adam optimizer
    train_steps = 5
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    optimizer = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)



    # save model
    def num2str_deciaml(x):
        s=str(x)
        c=''
        for i in range(len(s)):
            if s[i]=='0':
                c = c + 'z'
            elif s[i]=='.':
                c = c + 'p'
            elif s[i]=='-':
                c = c + 'n'
            else:
                c = c + s[i]

        return c


    epsi = 1



    #model_name = 'NLNTE_hsp_epsi_' + num2str_deciaml(epsi) + '_jw_' + str(J3_weight) + '_Nx_' + str(Nx_f) + '_nlr_' + str(nlr) + '_nlg_' + str(nlg)+ '_nur_' + str(nur)+ '_nug_' + str(nug) + '.' +  'h5'
    #model_name = 'NLNTE_hsp_epsi_1_TL_zp5.h5'
    model_name = 'NLRTE_hsp_epsi_1_jw_2_Nx_800_nlr_4_nlg_4_nur_50_nug_50.h5'
    model = load_model(model_name)
    read = bl_util(model, x_f, Tset, Nx_f, Nv_f)

    F_T, F_I = read.predict()

    F_I_pred = F_I.numpy().reshape(Nx_f,Nv_f)

    C_T_pred, C_I_pred = model([np.ones((1,1))*10, np.array([[10, 0]])])

    print('limit pred', C_T_pred, C_I_pred)

    #test_f0 = f_0_vec.reshape(Nx_f,Nv_f)

    xx,vv=np.meshgrid(x_f,v_f)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx, vv, F_I_pred.T)
    plt.title('f')

    plt.figure(2)
    plt.plot(x_f, F_T, 'r-o')
    plt.show()









