import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys
from scipy.optimize import fsolve
import scipy.special as sc


class stdst():
    def __init__(self, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Bd_weight, Tset, Train_BC_L, Train_BC_R, fL_train, TL, nbc, weights, dx,
                 dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl, nr, Nx, Nv, xt, vt, dxt, wt, Test, I_ref, I_ref_vec, T_ref):
        self.dtype = dtype
        self.epsi, self.Nx_f, self.Nv_f = epsi, Nx_f, Nv_f
        self.lx, self.lv, self.dx = lx, lv, dx
        self.Bd_weight = Bd_weight
        self.xx, self.vv = np.meshgrid(x_f, v_f)
        self.nbc = nbc

        # number of layers for rho and g
        self.nl, self.nr = nl, nr

        self.stop = 0.005

        self.file_name = file_name
        # convert np array to tensor
        self.x_f, self.v_f = tf.convert_to_tensor(x_f, dtype=self.dtype), tf.convert_to_tensor(v_f,
                                                                                               dtype=self.dtype)  # x_f and v_f are grid on x and v direction
        self.x_train, self.v_train = tf.convert_to_tensor(Tset[:, [0]], dtype=self.dtype), tf.convert_to_tensor(
            Tset[:, [1]], dtype=self.dtype)  # x_train and v_train are input trainning set for NN

        self.Tset = tf.convert_to_tensor(Tset, dtype=self.dtype)

        self.weights = tf.convert_to_tensor(weights, dtype=self.dtype)

        self.dxt, self.Nx, self.Nv = dxt, Nx, Nv
        self.wt = tf.convert_to_tensor(wt, dtype=self.dtype)

        self.wt_ori = wt

        self.xt, self.vt = tf.convert_to_tensor(xt, dtype=self.dtype), tf.convert_to_tensor(vt, dtype=self.dtype)
        self.Test = tf.convert_to_tensor(Test, dtype=self.dtype)

        self.I_ref = I_ref
        self.I_ref_vec = tf.convert_to_tensor(I_ref_vec, dtype=self.dtype)

        self.T_ref = T_ref

        self.xxt, self.vvt = np.meshgrid(xt, vt)

        # track loss
        self.epoch_vec = []
        self.loss_vec = []

        self.emp_loss_vec = []
        self.test_error_T_vec = []
        self.test_error_I_vec = []



        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        # define BC
        self.Train_BC_L = tf.convert_to_tensor(Train_BC_L, dtype=self.dtype)
        self.Train_BC_R = tf.convert_to_tensor(Train_BC_R, dtype=self.dtype)

        self.fL_train = tf.convert_to_tensor(fL_train, dtype=self.dtype)
        self.TL = tf.convert_to_tensor(TL, dtype=self.dtype)


        # test
        # self.limit_test()

        # Initialize NN
        self.nn = self.get_nn()
        self.nn.summary()

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []

        for layer in self.nn.layers[2:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            # print('ssssss', int(weights.shape[0]), int(biases.shape[0]))
            self.sizes_w.append(int(weights.shape[0]))
            self.sizes_b.append(int(biases.shape[0]))

    def get_nn(self):

        # define nn for rho
        input_rho = tf.keras.Input(shape=(1,))

        # for i in range(5):
        #     input_rho = tf.keras.layers.Dense(units=100, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
        #     input_rho)

        input_rho_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                              kernel_initializer='glorot_normal')(
            input_rho)

        for i in range(self.nl-1):
            input_rho_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                  kernel_initializer='glorot_normal')(
                input_rho_mid)

        output_rho = tf.keras.layers.Dense(units=2, activation=None, kernel_initializer='glorot_normal')(
            input_rho_mid)

        # define nn for g

        input_g = tf.keras.Input(shape=(2,))

        input_g_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
            input_g)

        for i in range(self.nl-1):
            input_g_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                kernel_initializer='glorot_normal')(
                input_g_mid)


        output_g = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer='glorot_normal')(
            input_g_mid)

        model = tf.keras.Model(inputs=[input_rho, input_g], outputs=[output_rho, output_g])

        return model

    def get_intv(self, f):

        sp = tf.ones([self.Nv_f, 1])

        sk = tf.linspace(np.float32(0), self.Nx_f - 1, self.Nx_f, name="linspace")

        sk = tf.reshape(sk, (self.Nx_f, 1))

        id = self.Kron_TF(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.Nx_f * self.Nv_f])

        dup_p = tf.constant([self.Nx_f, 1], tf.int32)

        weights_rep = tf.tile(self.weights, dup_p)

        res = tf.math.segment_sum(weights_rep * f, id)

        # print('ssp', weights_rep.shape, res.shape)

        return res

    def get_pde1(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho_T, g = self.nn([self.x_f, Train])

            g_x = tape.gradient(g, self.x_train)

            rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

            T_x = tape.gradient(T, self.x_f)

        T_xx = tape.gradient(T_x, self.x_f)

        int_vg = self.get_intv(self.v_train * g_x)

        pde1 = int_vg * 0.5 - T_xx

        # print('sp', pde1.shape, int_vg.shape, T_xx.shape)
        #
        # sdfgdfg

        return pde1

    def get_pde2(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho_T, g = self.nn([self.x_f, Train])

            g_x = tape.gradient(g, self.x_train)

            rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

            rho_x = tape.gradient(rho, self.x_f)

            T_x = tape.gradient(T,self.x_f)

        T_xx = tape.gradient(T_x, self.x_f)

        T_xx_vec =self.get_rho_vec(T_xx)

        rho_x_vec = self.get_rho_vec(rho_x)

        int_vg = self.get_intv(self.v_train * g_x)

        int_vg_vec = self.get_rho_vec(int_vg) * 0.5

        pde2 = self.v_train * rho_x_vec + self.epsi * self.v_train * g_x - self.epsi * T_xx_vec + g

        # print('sp', pde2.shape, rho_x_vec.shape, int_vg_vec.shape)

        return pde2

    def get_pde3(self):

        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho_T, g = self.nn([self.x_f, Train])

            rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

            T_x = tape.gradient(T, self.x_f)

        T_xx = tape.gradient(T_x, self.x_f)

        pde3 = self.epsi ** 2 * T_xx + (
                    rho - tf.pow(T, 4))

        return pde3

    def get_pde32(self):

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(self.x_f)
            with tf.GradientTape(persistent=True) as tape1:
                # Watching the two inputs we’ll need later, x and t
                tape1.watch(self.x_f)
                # Packing together the inputs
                Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

                # Getting the prediction
                rho_T, g = self.nn([self.x_f, Train])

                rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

                T_x = tape1.gradient(T, self.x_f)

                T4 = tf.pow(T,4)

                T4_x = tape1.gradient(T4, self.x_f)

                rho_x = tape1.gradient(rho, self.x_f)

            T_xx = tape2.gradient(T_x, self.x_f)

        T_xxx = tape2.gradient(T_xx, self.x_f)

        pde32 = self.epsi ** 2 * T_xxx + (rho_x - T4_x)
        #pde32 =  (rho_x - T4_x)

        return pde32

    def get_pde33(self):

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(self.x_f)
            with tf.GradientTape(persistent=True) as tape1:
                # Watching the two inputs we’ll need later, x and t
                tape1.watch(self.x_f)
                # Packing together the inputs
                Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

                # Getting the prediction
                rho_T, g = self.nn([self.x_f, Train])

                rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

                T_x = tape1.gradient(T, self.x_f)

                T4 = tf.pow(T,4)

                T4_x = tape1.gradient(T4, self.x_f)

                rho_x = tape1.gradient(rho, self.x_f)

            T4_xx = tape2.gradient(T4_x, self.x_f)

            T_xx = tape2.gradient(T_x, self.x_f)

            rho_xx = tape2.gradient(rho_x, self.x_f)

        T_xxx = tape2.gradient(T_xx, self.x_f)

        pde33 = self.epsi ** 2 * T_xxx + (rho_xx - T4_xx)
        #pde33 = (rho_xx - T4_xx)

        return pde33

    def get_pde34(self):

        with tf.GradientTape(persistent=True) as t3:
            t3.watch(self.x_f)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(self.x_f)
                with tf.GradientTape(persistent=True) as t1:
                    t1.watch(self.x_f)
                    # Packing together the inputs
                    Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

                    # Getting the prediction
                    rho_T, g = self.nn([self.x_f, Train])

                    rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

                    T4 = tf.pow(T,4)

                    rho_x = t1.gradient(rho, self.x_f)

                    T4_x = t1.gradient(T4, self.x_f)

                    T_x = t1.gradient(T, self.x_f)

                rho_xx = t2.gradient(rho_x, self.x_f)

                T4_xx = t2.gradient(T4_x, self.x_f)

                T_xx = t2.gradient(T_x, self.x_f)

            T_xxx = t3.gradient(T_xx, self.x_f)

        T_xxxx = t3.gradient(T_xxx, self.x_f)

        pde34 = self.epsi ** 2 * T_xxxx + (
                    rho_xx - T4_xx)

        return pde34

    def get_pde4(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho_T, g = self.nn([self.x_f, Train])

            rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

            rho_x = tape.gradient(rho, self.x_f)

            g_x = tape.gradient(g,self.x_train)

        rho_x_vec = self.get_rho_vec(rho_x)

        T4 = tf.pow(T, 4)

        T4_vec = self.get_rho_vec(T4)

        rho_vec = self.get_rho_vec(rho)

        pde4 = self.epsi*self.v_train*rho_x_vec + self.epsi**2 * self.v_train*g_x - T4_vec + rho_vec + self.epsi*g

        return pde4

    def get_pde5(self):

        rho_T, g = self.nn([self.x_f, self.Tset])

        return self.get_intv(g)

    def get_rho_vec(self, rho):

        sp = tf.ones((self.Nv_f, 1))

        rho_vec = self.Kron_TF(rho, sp)

        rho_vec = tf.reshape(rho_vec, [self.Nx_f * self.Nv_f, 1])

        return rho_vec

    def get_f_bc_loss(self):
        rho_TL, gL = self.nn([tf.zeros((1, 1)), self.Train_BC_L])

        rho_TR, gR = self.nn([tf.ones((1, 1)), self.Train_BC_R])

        rhoL, TL = tf.reshape(rho_TL[:, 0], [1, 1]), tf.reshape(rho_TL[:, 1], [1, 1])

        rhoR, TR = tf.reshape(rho_TR[:, 0], [1, 1]), tf.reshape(rho_TR[:, 1], [1, 1])

        sp = tf.ones((self.nbc, 1))

        rhoL_vec = self.Kron_TF(rhoL, sp)

        rhoL_vec = tf.reshape(rhoL_vec, [self.nbc, 1])

        rhoR_vec = self.Kron_TF(rhoR, sp)

        rhoR_vec = tf.reshape(rhoR_vec, [self.nbc, 1])

        fL = rhoL_vec + self.epsi * gL
        fR = rhoR_vec + self.epsi * gR

        BC1 = tf.reduce_mean(tf.square(fL-self.fL_train))
        BC2 = tf.reduce_mean(tf.square(fR))

        BC3 = tf.reduce_mean(tf.square(TL-self.TL))
        BC4 = tf.reduce_mean(tf.square(TR))

        # print('ss', TR.shape, self.Gamma_T_R_train.shape, self.Gamma_T_pred.shape)
        #
        # dsfgsdfg
        # print('ll', self.C_I_pred, self.C_T_pred)
        # agdfg

        BC = BC1 + BC2 + BC3 + BC4

        # print('bcsp',  fL.shape, BC.shape, BC1.shape, rhoL.shape, rhoL_vec.shape)
        # fsdfsdfsd
        return BC

    # define loss function
    def get_loss(self):
        # loss function contains 3 parts: PDE ( converted to IC), BC and Mass conservation
        # pde
        pde1 = self.get_pde1()

        pde2 = self.get_pde2()

        pde3 = self.get_pde3()

        pde32 = self.get_pde32()

        pde34 = self.get_pde34()

        pde4 = self.get_pde4()

        pde5 = self.get_pde5()

        J1 = tf.reduce_sum(tf.square(pde1))*self.dx

        J2 = tf.reduce_sum(self.get_intv(tf.square(pde2)))*self.dx + tf.reduce_sum(self.get_intv(tf.square(pde4)))*self.dx

        J3 = tf.reduce_sum(tf.square(pde3))*self.dx + tf.reduce_sum(tf.square(pde32))*self.dx + tf.reduce_sum(tf.square(pde34))*self.dx

        J4 = self.Bd_weight * self.get_f_bc_loss()

        J5 = tf.reduce_sum(tf.square(pde5))*self.dx

        loss = J1 + J2 + J3 + J4 + J5
        return loss, J1, J2, J3, J4




    def get_test_error(self):

        rhoT_pred, g_pred = self.nn([self.xt, self.Test])

        rho_pred, T_pred = tf.reshape(rhoT_pred[:, 0], [self.Nx, 1]), tf.reshape(rhoT_pred[:, 1], [self.Nx, 1])

        rho_pred_vec = self.Kron_TF(rho_pred, tf.ones((self.Nv, 1)))

        #print('sf', rho_pred.shape, rho_pred_vec.shape)

        rho_pred_vec = tf.reshape(rho_pred_vec, [self.Nx * self.Nv, 1])

        I_pred = rho_pred_vec + self.epsi*g_pred

        ##### convert to numpy

        # f_diff = (f_pred-self.f_ref_vec).numpy()
        #
        # f_diff = f_diff.reshape(self.Nx, self.Nv)
        #
        # f_diff_sq = np.square(f_diff)

        # test_error = 0
        #
        # for i in range(Nx):
        #     tmp = tf.reduce_sum(f_diff_sq[[i], :] * self.wt_ori.T) / 2
        #     test_error = test_error + tmp
        #
        # test_error = np.sum(test_error) * self.dxt

        ########

        # print('sf', f_diff.shape, f_pred.shape, rho_pred_vec.shape, f_diff_sq.shape)
        # asdasd
        #

        I_diff = I_pred - self.I_ref_vec

        I_diff_sq = tf.square(I_diff)

        sp = tf.ones([self.Nv, 1])

        sk = tf.linspace(np.float32(0), self.Nx - 1, self.Nx, name="linspace")

        sk = tf.reshape(sk, (self.Nx, 1))

        id = self.Kron_TF(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.Nx * self.Nv])

        dup_p = tf.constant([self.Nx, 1], tf.int32)

        weights_rep = tf.tile(self.wt, dup_p)

        e1 = tf.math.segment_sum(weights_rep * I_diff_sq, id)

        # error of I
        eI = tf.reduce_sum(e1)*self.dxt/2

        # error of T
        eT = tf.reduce_sum(tf.square(T_pred-self.T_ref))*self.dxt/2

        # print('ss', T_pred.shape, self.T_ref.shape, (T_pred-self.T_ref).shape)
        # dff

        return eT, eI


    def get_test_f(self):

        rhoT_pred, g_pred = self.nn([self.xt, self.Test])

        rho_pred, T_pred = tf.reshape(rhoT_pred[:, 0], [self.Nx, 1]), tf.reshape(rhoT_pred[:, 1], [self.Nx, 1])

        rho_pred_vec = self.Kron_TF(rho_pred, tf.ones((self.Nv, 1)))

        # print('sf', rho_pred.shape, rho_pred_vec.shape)

        rho_pred_vec = tf.reshape(rho_pred_vec, [self.Nx * self.Nv, 1])

        I_pred = rho_pred_vec + self.epsi * g_pred

        return T_pred, I_pred


    def T_test(self):
        return tf.pow(1 - self.x_f, 1)

    def get_limit_test(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_f)
            T = self.T_test()
            T_4_m_T = 1 / 3 * tf.pow(T, 4) + T

            T_4_m_T_x = tape.gradient(T_4_m_T, self.x_f)

        T_4_m_T_xx = tape.gradient(T_4_m_T_x, self.x_f)

        return T_4_m_T_xx

    def limit_test(self):
        test = self.get_limit_test()
        plt.plot(self.x_f, test)
        plt.show()

    def get_T_eq(self):

        Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

        # Getting the prediction
        rho_T, g = self.nn([self.x_f, Train])

        rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

        eq = 1/3*tf.pow(T,4) + T

        return eq

    def get_limit(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho_T, g = self.nn([self.x_f, Train])

            rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

            T_4_m_T = 1 / 3 * tf.pow(T, 4) + T

            T_4_m_T_x = tape.gradient(T_4_m_T, self.x_f)

        T_4_m_T_xx = tape.gradient(T_4_m_T_x, self.x_f)

        return T_4_m_T_xx

    def get_rho_T_d(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            tape.watch(self.x_f)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho_T, g = self.nn([self.x_f, Train])

            rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

            T4 = tf.pow(T, 4)

            T4_x = tape.gradient(T4, self.x_f)

            rho_x = tape.gradient(rho, self.x_f)

        T4_xx = tape.gradient(T4_x, self.x_f)

        rho_xx = tape.gradient(rho_x, self.x_f)

        return rho_x, T4_x, rho_xx, T4_xx

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

    # define gradient of loss function for optimization step
    def get_grad(self):
        with tf.GradientTape() as tape:
            loss, J1, J2, J3, J4 = self.get_loss()

        return loss, tape.gradient(loss, self.nn.trainable_variables)

    # Extracting weights from NN, and use as initial weights for L-BFGS
    def get_weights(self):
        w = []
        # print('wsp',len(self.nn.trainable_variables), tf.shape_n(self.nn.trainable_variables))
        self.nn.summary()
        for layer in self.nn.layers[2:]:
            weights_biases = layer.get_weights()
            # print('wbsp', len(weights_biases), weights_biases)
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        return tf.convert_to_tensor(w, dtype=self.dtype)

    # Update weights every step in L-BFGS
    def set_weights(self, w):
        for i, layer in enumerate(self.nn.layers[2:]):
            start_weights = sum(self.sizes_w[:i]) + sum(self.sizes_b[:i])
            end_weights = sum(self.sizes_w[:i + 1]) + sum(self.sizes_b[:i])
            weights = w[start_weights:end_weights]
            w_div = int(self.sizes_w[i] / self.sizes_b[i])
            weights = tf.reshape(weights, [w_div, self.sizes_b[i]])
            biases = w[end_weights:end_weights + self.sizes_b[i]]
            weights_biases = [weights, biases]
            layer.set_weights(weights_biases)

    # def fit(self):
    #     start_time = time.time()
    #     for epoch in range(self.num_ad_epochs):
    #         loss, grad = self.get_grad()
    #         elapsed = time.time() - start_time
    #         if epoch % 200 == 0:
    #             print('Epoch: %d, Loss: %.3e, Time: %.2f' %
    #                   (epoch, loss, elapsed))
    #             loss, J1, J2, J3, J4 = self.get_loss()
    #             print('loss 1-5', J1, J2, J3, J4)
    #             with open(self.file_name, 'a') as fw:
    #                 print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
    #                 print('Epoch: %d, Loss: %.3e, Time: %.2f' %
    #                       (epoch, loss, elapsed), file=fw)
    #                 print('loss=', loss, 'J1-J4 are ', J1, J2, J3, J4, file=fw)
    #
    #             rho_real, T_real, I = self.predict()
    #
    #             T_4_real = tf.pow(T_real, 4)
    #
    #             rho_x, T4_x, rho_xx, T4_xx = self.get_rho_T_d()
    #
    #             rho, T = self.predict_tilde()
    #
    #             T_4 = tf.pow(T, 4)
    #
    #             rho_TL, gL = self.nn([tf.zeros((1, 1)), self.Train_BC_L])
    #
    #             rho_TL1, gL1 = self.nn([1 / self.Nx_f * tf.ones((1, 1)), self.Train_BC_L])
    #
    #             print('bcl', rho_TL, rho_TL1)
    #
    #             # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
    #             plt.figure(1)
    #             plt.plot(self.x_f, rho_real, 'r-o', self.x_f, T_real, 'b-*', self.x_f, T_4_real, 'k--')
    #             plt.title('rho and T')
    #             # plt.legend('rho', 'T')
    #             fig = plt.figure(2)
    #             ax = fig.add_subplot(111, projection='3d')
    #             ax.plot_surface(self.xx, self.vv, I.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #             plt.title('f')
    #
    #             T_limit = self.get_limit()
    #             Teq = self.get_T_eq()
    #             plt.figure(4)
    #             plt.plot(self.x_f, T_limit, 'r', self.x_f, Teq, 'b', self.x_f, -(1/3*self.TL**4+self.TL)*(x_f-1), 'c')
    #             plt.title('check limit of T')
    #
    #             plt.figure(5)
    #             plt.plot(self.x_f, rho, 'r-o', self.x_f, T, 'b-*', self.x_f, T_4, 'k--',self.x_f, self.get_intv(self.v_train**2)/2, 'g')
    #             plt.title('tilde')
    #
    #             plt.figure(6)
    #             plt.plot(self.x_f, rho_x, 'r-o', self.x_f, T4_x, 'b-*', self.x_f, rho_xx, 'k--', self.x_f, T4_xx, 'g--')
    #             plt.title('rho T grad')
    #
    #             plt.show()
    #             loss, J1, J2, J3, J4 = self.get_loss()
    #             print('loss 1-5', J1, J2, J3, J4)
    #
    #         if loss < self.stop:
    #             print('training finished')
    #             loss, J1, J2, J3, J4 = self.get_loss()
    #             print('loss 1-5', loss, J1, J2, J3, J4)
    #             rho_real, T_real, I = self.predict()
    #             rho, T = self.predict_tilde()
    #             T_4_real = tf.pow(T_real, 4)
    #             # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
    #             plt.figure(1)
    #             plt.plot(self.x_f, rho_real, 'r-o', self.x_f, T_real, 'b-*', self.x_f, T_4_real, 'k--')
    #             plt.title('rho and T')
    #             plt.legend('rho', 'T')
    #             fig = plt.figure(2)
    #             ax = fig.add_subplot(111, projection='3d')
    #             ax.plot_surface(self.xx, self.vv, I.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #             plt.title('f')
    #
    #             T_limit = self.get_limit()
    #             plt.figure(4)
    #             plt.plot(self.x_f, T_limit)
    #             plt.title('check limit of T')
    #
    #             plt.figure(5)
    #             plt.plot(self.x_f, rho, 'r-o', self.x_f, T, 'b-*')
    #             plt.title('tilde')
    #
    #             plt.show()
    #             break
    #
    #         self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))
    #
    #     def loss_and_flat_grad(w):
    #         # since we are using l-bfgs, the built-in function require
    #         # value_and_gradients_function
    #         with tf.GradientTape() as tape:
    #             self.set_weights(w)
    #             loss, J1, J2, J3, J4 = self.get_loss()
    #
    #         grad = tape.gradient(loss, self.nn.trainable_variables)
    #         grad_flat = []
    #         for g in grad:
    #             grad_flat.append(tf.reshape(g, [-1]))
    #         grad_flat = tf.concat(grad_flat, 0)
    #
    #         return loss, grad_flat
    #
    #     tfp.optimizer.lbfgs_minimize(
    #         loss_and_flat_grad,
    #         initial_position=self.get_weights(),
    #         max_iterations=self.num_bfgs_epochs,
    #         num_correction_pairs=10,
    #         tolerance=1e-6)
    #
    #     rho_real, T_real, I = self.predict()
    #     # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
    #     plt.figure(1)
    #     plt.plot(self.x_f, rho_real, 'r-o', self.x_f, T_real, 'b-*')
    #     plt.title('rho and T')
    #     plt.legend('rho', 'T')
    #     fig = plt.figure(2)
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_surface(self.xx, self.vv, I.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #     plt.title('f')
    #
    #     T_limit = self.get_limit()
    #     plt.figure(4)
    #     plt.plot(self.x_f, T_limit)
    #     plt.title('check limit of T')
    #
    #     plt.show()
    #
    #     final_loss, J1, J2, J3, J4 = self.get_loss()
    #     print('Final loss is %.3e' % final_loss)
    #
    #     with open(self.file_name, 'a') as fw:
    #         print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
    #         print('Final loss=', final_loss, 'J1-J3 are ', J1, J2, J3, J4, file=fw)

    def fit(self):
        start_time = time.time()
        ad_stop = self.num_ad_epochs
        for epoch in range(self.num_ad_epochs):
            loss, grad = self.get_grad()
            elapsed = time.time() - start_time

            if epoch % 50 == 0:

                self.epoch_vec.append(epoch)

                self.emp_loss_vec.append(loss)

                error2ref_T, error2ref_I = self.get_test_error()

                self.test_error_T_vec.append(error2ref_T)
                self.test_error_I_vec.append(error2ref_I)

                print('Adam step: %d, Loss: %.3e, test error T: %.3e, test error I: %.3e' % (epoch, loss, error2ref_T, error2ref_I))



            if epoch % 500 == 0:
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss, elapsed))
                loss, J1, J2, J3, J4 = self.get_loss()
                print('loss 1-5', J1, J2, J3)

                rho_pred, T_pred, I_pred = self.predict()
                T_test_pred, I_test_pred = self.get_test_f()
                fig = plt.figure(1)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, I_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('I pred')
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.vvt, self.I_ref.T)
                plt.title('I ref')

                fig = plt.figure(3)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.vvt, I_test_pred.numpy().reshape(self.Nx, self.Nv).T)
                plt.title('I test pred')

                plt.figure(4)
                plt.plot(self.xt, self.T_ref, 'r', self.xt, T_test_pred, 'b')


                plt.figure(5)
                plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r-o', self.epoch_vec, self.test_error_T_vec, 'b-*', self.epoch_vec, self.test_error_I_vec, 'c--')
                plt.xlabel('iteration')
                plt.ylabel('empirical loss')

                plt.show(block=False)
                loss, J1, J2, J3, J4 = self.get_loss()


                with open(self.file_name, 'a') as fw:
                    print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
                    print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                          (epoch, loss, elapsed), file=fw)
                    print('loss=', loss, 'J1-J4 are ', J1, J2, J3, file=fw)

            if loss < self.stop:
                rho_pred, T_pred, I_pred = self.predict()
                T_test_pred, I_test_pred = self.get_test_f()
                fig = plt.figure(1)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, I_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('I pred')
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.vvt, self.I_ref.T)
                plt.title('I ref')

                fig = plt.figure(3)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.vvt, I_test_pred.numpy().reshape(self.Nx, self.Nv).T)
                plt.title('I test pred')

                plt.figure(4)
                plt.plot(self.xt, self.T_ref, 'r', self.xt, T_test_pred, 'b')

                plt.figure(5)
                plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r-o', self.epoch_vec, self.test_error_T_vec, 'b-*',
                             self.epoch_vec, self.test_error_I_vec, 'c--')
                plt.xlabel('iteration')
                plt.ylabel('empirical loss')

                plt.show(block=False)
                ad_stop = epoch
                print('training finished')
                break

            self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))

        def loss_and_flat_grad(w):
            # since we are using l-bfgs, the built-in function require
            # value_and_gradients_function
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss, J1, J2, J3, J4 = self.get_loss()

            grad = tape.gradient(loss, self.nn.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)

            return loss, grad_flat

        step_size = 500

        cout = int(self.num_bfgs_epochs/step_size)

        for c in range(cout):
            tfp.optimizer.lbfgs_minimize(
                loss_and_flat_grad,
                initial_position=self.get_weights(),
                max_iterations=step_size,
                num_correction_pairs=10,
                tolerance=1e-6)

            loss, J1, J2, J3, J4 = self.get_loss()

            self.epoch_vec.append(ad_stop+step_size*c)

            self.loss_vec.append(loss)

            self.emp_loss_vec.append(loss)

            error2ref_T, error2ref_I = self.get_test_error()

            self.test_error_T_vec.append(error2ref_T)
            self.test_error_I_vec.append(error2ref_I)

            rho_pred, T_pred, I_pred = self.predict()
            T_test_pred, I_test_pred = self.get_test_f()
            fig = plt.figure(1)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.xx, self.vv, I_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
            plt.title('I pred')
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.xxt, self.vvt, self.I_ref.T)
            plt.title('I ref')

            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.xxt, self.vvt, I_test_pred.numpy().reshape(self.Nx, self.Nv).T)
            plt.title('I test pred')

            plt.figure(4)
            plt.plot(self.xt, self.T_ref, 'r', self.xt, T_test_pred, 'b')

            plt.figure(5)
            plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r-o', self.epoch_vec, self.test_error_T_vec, 'b-*',
                         self.epoch_vec, self.test_error_I_vec, 'c--')
            plt.xlabel('iteration')
            plt.ylabel('empirical loss')

            plt.show(block=False)


        final_loss, J1, J2, J3, J4 = self.get_loss()
        print('Final loss is %.3e' % final_loss)

        with open(self.file_name, 'a') as fw:
            print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
            print('Final loss=', final_loss, 'J1-J3 are ', J1, J2, J3, file=fw)

    def predict(self):
        rho_T, g = self.nn([self.x_f, self.Tset])

        rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

        rho_vec = self.get_rho_vec(rho)

        I = rho_vec + self.epsi * g

        T_real = T

        rho_real = self.get_intv(I) * 0.5

        # print('p', g1.shape, g2.shape, g_pred.shape)
        # afdadf

        return rho_real, T_real, I

    def predict_tilde(self):
        rho_T, g = self.nn([self.x_f, self.Tset])

        rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

        return rho, T

    def save(self, model_name):
        self.nn.save(model_name)

    def get_loss_vec(self):
        return self.epoch_vec, self.emp_loss_vec, self.test_error_T_vec, self.test_error_I_vec


if __name__ == "__main__":

    ############################################ Initialization for PINN #####################################

    # input parameters
    # epsi = np.float32(sys.argv[1])
    # Bd_weight = int(sys.argv[2])
    # Nx_f = int(sys.argv[3])
    # nl = int(sys.argv[4])
    # nr = int(sys.argv[5])

    epsi = 1
    Bd_weight = 1
    Nx_f = 80
    nl = 4
    nr = 50

    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv = 1, 1

    # define training set
    # [x_f, v_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f = 60

    dx = 1 / (Nx_f - 1)

    #dx = lx/(Nx_f+1)

    x_p = np.linspace(0 , lx , Nx_f).T[:, None]
    x_f = np.linspace(dx, lx, Nx_f).T[:, None]
    #x_f = np.linspace(0, lx, Nx_f).T[:, None]
    #x_f = np.linspace(dx, lx-dx, Nx_f).T[:, None]
    # dx = 1 / (Nx_f - 1)

    # print('si', x_f.shape)
    #
    # sdfg

    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points = lv * points
    weights = lv * weights
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])

    Tset = np.ones((Nx_f * Nv_f, 2))  # Training set, the first column is x and the second column is v

    for i in range(Nx_f):
        Tset[i * Nv_f:(i + 1) * Nv_f, [0]] = x_f[i][0] * np.ones_like(v_f)
        Tset[i * Nv_f:(i + 1) * Nv_f, [1]] = v_f



    # For BC, there are two BCs, for v>0 and for v<0
    nbc = 60
    v_bc_pos, v_bc_neg = np.random.rand(1, nbc).T, -np.random.rand(1, nbc).T
    # v_bc_pos, v_bc_neg = np.linspace(0, lv, nbc).T[:, None], np.linspace(-lv, 0, nbc).T[:, None]
    x_bc_pos, x_bc_zeros = np.ones((nbc, 1)), np.zeros((nbc, 1))

    Train_BC_L = np.float32(np.concatenate((x_bc_zeros, v_bc_pos), axis=1))
    Train_BC_R = np.float32(np.concatenate((x_bc_pos, v_bc_neg), axis=1))

    #G_x_bc_pos = 1 / epsi * np.ones_like(v_bc_neg)

    #BC_R_Gamma = np.float32(np.concatenate((G_x_bc_pos, v_bc_neg), axis=1))

    fL_train = np.ones_like(v_bc_pos)
    #fL_train = 5*np.sin(v_bc_pos)
    TL = 1


    # define parameter for model
    dtype = tf.float32
    num_ad_epochs = 15000
    num_bfgs_epochs = 6000
    # define adam optimizer
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=1000,
        decay_rate=0.95)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)


    # save model
    def num2str_deciaml(x):
        s = str(x)
        c = ''
        for i in range(len(s)):
            if s[i] == '0':
                c = c + 'z'
            elif s[i] == '.':
                c = c + 'p'
            elif s[i] == '-':
                c = c + 'n'
            else:
                c = c + s[i]

        return c


    ########################################################################################################

    ################################# compute reference solution ############################################
    Nx = 200
    Nv = 80

    lx, lv = 1, 1
    dxr = lx / (Nx + 1)
    x = np.linspace(dxr, lx - dxr, Nx).T[:, None]

    points, weights = np.polynomial.legendre.leggauss(Nv)
    points = lv * points
    weights = lv * weights
    v, w = np.float32(points[:, None]), np.float32(weights[:, None])

    Test_x = np.kron(x, np.ones((Nv, 1)))
    Test_v = np.kron(np.ones((Nx, 1)), v)

    Test = np.concatenate([Test_x, Test_v], axis=1)

    N = Nx * (Nv + 1)

    v_neg, v_pos = v[:int(Nv / 2)], v[int(Nv / 2):]

    TL, TR = 1.0, 0.0

    phiL = np.ones((int(Nv / 2), 1))
    for i in range(int(Nv / 2)):
        #phiL[i] = 5 * np.sin(v[i + int(Nv / 2)])
        phiL[i] = 1

    phiR = np.zeros((int(Nv / 2), 1))


    def nlf(x, epsi, dxr, Nx, Nv, phiL, phiR, TL, TR, v, w):
        F = np.empty(Nx * (Nv + 1))
        N = Nx * Nv
        # x = dx
        for i in range(int(Nv / 2)):
            F[i] = epsi * v[i] * (x[Nv + i] - x[i]) / dxr + x[i] - x[Nx * Nv] ** 4

        for i in range(int(Nv / 2), Nv):
            F[i] = epsi * v[i] * (x[i] - phiL[i - int(Nv / 2)]) / dxr + x[i] - x[Nx * Nv] ** 4

        # x = 2dx:dx:1-2dx
        for l in range(2, Nx):
            for m in range(int(Nv / 2)):
                F[(l - 1) * Nv + m] = epsi * v[m] * (x[(l) * Nv + m] - x[(l - 1) * Nv + m]) / dxr + x[(l - 1) * Nv + m] - \
                                      x[Nx * Nv + (l - 1)] ** 4

            for m in range(int(Nv / 2), Nv):
                F[(l - 1) * Nv + m] = epsi * v[m] * (x[(l - 1) * Nv + m] - x[(l - 2) * Nv + m]) / dxr + x[
                    (l - 1) * Nv + m] - x[Nx * Nv + (l - 1)] ** 4

        # x = 1-dx
        for i in range(int(Nv / 2)):
            F[(Nx - 1) * Nv + i] = epsi * v[i] * (phiR[i] - x[(Nx - 1) * Nv + i]) / dxr + x[(Nx - 1) * Nv + i] - x[
                Nx * (Nv + 1) - 1] ** 4

        for i in range(int(Nv / 2), Nv):
            F[(Nx - 1) * Nv + i] = epsi * v[i] * (x[(Nx - 1) * Nv + i] - x[(Nx - 2) * Nv + i]) / dxr + x[
                (Nx - 1) * Nv + i] - x[Nx * (Nv + 1) - 1] ** 4

        # second pde
        # x=dx
        # tmp = x[:Nv][:,None]
        # print('ss', w.shape, x[:Nv].shape, tmp.shape )
        # adfaf
        F[N] = epsi ** 2 * (x[N + 1] - 2 * x[N] + TL) / dxr ** 2 - x[N] ** 4 + np.sum(w * x[:Nv][:, None]) / 2

        for i in range(1, Nx - 1):
            F[N + i] = epsi ** 2 * (x[N + i + 1] - 2 * x[N + i] + x[N + i - 1]) / dxr ** 2 - x[N + i] ** 4 + np.sum(
                w * x[(i) * Nv:(i + 1) * Nv][:, None]) / 2

        F[N + Nx - 1] = epsi ** 2 * (TR - 2 * x[N + Nx - 1] + x[N + Nx - 2]) / dxr ** 2 - x[N + Nx - 1] ** 4 + np.sum(
            w * x[(Nx - 1) * Nv:Nx * Nv][:, None]) / 2

        return F


    x0 = np.zeros((N, 1))[:, None]
    # x0 = -np.ones((N, 1))
    # x0 = np.random.rand(N,1)

    res = fsolve(nlf, x0, args=(epsi, dxr, Nx, Nv, phiL, phiR, TL, TR, v, w), xtol=1e-5)
    I_ref_vec = res[:Nx * Nv]
    T_ref = res[Nx * Nv:]

    T_ref= T_ref[:,None]
    I_ref_vec = I_ref_vec[:,None]

    I_vec_l2 = 0

    for i in range(Nx):
        I_vec_l2 = I_vec_l2 + np.sum(np.square(I_ref_vec[i * Nv:(i + 1) * Nv]) * w)

    I_vec_l2 = np.sum(I_vec_l2) * dxr

    T_ref_l2 = np.sum(np.square(T_ref))*dxr

    print('ttt', I_vec_l2, T_ref_l2)

    zzzz

    I_ref = I_ref_vec.reshape(Nx, Nv)

    rho_ref = np.zeros((Nx, 1))
    for i in range(Nx):
        rho_ref[i] = np.sum(w * I_ref[i, :][:, None]) / 2

    xxr, vvr = np.meshgrid(x, v)

    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xxr, vvr, I_ref.T)
    # plt.title(r'$I(x,v)$ reference')
    # plt.xlabel('x')
    # plt.ylabel('v')
    #
    #
    # plt.figure(2)
    # plt.plot(x, T_ref, 'r', x, rho_ref, 'b')
    #
    # print(T_ref.shape, rho_ref.shape)
    #
    # plt.show()

    xt, vt, wt, dxt = x, v, w, dxr

    xxt, vvt = np.meshgrid(xt, vt)

    ############################ define and train model ##############################################################


    fname = 'rg_nonl_epsi_' + num2str_deciaml(epsi) + '_bd_' + str(Bd_weight) + '_Nx_' + str(
        Nx_f) + '_nl_' + str(nl) + '_nr_' + str(nr)

    file_name = fname + '.txt'

    # define model
    mdl = stdst(epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Bd_weight, Tset, Train_BC_L, Train_BC_R, fL_train, TL, nbc, weights_vf, dx,
                dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl, nr, Nx, Nv, xt, vt, dxt, wt, Test, I_ref, I_ref_vec, T_ref)

    # train model
    mdl.fit()

    model_name = fname + '.h5'
    mdl.save('mdls/' + model_name)

    rho_pred, T_pred, I_pred = mdl.predict()

    T_limit = mdl.get_limit()

    I_pred = I_pred.numpy().reshape(Nx_f, Nv_f)
    rho_pred = rho_pred.numpy()
    T_pred = T_pred.numpy()

    T_4_pred = tf.pow(T_pred, 4)

    # test_f0 = f_0_vec.reshape(Nx_f,Nv_f)

    xx, vv = np.meshgrid(x_f, v_f)

    epoch_vec, emp_loss_vec, test_error_T_vec,  test_error_I_vec= mdl.get_loss_vec()

    T_test_pred, I_test_pred = mdl.get_test_f()

    npy_name = fname + '.npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, x_f)
        np.save(ss, rho_pred)
        np.save(ss, x)
        np.save(ss, rho_ref)
        np.save(ss, T_ref)
        np.save(ss, xx)
        np.save(ss, vv)
        np.save(ss, I_pred)
        np.save(ss, xxt)
        np.save(ss, vvt)
        np.save(ss, T_test_pred)
        np.save(ss, I_ref)
        np.save(ss, I_test_pred)
        np.save(ss, epoch_vec)
        np.save(ss, emp_loss_vec)
        np.save(ss, test_error_T_vec)
        np.save(ss, test_error_I_vec)









