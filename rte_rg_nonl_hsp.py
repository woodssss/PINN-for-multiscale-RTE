import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys
import scipy.special as sc


class stdst():
    # this is for linear transport equation epsi * \partial_t f + v \partial_x f = 1/epsi * L(f)
    # L(f) = 1/2 \int_-1^1 f  - f
    # the expect limit system is \partial_t rho - 1/3 \partial_xx rho = 0
    def __init__(self, epsi, Nx_f, Nv_f, x_f, x_p, v_f, lx, lv, Bd_weight, Tset, Test, Train_BC_L, Train_BC_R, C_T_pred,
                 C_I_pred, Gamma_T_R_train, Gamma_I_R_train, Gamma_T_pred ,Gamma_T_pred_p, Gamma_I_pred_p, Gamma_I_pred_vec_p, nbc, weights, dx,
                 dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl, nr):
        self.dtype = dtype
        self.epsi, self.Nx_f, self.Nv_f = epsi, Nx_f, Nv_f
        self.lx, self.lv, self.dx = lx, lv, dx
        self.Bd_weight = Bd_weight
        self.xx, self.vv = np.meshgrid(x_f, v_f)
        self.nbc = nbc

        # number of layers for rho and g
        self.nl, self.nr = nl, nr

        self.stop = 0.0005

        self.file_name = file_name
        # convert np array to tensor
        self.x_f, self.v_f = tf.convert_to_tensor(x_f, dtype=self.dtype), tf.convert_to_tensor(v_f,
                                                                                               dtype=self.dtype)  # x_f and v_f are grid on x and v direction
        self.x_train, self.v_train = tf.convert_to_tensor(Tset[:, [0]], dtype=self.dtype), tf.convert_to_tensor(
            Tset[:, [1]], dtype=self.dtype)  # x_train and v_train are input trainning set for NN

        self.Tset = tf.convert_to_tensor(Tset, dtype=self.dtype)
        self.Test = tf.convert_to_tensor(Test, dtype=self.dtype)

        self.x_p = tf.convert_to_tensor(x_p, dtype=self.dtype)

        self.weights = tf.convert_to_tensor(weights, dtype=self.dtype)

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        # define BC
        self.Train_BC_L = tf.convert_to_tensor(Train_BC_L, dtype=self.dtype)
        self.Train_BC_R = tf.convert_to_tensor(Train_BC_R, dtype=self.dtype)

        self.C_T_pred = tf.convert_to_tensor(C_T_pred, dtype=self.dtype)
        self.C_I_pred = tf.convert_to_tensor(C_I_pred, dtype=self.dtype)
        self.Gamma_T_R_train = tf.convert_to_tensor(Gamma_T_R_train, dtype=self.dtype)
        self.Gamma_I_R_train = tf.convert_to_tensor(Gamma_I_R_train, dtype=self.dtype)
        self.Gamma_T_pred = tf.convert_to_tensor(Gamma_T_pred, dtype=self.dtype)
        self.Gamma_T_pred_p = tf.convert_to_tensor(Gamma_T_pred_p, dtype=self.dtype)
        self.Gamma_I_pred_p = tf.convert_to_tensor(Gamma_I_pred_p, dtype=self.dtype)
        self.Gamma_I_pred_vec_p = tf.convert_to_tensor(Gamma_I_pred_vec_p, dtype=self.dtype)

        self.TL = 1

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

            T_x = tape.gradient(T, self.x_f)

        T_xx = tape.gradient(T_x, self.x_f)

        T_xx_vec = self.get_rho_vec(T_xx)

        rho_x_vec = self.get_rho_vec(rho_x)

        int_vg = self.get_intv(self.v_train * g_x)

        int_vg_vec = self.get_rho_vec(int_vg) * 0.5

        pde2 = self.v_train * rho_x_vec + self.epsi * self.v_train * g_x - self.epsi * T_xx_vec + g

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
                    rho - tf.pow(T, 4) - 4 * T * tf.pow(self.Gamma_T_pred, 3) - 6 * tf.pow(T, 2) * tf.pow(
                self.Gamma_T_pred, 2) - 4 * self.Gamma_T_pred * tf.pow(T, 3))
        # print('sss', pde3.shape, T.shape, self.Gamma_T_pred.shape)
        #
        # dasd

        return pde3

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

            T_x = tape.gradient(T, self.x_f)

            rho_x = tape.gradient(rho, self.x_f)

            g_x = tape.gradient(g, self.x_train)

        T_bundle = tf.pow(T, 4) + 4*tf.pow(T, 3)*self.Gamma_T_pred +6 * tf.pow(T, 2) * tf.pow(
                self.Gamma_T_pred, 2) + 4 * self.Gamma_T_pred * tf.pow(T, 3)

        T_bundle_vec = self.get_rho_vec(T_bundle)

        rho_vec = self.get_rho_vec(rho)

        rho_x_vec = self.get_rho_vec(rho_x)

        pde4 = self.epsi*self.v_train*rho_x_vec + self.epsi**2 * self.v_train*g_x -T_bundle_vec + rho_vec+self.epsi*g

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

        BC1 = tf.reduce_mean(tf.square(fL))
        BC2 = tf.reduce_mean(tf.square(fR + self.Gamma_I_R_train))

        BC3 = tf.reduce_mean(tf.square(TL))
        BC4 = tf.reduce_mean(tf.square(TR + self.Gamma_T_R_train))

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

        # pde32 = self.get_pde32()
        #
        # pde34 = self.get_pde34()

        pde4 = self.get_pde4()

        pde5 = self.get_pde5()

        J1 = tf.reduce_sum(tf.square(pde1)) * self.dx

        J2 = tf.reduce_sum(self.get_intv(tf.square(pde2))) * self.dx #+ tf.reduce_sum(self.get_intv(tf.square(pde4))) * self.dx

        J3 = tf.reduce_sum(tf.square(pde3)) * self.dx

        J4 = self.Bd_weight * self.get_f_bc_loss()

        J5 = tf.reduce_sum(tf.square(pde5)) * self.dx

        loss = J1 + J2 + J3 + J4 + J5
        return loss, J1, J2, J3, J4

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

        T_real = T + self.Gamma_T_pred

        eq = 1/3*tf.pow(T_real, 4) + T_real

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

            T_real = T + self.Gamma_T_pred

            T_4_m_T = 1 / 3 * tf.pow(T_real, 4) + T_real

            T_4_m_T_x = tape.gradient(T_4_m_T, self.x_f)

        T_4_m_T_xx = tape.gradient(T_4_m_T_x, self.x_f)

        return T_4_m_T_xx

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

    def fit(self):
        start_time = time.time()
        for epoch in range(self.num_ad_epochs):
            loss, grad = self.get_grad()
            elapsed = time.time() - start_time
            if epoch % 500 == 0:
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss, elapsed))
                loss, J1, J2, J3, J4 = self.get_loss()
                print('loss 1-5', J1, J2, J3, J4)
                with open(self.file_name, 'a') as fw:
                    print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
                    print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                          (epoch, loss, elapsed), file=fw)
                    print('loss=', loss, 'J1-J4 are ', J1, J2, J3, J4, file=fw)

                rho_real, T_real, I = self.predict()

                T_4_real = tf.pow(T_real, 4)

                rho, T = self.predict_tilde()

                T_4 = tf.pow(T, 4)

                rho_TL, gL = self.nn([tf.zeros((1, 1)), self.Train_BC_L])

                rho_TL1, gL1 = self.nn([1 / self.Nx_f * tf.ones((1, 1)), self.Train_BC_L])

                print('bcl', rho_TL, rho_TL1)

                # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
                plt.figure(1)
                plt.plot(self.x_f, rho_real, 'r-o', self.x_f, T_real, 'b-*', self.x_f, T_4_real, 'k--')
                plt.title('rho and T')
                # plt.legend('rho', 'T')
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, I.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('f')

                T_limit = self.get_limit()
                Teq = self.get_T_eq()
                plt.figure(4)
                plt.plot(self.x_f, T_limit, 'r', self.x_f, Teq, 'b', self.x_f, -(1/3*self.C_T_pred**4+self.C_T_pred)*(x_f-1), 'c')
                plt.title('check limit of T')

                plt.figure(5)
                plt.plot(self.x_f, rho, 'r-o', self.x_f, T, 'b-*', self.x_f, T_4, 'k--')
                plt.title('tilde')

                plt.show()
                loss, J1, J2, J3, J4 = self.get_loss()
                print('loss 1-5', J1, J2, J3, J4)

            if loss < self.stop:
                print('training finished')
                loss, J1, J2, J3, J4 = self.get_loss()
                print('loss 1-5', loss, J1, J2, J3, J4)
                rho_real, T_real, I = self.predict()
                rho, T = self.predict_tilde()
                T_4_real = tf.pow(T_real, 4)
                # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
                plt.figure(1)
                plt.plot(self.x_f, rho_real, 'r-o', self.x_f, T_real, 'b-*', self.x_f, T_4_real, 'k--')
                plt.title('rho and T')
                plt.legend('rho', 'T')
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, I.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('f')

                T_limit = self.get_limit()
                plt.figure(4)
                plt.plot(self.x_f, T_limit)
                plt.title('check limit of T')

                plt.figure(5)
                plt.plot(self.x_f, rho, 'r-o', self.x_f, T, 'b-*')
                plt.title('tilde')

                plt.show()
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

        tfp.optimizer.lbfgs_minimize(
            loss_and_flat_grad,
            initial_position=self.get_weights(),
            max_iterations=self.num_bfgs_epochs,
            num_correction_pairs=10,
            tolerance=1e-8)

        rho_real, T_real, I = self.predict()
        # print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
        plt.figure(1)
        plt.plot(self.x_f, rho_real, 'r-o', self.x_f, T_real, 'b-*')
        plt.title('rho and T')
        plt.legend('rho', 'T')
        fig = plt.figure(2)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.xx, self.vv, I.numpy().reshape(self.Nx_f, self.Nv_f).T)
        plt.title('f')

        T_limit = self.get_limit()
        plt.figure(4)
        plt.plot(self.x_f, T_limit)
        plt.title('check limit of T')

        plt.show()

        final_loss, J1, J2, J3, J4 = self.get_loss()
        print('Final loss is %.3e' % final_loss)

        with open(self.file_name, 'a') as fw:
            print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
            print('Final loss=', final_loss, 'J1-J3 are ', J1, J2, J3, J4, file=fw)

    def predict(self):
        rho_T, g = self.nn([self.x_p, self.Test])

        rho, T = tf.reshape(rho_T[:, 0], [self.Nx_f, 1]), tf.reshape(rho_T[:, 1], [self.Nx_f, 1])

        rho_vec = self.get_rho_vec(rho)

        I = rho_vec + self.epsi * g + self.Gamma_I_pred_vec_p

        T_real = T + self.Gamma_T_pred_p

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


if __name__ == "__main__":
    # input parameters

    epsi = np.float32(sys.argv[1])
    Bd_weight = int(sys.argv[2])
    Nx_f = int(sys.argv[3])
    nl = int(sys.argv[4])
    nr = int(sys.argv[5])

    # epsi = 1
    # Bd_weight = 1
    # Nx_f = 80
    # nl = 4
    # nr = 50

    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv = 1, 1

    # define training set
    # [x_f, v_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f = 60

    # dx = 1 / (Nx_f - 1)

    dx = 1 / Nx_f

    x_p = np.linspace(0 , lx , Nx_f).T[:, None]
    x_f = np.linspace(dx, lx, Nx_f).T[:, None]

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

    tsp = np.ones((Nv_f, 1))
    tsk = np.ones((Nx_f, 1))

    Test_v = np.kron(tsk, v_f)
    Test_x = np.kron(x_p, tsp)

    Test = np.concatenate([Test_x, Test_v], axis=1)



    # For BC, there are two BCs, for v>0 and for v<0
    nbc = 80
    v_bc_pos, v_bc_neg = np.random.rand(1, nbc).T, -np.random.rand(1, nbc).T
    # v_bc_pos, v_bc_neg = np.linspace(0, lv, nbc).T[:, None], np.linspace(-lv, 0, nbc).T[:, None]
    x_bc_pos, x_bc_zeros = np.ones((nbc, 1)), np.zeros((nbc, 1))

    Train_BC_L = np.float32(np.concatenate((x_bc_zeros, v_bc_pos), axis=1))
    Train_BC_R = np.float32(np.concatenate((x_bc_pos, v_bc_neg), axis=1))

    G_x_bc_pos = 1 / epsi * np.ones_like(v_bc_neg)

    BC_R_Gamma = np.float32(np.concatenate((G_x_bc_pos, v_bc_neg), axis=1))

    # Load pretrained Gamma NN
    G_file_name = 'NLRTE_hsp_epsi_1_jw_2_Nx_800_nlr_4_nlg_4_nur_50_nug_50.h5'

    G_mdl = tf.keras.models.load_model(G_file_name)
    # G_mdl = load_model(G_file_name)

    C_T_pred, C_I_pred = G_mdl([10 * np.ones((1, 1)), np.array([[10, 0]])])

    C_T_pred, C_I_pred = C_T_pred.numpy(), C_I_pred.numpy()

    # print('limit', C_T_pred, C_I_pred)
    #
    # asdfasdf

    F_T_R_train, F_I_R_train = G_mdl([1 / epsi * np.ones((1, 1)), BC_R_Gamma])
    Gamma_T_R_train, Gamma_I_R_train = F_T_R_train , F_I_R_train

    if 1 / epsi > 10:
        Gamma_T_R_train = C_T_pred * np.ones_like(Gamma_T_R_train)
        Gamma_I_R_train = C_I_pred * np.ones_like(Gamma_I_R_train)

    # print('tttr', C_T_pred, C_I_pred, Gamma_T_R_train, Gamma_I_R_train)
    #
    # asdfg

    # Now, we compute Gamma_I(x_f/epsi, v) and Gamma_T(x_f/epsi)
    # Firstly, we need x_G = x/epsi

    lx_G = 10
    x_cut = 1 / epsi
    if epsi >= 1 / lx_G:
        # in this case, epsi is large, x/epsi are all in [0, lx_G]
        x_f_G = x_f / epsi
        sp = np.ones((Nv_f, 1))
        sk = np.ones((Nx_f, 1))
        x_train_G = np.kron(x_f_G, sp)
        v_train_G = np.kron(sk, v_f)
        Train_G = np.concatenate([x_train_G, v_train_G], axis=1)
        F_T_pred, F_I_pred = G_mdl([x_f_G, Train_G])
        Gamma_I_pred = F_I_pred.numpy().reshape(Nx_f, Nv_f)
        Gamma_I_pred_vec = F_I_pred.numpy()

        Gamma_T_pred = F_T_pred

        # G = G_mdl(Train_G).numpy().reshape(Nx_f, Nv_f)

    elif epsi < 1 / lx_G and epsi >= dx / 10:
        # in this case, epsi is small, only part of x/epsi in [0, lx_G]
        N_cut = int(np.floor(lx_G * epsi / dx))
        x_f_G = x_f[:N_cut] / epsi
        # print('ts', dx, 1/80, x_f[0], x_f[1], x_f_G)
        sp = np.ones((Nv_f, 1))
        sk = np.ones((N_cut, 1))
        x_train_G = np.kron(x_f_G, sp)
        v_train_G = np.kron(sk, v_f)
        Train_G = np.concatenate([x_train_G, v_train_G], axis=1)

        F_T_pred_1, F_I_pred_1 = G_mdl([x_f_G, Train_G])

        F_T_pred_2, F_I_pred_2 = C_T_pred * np.ones(((Nx_f - N_cut), 1)), C_I_pred * np.ones(((Nx_f - N_cut) * Nv_f, 1))

        F_T_pred = np.concatenate([F_T_pred_1, F_T_pred_2], axis=0)

        F_I_pred = np.concatenate([F_I_pred_1, F_I_pred_2], axis=0)

        Gamma_I_pred = F_I_pred.reshape(Nx_f, Nv_f)

        Gamma_I_pred_vec = F_I_pred

        Gamma_T_pred = F_T_pred
    else:
        F_T_pred, F_I_pred = C_T_pred * np.ones((Nx_f, 1)), C_I_pred * np.ones((Nx_f * Nv_f, 1))

        Gamma_I_pred = F_I_pred.reshape(Nx_f, Nv_f)

        Gamma_I_pred_vec = F_I_pred

        Gamma_T_pred = F_T_pred


    if epsi >= 1 / lx_G:
        # in this case, epsi is large, x/epsi are all in [0, lx_G]
        x_p_G = x_p / epsi
        sp = np.ones((Nv_f, 1))
        sk = np.ones((Nx_f, 1))
        x_train_G = np.kron(x_p_G, sp)
        v_train_G = np.kron(sk, v_f)
        Train_G = np.concatenate([x_train_G, v_train_G], axis=1)
        F_T_pred, F_I_pred = G_mdl([x_p_G, Train_G])
        Gamma_I_pred_p = F_I_pred.numpy().reshape(Nx_f, Nv_f)
        Gamma_I_pred_vec_p = F_I_pred.numpy()

        Gamma_T_pred_p = F_T_pred

        # G = G_mdl(Train_G).numpy().reshape(Nx_f, Nv_f)

    elif epsi < 1 / lx_G and epsi >= dx / 10:
        # in this case, epsi is small, only part of x/epsi in [0, lx_G]
        N_cut = int(np.floor(lx_G * epsi / dx))
        x_f_G = x_f[:N_cut] / epsi
        # print('ts', dx, 1/80, x_f[0], x_f[1], x_f_G)
        sp = np.ones((Nv_f, 1))
        sk = np.ones((N_cut, 1))
        x_train_G = np.kron(x_f_G, sp)
        v_train_G = np.kron(sk, v_f)
        Train_G = np.concatenate([x_train_G, v_train_G], axis=1)

        F_T_pred_1, F_I_pred_1 = G_mdl([x_f_G, Train_G])

        F_T_pred_2, F_I_pred_2 = C_T_pred * np.ones(((Nx_f - N_cut), 1)), C_I_pred * np.ones(((Nx_f - N_cut) * Nv_f, 1))

        F_T_pred = np.concatenate([F_T_pred_1, F_T_pred_2], axis=0)

        F_I_pred = np.concatenate([F_I_pred_1, F_I_pred_2], axis=0)

        Gamma_I_pred_p = F_I_pred.reshape(Nx_f, Nv_f)

        Gamma_I_pred_vec_p = F_I_pred

        Gamma_T_pred_p = F_T_pred
    else:
        Train_G = np.concatenate([np.zeros_like(v_f), v_f], axis=1)

        F_T_pred_1, F_I_pred_1 = G_mdl([np.zeros((1, 1)), Train_G])

        F_T_pred_2, F_I_pred_2 = C_T_pred * np.ones(((Nx_f - 1), 1)), C_I_pred * np.ones(((Nx_f - 1) * Nv_f, 1))

        F_T_pred = np.concatenate([F_T_pred_1, F_T_pred_2], axis=0)

        F_I_pred = np.concatenate([F_I_pred_1, F_I_pred_2], axis=0)

        Gamma_I_pred_p = F_I_pred.reshape(Nx_f, Nv_f)

        Gamma_I_pred_vec_p = F_I_pred

        Gamma_T_pred_p = F_T_pred



    plt.plot(x_f, Gamma_T_pred, 'r-o')
    plt.show()

    # print('bc', Train_BC_L, Train_BC_R)

    # define parameter for model
    dtype = tf.float32
    num_ad_epochs = 15000
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


    file_name = 'rg_nonl_hsp_epsi_' + num2str_deciaml(epsi) + '_jw_' + str(Bd_weight) + '_Nx_' + str(
        Nx_f) + '_nl_' + str(nl) + '_nr_' + str(nr) + '.' + 'txt'

    # define model
    mdl = stdst(epsi, Nx_f, Nv_f, x_f, x_p, v_f, lx, lv, Bd_weight, Tset, Test, Train_BC_L, Train_BC_R, C_T_pred, C_I_pred,
                Gamma_T_R_train, Gamma_I_R_train, Gamma_T_pred ,Gamma_T_pred_p, Gamma_I_pred_p, Gamma_I_pred_vec_p, nbc, weights_vf, dx,
                dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl, nr)

    # train model
    mdl.fit()

    model_name = 'NLNTE_hsp_epsi_' + num2str_deciaml(epsi) + '_jw_' + str(Bd_weight) + '_Nx_' + str(
        Nx_f) + '_nl_' + str(nl) + '_nr_' + str(nr) + '.' + 'h5'
    mdl.save('mdls/' + model_name)

    rho_pred, T_pred, I_pred = mdl.predict()

    I_pred = I_pred.numpy().reshape(Nx_f, Nv_f)
    rho_pred = rho_pred.numpy()
    T_pred = T_pred.numpy()

    T_4_pred = tf.pow(T_pred, 4)

    # test_f0 = f_0_vec.reshape(Nx_f,Nv_f)

    xx, vv = np.meshgrid(x_f, v_f)

    T_limit = mdl.get_limit()

    npy_name = 'rg_nonl_hsp_epsi_' + num2str_deciaml(epsi) + '_nl_' + str(nl) + '_nr_' + str(nr) + '.' + 'npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, x_f)
        np.save(ss, rho_pred)
        np.save(ss, xx)
        np.save(ss, vv)
        np.save(ss, T_pred)
        np.save(ss, I_pred)
        np.save(ss, Gamma_I_pred)
        np.save(ss, Gamma_T_pred)
        np.save(ss, C_I_pred)
        np.save(ss, C_T_pred)
        np.save(ss, T_limit)


