import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.sparse.linalg
import scipy as scp
import time
import sys
import scipy.special as sc
# from tensorflow.keras.models import load_model
from keras.models import load_model


# This is for rho g decomposition with solution to the half space problem as corrector.

class stdst():
    def __init__(self, epsi, Nx_f, Ny_f, Nv_f, x_f, x_p, y_f, v_f, lx, ly, lv, Bd_weight, Txy_set, Txy_pset, Tset,
                 Tpset, Train_BC_L, Train_BC_R, Train_BC_T, Train_BC_B, Train_BC_L_r, Train_BC_R_r, Train_BC_T_r,
                 Train_BC_B_r, fL_train, fR_train, fT_train, fB_train, Gamma_y, Gamma, Gamma_p, Gamma_R, Gamma_T,
                 Gamma_B, hL, nxbc, nybc, nvbc, weights_vf, dx, dy, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name,
                 nl, nr, Nx1, Nx2, Ny, Nv, xt, yt, vt, dxt1, dxt2, dyt, wt, Test_xy, Test, rho_ref, f_ref, Gamma_t):
        self.dtype = dtype
        self.epsi, self.Nx_f, self.Ny_f, self.Nv_f = epsi, Nx_f, Ny_f, Nv_f
        self.lx, self.ly, self.lv = lx, ly, lv
        self.dx, self.dy = dx, dy
        self.Bd_weight = Bd_weight
        self.xx, self.yy = np.meshgrid(x_f, y_f)
        self.xxp, self.yy = np.meshgrid(x_p, y_f)
        self.xxt, self.yyt = np.meshgrid(xt, yt)
        self.nxbc, self.nybc, self.nvbc = nxbc, nybc, nvbc
        self.Gamma = tf.convert_to_tensor(Gamma, dtype=self.dtype)
        self.Gamma_y = tf.convert_to_tensor(Gamma_y, dtype=self.dtype)

        # number of layers for rho and g
        self.nl, self.nr = nl, nr

        self.stop = 0.01

        self.file_name = file_name
        # convert np array to tensor
        self.x_f, self.v_f = tf.convert_to_tensor(x_f, dtype=self.dtype), tf.convert_to_tensor(v_f,
                                                                                               dtype=self.dtype)  # x_f and v_f are grid on x and v direction
        self.y_f = tf.convert_to_tensor(y_f, dtype=self.dtype)
        self.x_p = tf.convert_to_tensor(x_p, dtype=self.dtype)

        self.x_train, self.y_train, self.v_train = tf.convert_to_tensor(Tset[:, [0]],
                                                                        dtype=self.dtype), tf.convert_to_tensor(
            Tset[:, [1]], dtype=self.dtype), tf.convert_to_tensor(Tset[:, [2]],
                                                                  dtype=self.dtype)  # x_train and v_train are input trainning set for NN
        self.Tset = tf.convert_to_tensor(Tset, dtype=self.dtype)
        self.Tpset = tf.convert_to_tensor(Tpset, dtype=self.dtype)

        self.xr, self.yr = tf.convert_to_tensor(Txy_set[:, [0]], dtype=self.dtype), tf.convert_to_tensor(
            Txy_set[:, [1]], dtype=self.dtype)
        self.Txy_set = tf.convert_to_tensor(Txy_set, dtype=self.dtype)
        self.Txy_pset = tf.convert_to_tensor(Txy_pset, dtype=self.dtype)

        self.weights = tf.convert_to_tensor(weights_vf, dtype=self.dtype)

        self.dxt1, self.dxt2, self.dyt, self.Nx1, self.Nx2, self.Ny, self.Nv = dxt1, dxt2, dyt, Nx1, Nx2, Ny, Nv
        self.Nx = self.Nx1+self.Nx2
        self.wt = tf.convert_to_tensor(wt, dtype=self.dtype)

        self.wt_ori = wt

        self.xt, self.yt, self.vt = tf.convert_to_tensor(xt, dtype=self.dtype), tf.convert_to_tensor(yt,
                                                                                                     dtype=self.dtype), tf.convert_to_tensor(
            vt, dtype=self.dtype)
        self.Test_xy = tf.convert_to_tensor(Test_xy, dtype=self.dtype)
        self.Test = tf.convert_to_tensor(Test, dtype=self.dtype)

        self.rho_ref = tf.convert_to_tensor(rho_ref, dtype=self.dtype)
        self.f_ref = tf.convert_to_tensor(f_ref, dtype=self.dtype)
        self.Gamma_t = tf.convert_to_tensor(Gamma_t, dtype=self.dtype)


        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        # define BC
        self.Train_BC_L = tf.convert_to_tensor(Train_BC_L, dtype=self.dtype)
        self.Train_BC_R = tf.convert_to_tensor(Train_BC_R, dtype=self.dtype)
        self.Train_BC_T = tf.convert_to_tensor(Train_BC_T, dtype=self.dtype)
        self.Train_BC_B = tf.convert_to_tensor(Train_BC_B, dtype=self.dtype)

        self.Train_BC_L_r = tf.convert_to_tensor(Train_BC_L_r, dtype=self.dtype)
        self.Train_BC_R_r = tf.convert_to_tensor(Train_BC_R_r, dtype=self.dtype)
        self.Train_BC_T_r = tf.convert_to_tensor(Train_BC_T_r, dtype=self.dtype)
        self.Train_BC_B_r = tf.convert_to_tensor(Train_BC_B_r, dtype=self.dtype)

        self.fL_train = tf.convert_to_tensor(fL_train, dtype=self.dtype)
        self.fR_train = tf.convert_to_tensor(fR_train, dtype=self.dtype)
        self.fT_train = tf.convert_to_tensor(fT_train, dtype=self.dtype)
        self.fB_train = tf.convert_to_tensor(fB_train, dtype=self.dtype)

        # define corrector
        self.Gamma = tf.convert_to_tensor(Gamma, dtype=self.dtype)
        self.Gamma_y = tf.convert_to_tensor(Gamma_y, dtype=self.dtype)

        self.Gamma_p = tf.convert_to_tensor(Gamma_p, dtype=self.dtype)
        self.Gamma_R = tf.convert_to_tensor(Gamma_R, dtype=self.dtype)
        self.Gamma_T = tf.convert_to_tensor(Gamma_T, dtype=self.dtype)
        self.Gamma_B = tf.convert_to_tensor(Gamma_B, dtype=self.dtype)

        self.hL = tf.convert_to_tensor(hL, dtype=self.dtype)

        # track loss
        self.epoch_vec = []
        self.loss_vec = []

        self.emp_loss_vec = []
        self.test_error_vec = []

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

    def convert_sparse_matrix_to_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    # define model
    def get_nn(self):

        # define nn for rho
        input_rho = tf.keras.Input(shape=(2,))

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

        output_rho = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer='glorot_normal')(
            input_rho_mid)

        # define nn for g

        input_g = tf.keras.Input(shape=(3,))

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

    # define some function for calculation L2 loss
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

    def get_intv(self, f):
        # compute average

        sp = tf.ones([self.Nv_f, 1])

        sk = tf.linspace(np.float32(0), self.Nx_f * self.Ny_f - 1, self.Nx_f * self.Ny_f, name="linspace")

        sk = tf.reshape(sk, (self.Nx_f * self.Ny_f, 1))

        id = self.Kron_TF(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.Nx_f * self.Nv_f * self.Ny_f])

        dup_p = tf.constant([self.Nx_f * self.Ny_f, 1], tf.int32)

        weights_rep = tf.tile(self.weights, dup_p)

        res = tf.math.segment_sum(weights_rep * f, id)

        # print('ssp', weights_rep.shape, res.shape)

        return res

    def get_rho_vec(self, rho):
        # extend rho(x,y) to same dimension as (x,y,theta)

        sp = tf.ones((self.Nv_f, 1))

        rho_vec = self.Kron_TF(rho, sp)

        rho_vec = tf.reshape(rho_vec, [self.Nx_f * self.Nv_f * self.Ny_f, 1])

        return rho_vec

    # define loss function
    def get_pde1(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.y_train)
            tape.watch(self.v_train)
            tape.watch(self.xr)
            tape.watch(self.yr)
            # Packing together the inputs
            Train_rho = tf.stack([self.xr[:, 0], self.yr[:, 0]], axis=1)
            Train_g = tf.stack([self.x_train[:, 0], self.y_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho, g = self.nn([Train_rho, Train_g])

            # print('nns', rho.shape, g.shape)

            # use the first output for transportation
            g_x = tape.gradient(g, self.x_train)
            g_y = tape.gradient(g, self.y_train)

        v_grad_g = tf.cos(self.v_train) * g_x + tf.sin(self.v_train) * g_y

        avg_v_grad_g = 1 / 2 / np.pi * self.get_intv(v_grad_g)

        # print('sp1', g_x.shape, vg_x.shape)

        pde1 = avg_v_grad_g + 1 / self.epsi * 1 / 2 / np.pi * self.get_intv(tf.sin(self.v_train) * self.Gamma_y)

        return pde1

    def get_pde2(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.y_train)
            tape.watch(self.xr)
            tape.watch(self.yr)
            # Packing together the inputs
            Train_rho = tf.stack([self.xr[:, 0], self.yr[:, 0]], axis=1)
            Train_g = tf.stack([self.x_train[:, 0], self.y_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho, g = self.nn([Train_rho, Train_g])

            # print('nns', rho.shape, g.shape)

            rho_x = tape.gradient(rho, self.xr)
            rho_y = tape.gradient(rho, self.yr)

            g_x = tape.gradient(g, self.x_train)
            g_y = tape.gradient(g, self.y_train)

        rho_x_vec = self.get_rho_vec(rho_x)
        rho_y_vec = self.get_rho_vec(rho_y)
        # print('sp1', g_x.shape, vg_x.shape)

        pde2 = tf.cos(self.v_train) * rho_x_vec + tf.sin(self.v_train) * rho_y_vec + self.epsi * tf.cos(
            self.v_train) * g_x + self.epsi * tf.sin(self.v_train) * g_y + tf.sin(self.v_train) * self.Gamma_y + g

        return pde2

    def get_f_bc_loss(self):
        rhoL, gL = self.nn([self.Train_BC_L_r, self.Train_BC_L])
        rhoR, gR = self.nn([self.Train_BC_R_r, self.Train_BC_R])
        rhoT, gT = self.nn([self.Train_BC_T_r, self.Train_BC_T])
        rhoB, gB = self.nn([self.Train_BC_B_r, self.Train_BC_B])

        # print('spp', rhoL.shape, gL.shape, self.nybc, self.nvbc)

        sp = tf.ones((self.nvbc, 1))

        rhoL_vec = self.Kron_TF(rhoL, sp)

        rhoL_vec = tf.reshape(rhoL_vec, [self.nvbc * self.nybc, 1])

        rhoR_vec = self.Kron_TF(rhoR, sp)

        rhoR_vec = tf.reshape(rhoR_vec, [self.nvbc * self.nybc, 1])

        rhoT_vec = self.Kron_TF(rhoT, sp)

        rhoT_vec = tf.reshape(rhoT_vec, [self.nvbc * self.nxbc, 1])

        rhoB_vec = self.Kron_TF(rhoB, sp)

        rhoB_vec = tf.reshape(rhoB_vec, [self.nvbc * self.nxbc, 1])

        fL = rhoL_vec + self.epsi * gL
        fR = rhoR_vec + self.epsi * gR + self.Gamma_R
        fT = rhoT_vec + self.epsi * gT + self.Gamma_T
        fB = rhoB_vec + self.epsi * gB + self.Gamma_B

        # print('bcsp',  fL.shape, self.fL_train.shape)
        return tf.reduce_mean(tf.square(fL - self.hL)) + tf.reduce_mean(tf.square(fR - self.fR_train)) + tf.reduce_mean(
            tf.square(fT - self.fT_train)) + tf.reduce_mean(tf.square(fB - self.fB_train))

    def get_loss(self):
        pde1 = self.get_pde1()

        pde2 = self.get_pde2()

        J1 = tf.reduce_sum(tf.square(pde1)) * self.dx * self.dy

        J2 = tf.reduce_sum(self.get_intv(tf.square(pde2))) * self.dx * self.dy

        # BC rho
        J3 = self.Bd_weight * self.get_f_bc_loss()

        # print('loss shape', pde1.shape, pde2.shape, rho_vec.shape, tmp.shape, J1.numpy(), J3.numpy(), J4.numpy())

        loss = J1 + J2 + self.Bd_weight * J3
        return loss, J1, J2, J3

    def get_test_error(self):
        rho_pred, g_pred = self.nn([self.Test_xy, self.Test])

        rho_pred_vec = self.Kron_TF(rho_pred, tf.ones((self.Nv, 1)))

        # print('sf', rho_pred.shape, rho_pred_vec.shape)

        rho_pred_vec = tf.reshape(rho_pred_vec, [self.Nx * self.Ny * self.Nv, 1])

        f_pred = rho_pred_vec + self.epsi * g_pred + self.Gamma_t

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

        f_diff = f_pred - self.f_ref

        f_diff_sq = tf.square(f_diff)

        sp = tf.ones([self.Nv, 1])

        sk = tf.linspace(np.float32(0), self.Nx * self.Ny - 1, self.Nx * self.Ny, name="linspace")

        sk = tf.reshape(sk, (self.Nx * self.Ny, 1))

        id = self.Kron_TF(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.Nx * self.Ny * self.Nv])

        dup_p = tf.constant([self.Nx * self.Ny, 1], tf.int32)

        weights_rep = tf.tile(self.wt/2/np.pi, dup_p)

        e1 = tf.math.segment_sum(weights_rep * f_diff_sq, id)

        er = e1.numpy()

        test_error1 = np.sum(er[:self.Nx1*self.Ny]) * self.dxt1 * self.dyt

        test_error2 = np.sum(er[self.Nx1*self.Ny:]) * self.dxt2 * self.dyt

        test_error = test_error1 + test_error2

        return test_error

    def get_test_f(self):
        rho_pred, g_pred = self.nn([self.Test_xy, self.Test])

        rho_pred_vec = self.Kron_TF(rho_pred, tf.ones((self.Nv, 1)))

        # print('sf', rho_pred.shape, rho_pred_vec.shape)

        rho_pred_vec = tf.reshape(rho_pred_vec, [self.Nx * self.Ny * self.Nv, 1])

        f_pred = rho_pred_vec + self.epsi * g_pred + self.Gamma_t

        return f_pred

    # define functions for neural network

    def get_grad(self):
        with tf.GradientTape() as tape:
            loss, J1, J2, J3 = self.get_loss()

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
        ad_stop = self.num_ad_epochs
        for epoch in range(self.num_ad_epochs):
            loss, grad = self.get_grad()
            elapsed = time.time() - start_time
            if epoch % 100 == 0:
                self.epoch_vec.append(epoch)

                self.emp_loss_vec.append(loss)

                error2ref = self.get_test_error()

                self.test_error_vec.append(error2ref)

                print('Adam step: %d, Loss: %.3e, test error: %.3e' % (epoch, loss, error2ref))

                with open(self.file_name, 'a') as fw:
                    print('Adam step: %d, Loss: %.3e, test error: %.3e' % (epoch, loss, error2ref), file=fw)

            if epoch % 200 == 0:
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss, elapsed))

                loss, J1, J2, J3 = self.get_loss()
                print('loss 1-5', J1, J2, J3)
                with open(self.file_name, 'a') as fw:
                    print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
                    print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                          (epoch, loss, elapsed), file=fw)
                    print('loss=', loss, 'J1-J4 are ', J1, J2, J3, file=fw)
                rho, rho_pred, g, f = self.predict()
                rho_test = self.rho_test_predict()
                fig = plt.figure(1)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.yyt, rho_test.reshape(self.Nx, self.Ny).T)
                plt.title('rho')

                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.yyt, self.rho_ref.numpy())
                plt.title('rho ref')
                plt.show()

            if loss < self.stop:
                loss, J1, J2, J3 = self.get_loss()
                print('loss 1-5', loss, J1, J2, J3)
                print('training finished')
                rho_test = self.rho_test_predict()
                fig = plt.figure(1)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.yyt, rho_test.reshape(self.Nx, self.Ny).T)
                plt.title('rho')

                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.yyt, self.rho_ref.numpy())
                plt.title('rho ref')
                plt.show()
                ad_stop = epoch
                break

            self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))

        def loss_and_flat_grad(w):
            # since we are using l-bfgs, the built-in function require
            # value_and_gradients_function
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss, J1, J2, J3 = self.get_loss()

            grad = tape.gradient(loss, self.nn.trainable_variables)
            grad_flat = []
            for g in grad:
                grad_flat.append(tf.reshape(g, [-1]))
            grad_flat = tf.concat(grad_flat, 0)

            return loss, grad_flat

        step_size = 500

        cout = int(self.num_bfgs_epochs / step_size)

        for c in range(cout):
            tfp.optimizer.lbfgs_minimize(
                loss_and_flat_grad,
                initial_position=self.get_weights(),
                max_iterations=step_size,
                num_correction_pairs=10,
                tolerance=1e-6)

            loss, J1, J2, J3 = self.get_loss()

            self.epoch_vec.append(ad_stop + step_size * c)

            self.loss_vec.append(loss)

            self.emp_loss_vec.append(loss)

            error2ref = self.get_test_error()

            self.test_error_vec.append(error2ref)

            with open(self.file_name, 'a') as fw:
                print('LBFGS step: %d, Loss: %.3e, test error: %.3e' % (step_size * c, loss, error2ref), file=fw)


    def predict(self):

        rho, g = self.nn([self.Txy_set, self.Tset])

        rho_vec = self.get_rho_vec(rho)

        f = rho_vec + self.epsi*g

        rho_pred = self.get_intv(f)/2/np.pi

        return rho, rho_pred, g, f

    def save(self, model_name):
        self.nn.save(model_name)

    def rho_test_predict(self):
        f_test_pred = self.get_test_f()

        f_test_pred = f_test_pred.numpy()

        rho_test = np.zeros((self.Nx * self.Ny, 1))

        for i in range(self.Nx * self.Ny):
            rho_test[i] = 1 / 2 / np.pi * np.sum(f_test_pred[i * self.Nv:(i + 1) * self.Nv] * self.wt_ori)



        return rho_test

    def get_loss_vec(self):
        return self.epoch_vec, self.emp_loss_vec, self.test_error_vec


if __name__ == "__main__":
    ############################################ Initialization for PINN #####################################

    # input parameters

    # epsi = np.float32(sys.argv[1])
    # Bd_weight = int(sys.argv[2])
    # Nx_f = int(sys.argv[3])
    # nl = int(sys.argv[4])
    # nr = int(sys.argv[5])

    epsi = 0.001
    Bd_weight = 1
    nl = 4
    nr = 30

    # Initialize, let x \in [-1,1], y \in [-1, 1], v \in [0, 2 pi]
    lx, ly = 1, 1
    lv = 2 * np.pi

    Nx_f, Ny_f, Nv_f = 40, 40, 30

    dx = 2 * lx / Nx_f
    dy = 2 * ly / Ny_f

    x_f = np.linspace(-lx + dx, lx, Nx_f).T[:, None]
    x_p = np.linspace(-lx, lx, Nx_f).T[:, None]
    y_f = np.linspace(-ly + dy, ly - dy, Ny_f).T[:, None]

    # define quadrature points and weights
    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points = (points + 1) * np.pi
    weights = weights * np.pi
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])
    v_f = np.reshape(v_f, (Nv_f, 1))
    weights_vf = np.reshape(weights_vf, (Nv_f, 1))


    # # prepare training data for rho
    # sp = np.ones((Ny_f, 1))
    # xr = np.kron(x_f, sp)
    #
    # sk = np.ones((Nx_f, 1))
    # yr = np.kron(sk, y_f)
    #
    # Txy_set = np.concatenate([xr, yr], axis=1)
    #
    # # prepare training data for g
    # spx = np.ones((Ny_f*Nv_f, 1))
    # xt = np.kron(x_f, spx)
    #
    # spy = np.ones((Nv_f, 1))
    # y_tmp = np.kron(y_f, spy)
    # sky = np.ones((Nx_f, 1))
    # yt = np.kron(sky, y_tmp)
    #
    # skv = np.ones((Nx_f*Ny_f, 1))
    # vt = np.kron(skv, v_f)
    #
    # Tset = np.concatenate([xt, yt, vt], axis=1)

    # data preparing function for rho
    def get_xy(x, y, Nx, Ny):
        sp = np.ones((Ny, 1))
        xr = np.kron(x, sp)

        sk = np.ones((Nx, 1))
        yr = np.kron(sk, y)

        set = np.concatenate([xr, yr], axis=1)
        return set


    # data preparing function for g
    def get_xyv(x, y, v, Nx, Ny, Nv):
        spx = np.ones((Ny * Nv, 1))
        xt = np.kron(x, spx)

        spy = np.ones((Nv, 1))
        y_tmp = np.kron(y, spy)
        sky = np.ones((Nx, 1))
        yt = np.kron(sky, y_tmp)

        skv = np.ones((Nx * Ny, 1))
        vt = np.kron(skv, v)

        set = np.concatenate([xt, yt, vt], axis=1)

        return set


    Txy_set = get_xy(x_f, y_f, Nx_f, Ny_f)
    Tset = get_xyv(x_f, y_f, v_f, Nx_f, Ny_f, Nv_f)

    Tpset = get_xyv(x_p, y_f, v_f, Nx_f, Ny_f, Nv_f)
    Txy_pset = get_xy(x_p, y_f, Nx_f, Ny_f)

    # prepare B.C.
    nxbc, nybc, nvbc = 50, 50, 50


    # randomly sample theta
    # this function sample from union of two intervals
    def sample_theta(a, b, c, d, N):
        r = np.random.uniform(a - b, d - c, N)
        r += np.where(r < 0, b, c)
        r = np.reshape(r, (N, 1))
        return r


    # left bound
    # x=-1,  cos theta >0, i.e. theta \in [0, pi/2]U[3*pi/2, 2*pi]
    # In fact, we only need Train_BC_xy_L, Train_BC_L, and hy_L

    xL = -lx * np.ones((1, 1))
    yL = np.linspace(-ly, ly, nybc).T[:, None]

    a, b, c, d = 0, np.pi / 2, 3 * np.pi / 2, 2 * np.pi
    theta_BCL = sample_theta(a, b, c, d, nvbc)

    Train_BC_xy_L = get_xy(xL, yL, 1, nybc)
    Train_BC_L = get_xyv(xL, yL, theta_BCL, 1, nybc, nvbc)

    fL_v = theta_BCL
    fL_y = (1 - yL ** 2)
    fL_train = np.kron(fL_y, fL_v)

    # xx, vv = np.meshgrid(yL, theta_BCL)
    #
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(xx, vv, hy_L.numpy().reshape(nybc, nvbc).T)
    # plt.show()
    # asdfsdf

    # print('ss', Train_BC_L.shape, fL_train.shape, Train_BC_xy_L.shape)

    # right bound
    # x=1,  cos theta <0, i.e. theta \in [pi/2, 3*pi/2]
    # In fact, we only need Train_BC_xy_R, Train_BC_R, and Gamma_R
    xR = lx * np.ones((1, 1))
    yR = np.linspace(-ly, ly, nybc).T[:, None]
    theta_BCR = np.reshape(np.random.uniform(np.pi / 2, 3 * np.pi / 2, nvbc), (nvbc, 1))

    Train_BC_xy_R = get_xy(xR, yR, 1, nybc)
    Train_BC_R = get_xyv(xR, yR, theta_BCR, 1, nybc, nvbc)

    fR_train = np.zeros((nvbc * nybc, 1))

    # top bound
    # y = 1,  sin theta < 0, i.e. theta \in [pi, 2*pi]
    dxbc = 2 * lx / nxbc
    xT = np.linspace(-lx + dxbc, lx, nxbc).T[:, None]
    yT = ly * np.ones((1, 1))
    theta_BCT = np.reshape(np.random.uniform(np.pi, 2 * np.pi, nvbc), (nvbc, 1))

    Train_BC_xy_T = get_xy(xT, yT, nxbc, 1)
    Train_BC_T = get_xyv(xT, yT, theta_BCT, nxbc, 1, nvbc)

    fT_train = np.zeros((nvbc * nxbc, 1))

    # bot bound
    # y = -1,  sin theta > 0, i.e. theta \in [0, pi]
    xB = np.linspace(-lx + dxbc, lx, nxbc).T[:, None]
    yB = -ly * np.ones((1, 1))
    theta_BCB = np.reshape(np.random.uniform(0, np.pi, nvbc), (nvbc, 1))

    Train_BC_xy_B = get_xy(xB, yB, nxbc, 1)
    Train_BC_B = get_xyv(xB, yB, theta_BCB, nxbc, 1, nvbc)

    fB_train = np.zeros((nvbc * nxbc, 1))

    Train_BC_L_r, Train_BC_R_r, Train_BC_T_r, Train_BC_B_r = Train_BC_xy_L, Train_BC_xy_R, Train_BC_xy_T, Train_BC_xy_B

    # prepare Gamma
    # first we load the precompute Gamma NN
    #G_file_name = 'half_space_2d_aux_Gamma_with_lx10.h5'
    G_file_name = 'half_space_auxxxx_Gamma_with_lx10.h5'


    def my_act(x):
        return tf.nn.sigmoid(x) * np.max(3 * np.pi)


    G_mdl = load_model(G_file_name, custom_objects={"my_act": my_act})
    # get h(y) from Gamma(z,y,theta)
    lim_x = 10 * np.ones((Ny_f, 1))
    lim_v = np.pi * np.ones((Ny_f, 1))
    lim_set = np.concatenate([lim_x, y_f, lim_v], axis=1)
    hy = G_mdl(lim_set)


    # define function for computing Gamma_y
    def get_Gy(mdl, Tset):
        x, y, v = tf.convert_to_tensor(Tset[:, [0]], dtype=tf.float32), tf.convert_to_tensor(Tset[:, [1]],
                                                                                             dtype=tf.float32), tf.convert_to_tensor(
            Tset[:, [2]], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(y)
            Train = tf.concat([x, y, v], axis=1)
            Gamma = mdl(Train)

            Gamma_y = tape.gradient(Gamma, y)

        return Gamma_y


    hy_y = get_Gy(G_mdl, lim_set)
    # now extend hy and hy_y to the same size as Gamma_ori
    sph = np.ones((Nv_f, 1))
    skh = np.ones((Nx_f, 1))

    hy_vec = np.kron(skh, np.kron(hy, sph))
    hy_y_vec = np.kron(skh, np.kron(hy_y, sph))


    # plt.plot(y_f, hy, 'r-o', y_f, hy_y, 'r-*')
    # plt.show()

    # Now, we compute Gamma(x, y, v) = Gamma_ori((x+1)/epsi, y, v) - h(y)
    # Firstly, we define Gamma function
    def get_Gamma(G_mdl, x, y, v, nx, ny, nv, lx, dx, epsi):
        lx_G = 10
        if epsi > 2 * lx / lx_G:
            x_G = (x + 1) / epsi
            Gamma_set = get_xyv(x_G, y, v, nx, ny, nv)
            Gamma_ori = G_mdl(Gamma_set)
            Gamma_y_ori = get_Gy(G_mdl, Gamma_set)

            lim_x = 10 * np.ones((ny, 1))
            lim_v = np.pi * np.ones((ny, 1))
            lim_set = np.concatenate([lim_x, y, lim_v], axis=1)
            hy = G_mdl(lim_set)
            hy_y = get_Gy(G_mdl, lim_set)

            sph = np.ones((nv, 1))
            skh = np.ones((nx, 1))
            hy_vec = np.kron(skh, np.kron(hy, sph))
            hy_y_vec = np.kron(skh, np.kron(hy_y, sph))

            Gamma = Gamma_ori - hy_vec
            Gamma_y = Gamma_y_ori - hy_y_vec

        elif epsi < 2 * lx / lx_G and epsi >= dx / 10:
            n_cut = int(np.floor(lx_G * epsi / dx)) + 1
            x_G = (x[:n_cut] + 1) / epsi
            n_res = nx - n_cut

            # print('sp', n_cut, x_G.shape, x_G)
            Gamma_set = get_xyv(x_G, y, v, n_cut, ny, nv)
            Gamma_ori_1 = G_mdl(Gamma_set)
            Gamma_y_ori_1 = get_Gy(G_mdl, Gamma_set)

            lim_x = 10 * np.ones((ny, 1))
            lim_v = np.pi * np.ones((ny, 1))
            lim_set = np.concatenate([lim_x, y, lim_v], axis=1)
            hy = G_mdl(lim_set)
            hy_y = get_Gy(G_mdl, lim_set)

            sph = np.ones((nv, 1))
            skh = np.ones((n_cut, 1))
            hy_vec = np.kron(skh, np.kron(hy, sph))
            hy_y_vec = np.kron(skh, np.kron(hy_y, sph))

            Gamma_1 = Gamma_ori_1 - hy_vec
            Gamma_2 = np.zeros((n_res * ny * nv, 1))

            Gamma_y_1 = Gamma_y_ori_1 - hy_y_vec
            Gamma_y_2 = np.zeros((n_res * ny * nv, 1))

            Gamma = np.concatenate([Gamma_1, Gamma_2], axis=0)
            Gamma_y = np.concatenate([Gamma_y_1, Gamma_y_2], axis=0)
            # print('qq', n_cut, n_res)
        else:
            Gamma = np.zeros((nx * ny * nv, 1))
            Gamma_y = np.zeros((nx * ny * nv, 1))

        return Gamma, Gamma_y


    def get_Gamma_p(G_mdl, x, y, v, nx, ny, nv, lx, dx, epsi):
        lx_G = 10
        if epsi > 2 * lx / lx_G:
            x_G = (x + 1) / epsi
            Gamma_set = get_xyv(x_G, y, v, nx, ny, nv)
            Gamma_ori = G_mdl(Gamma_set)
            Gamma_y_ori = get_Gy(G_mdl, Gamma_set)

            lim_x = 10 * np.ones((ny, 1))
            lim_v = np.pi * np.ones((ny, 1))
            lim_set = np.concatenate([lim_x, y, lim_v], axis=1)
            hy = G_mdl(lim_set)
            hy_y = get_Gy(G_mdl, lim_set)

            sph = np.ones((nv, 1))
            skh = np.ones((nx, 1))
            hy_vec = np.kron(skh, np.kron(hy, sph))
            hy_y_vec = np.kron(skh, np.kron(hy_y, sph))

            Gamma = Gamma_ori - hy_vec
            Gamma_y = Gamma_y_ori - hy_y_vec

        else:
            n_cut = int(np.floor(lx_G * epsi / dx)) + 1
            x_G = (x[:n_cut] + 1) / epsi
            n_res = nx - n_cut

            # print('sp', n_cut, x_G.shape, x_G)
            Gamma_set = get_xyv(x_G, y, v, n_cut, ny, nv)
            Gamma_ori_1 = G_mdl(Gamma_set)
            Gamma_y_ori_1 = get_Gy(G_mdl, Gamma_set)

            lim_x = 10 * np.ones((ny, 1))
            lim_v = np.pi * np.ones((ny, 1))
            lim_set = np.concatenate([lim_x, y, lim_v], axis=1)
            hy = G_mdl(lim_set)
            hy_y = get_Gy(G_mdl, lim_set)

            sph = np.ones((nv, 1))
            skh = np.ones((n_cut, 1))
            hy_vec = np.kron(skh, np.kron(hy, sph))
            hy_y_vec = np.kron(skh, np.kron(hy_y, sph))

            Gamma_1 = Gamma_ori_1 - hy_vec
            Gamma_2 = np.zeros((n_res * ny * nv, 1))

            Gamma_y_1 = Gamma_y_ori_1 - hy_y_vec
            Gamma_y_2 = np.zeros((n_res * ny * nv, 1))

            Gamma = np.concatenate([Gamma_1, Gamma_2], axis=0)
            Gamma_y = np.concatenate([Gamma_y_1, Gamma_y_2], axis=0)
            # print('qq', x_G, Gamma_set)
            # sadfsa

        return Gamma, Gamma_y


    x_hL = 10 * np.ones((nvbc * nybc, 1))
    y_hL = np.kron(yL, np.ones((nvbc, 1)))
    v_hL = np.kron(np.ones((nybc, 1)), theta_BCL)

    hL_set = np.concatenate([x_hL, y_hL, v_hL], axis=1)
    hL = G_mdl(hL_set)

    dx = 2 * lx / Nx_f

    Gamma, Gamma_y = get_Gamma(G_mdl, x_f, y_f, v_f, Nx_f, Ny_f, Nv_f, lx, dx, epsi)
    Gamma_p, Gamma_y_p = get_Gamma_p(G_mdl, x_p, y_f, v_f, Nx_f, Ny_f, Nv_f, lx, dx, epsi)
    # Gamma_L, _ = get_Gamma_BC(G_mdl, -np.ones((1,1)), yL, theta_BCL, 1, nybc, nvbc, lx, dxbc, epsi)
    Gamma_R, _ = get_Gamma(G_mdl, np.ones((1, 1)), yR, theta_BCR, 1, nybc, nvbc, lx, dxbc, epsi)
    Gamma_T, _ = get_Gamma(G_mdl, xT, np.ones((1, 1)), theta_BCT, nxbc, 1, nvbc, lx, dxbc, epsi)
    Gamma_B, _ = get_Gamma(G_mdl, xB, -np.ones((1, 1)), theta_BCB, nxbc, 1, nvbc, lx, dxbc, epsi)



    # Now we define our NN

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
    # generate test set

    Nx1 = 80
    Nx2 = 80
    Nx=Nx1+Nx2

    Ny = 80
    Nv = 50

    N = Nx * Ny * Nv

    Lx = 1
    Ly = Lx
    ce = 1
    dxr1 = ce*epsi / (Nx1)
    dxr2 = (2*Lx-ce*epsi)/Nx2
    dyr = 2 * Ly / (Ny - 1)

    x1 = np.linspace(-Lx, -Lx + ce*epsi, Nx1, endpoint=False).T[:, None]
    x2 = np.linspace(-Lx+ce*epsi, Lx, Nx2).T[:, None]

    # print('q', x1, x1[-1], x2[0], x2[1])
    # dd

    x = np.concatenate([x1, x2], axis=0)


    y = np.linspace(-Ly + dyr, Ly - dyr, Ny).T[:, None]

    points, weights = np.polynomial.legendre.leggauss(Nv)
    points = (points + 1) * np.pi
    weights = weights * np.pi
    v, w = np.float32(points[:, None]), np.float32(weights[:, None])
    v = np.reshape(v, (Nv, 1))
    w = np.reshape(w, (Nv, 1))

    N1, N2, N3 = np.where(v > np.pi / 2), np.where(v > np.pi), np.where(v > 3 * np.pi / 2)
    N1, N2, N3 = N1[0][0], N2[0][0], N3[0][0]

    # define T_x
    D_xp = np.zeros((Nx, Nx))
    D_xm = np.zeros((Nx, Nx))

    for i in range(Nx1):
        D_xp[i][i] = 1 / dxr1

    for i in range(1, Nx1):
        D_xp[i][i - 1] = -1 / dxr1

    for i in range(Nx1, Nx):
        D_xp[i][i] = 1 / dxr2
        D_xp[i][i - 1] = -1 / dxr2

    for i in range(Nx1):
        D_xm[i][i] = - 1 / dxr1
        D_xm[i][i + 1] = 1 / dxr1

    for i in range(Nx1, Nx):
        D_xm[i][i] = -1 / dxr2

    for i in range(Nx1, Nx - 1):
        D_xm[i][i + 1] = 1 / dxr2

    I_y = np.eye(Ny)

    R_xm = np.kron(D_xm, I_y)
    R_xp = np.kron(D_xp, I_y)

    V_xm = np.zeros((Nv, Nv))
    V_xp = np.zeros((Nv, Nv))

    for i in range(N1, N3):
        V_xm[i, i] = np.cos(v[i]) * epsi

    for i in range(N1):
        V_xp[i, i] = np.cos(v[i]) * epsi

    for i in range(N3, Nv):
        V_xp[i, i] = np.cos(v[i]) * epsi

    Tx = sparse.kron(R_xm, V_xm) + sparse.kron(R_xp, V_xp)
    # print('tt', V_xp)
    # sdfsdf

    # define Ty
    D_yp = np.zeros((Ny, Ny))
    D_ym = np.zeros((Ny, Ny))

    for i in range(Ny):
        D_yp[i, i] = 1
        D_ym[i, i] = -1

    for i in range(Ny - 1):
        D_ym[i, i + 1] = 1
        D_yp[i + 1, i] = -1

    I_x = np.eye(Nx)

    R_ym = np.kron(I_x, D_ym)
    R_yp = np.kron(I_x, D_yp)

    V_ym = np.zeros((Nv, Nv))
    V_yp = np.zeros((Nv, Nv))

    for i in range(N2):
        V_yp[i, i] = np.sin(v[i]) * epsi / dyr

    for i in range(N2, Nv):
        V_ym[i, i] = np.sin(v[i]) * epsi / dyr

    Ty = sparse.kron(R_ym, V_ym) + sparse.kron(R_yp, V_yp)

    T = Tx + Ty


    # deine BC
    def get_bc(y, v, Lx):
        return (Lx**2 - y**2)*v
        #return (Lx ** 2 - y ** 2)


    BC = np.zeros((N, 1))

    for l in range(Ny):
        for m in range(N1):
            BC[l * Nv + m] = get_bc(y[l], v[m], Lx) * epsi * np.cos(v[m]) / dxr1

        for m in range(N3, Nv):
            BC[l * Nv + m] = get_bc(y[l], v[m], Lx) * epsi * np.cos(v[m]) / dxr1

    # define L
    w_mat = np.zeros((Nv, Nv))
    for i in range(Nv):
        w_mat[i, :] = 1 / 2 / np.pi * w.T

    L = sparse.kron(np.eye(Nx * Ny), w_mat)

    f_vec = scipy.sparse.linalg.spsolve(T - L + sparse.eye(N), BC)

    f_vec = f_vec[:,None]

    f_vec_l2 = 0

    for i in range(Nx * Ny):
        f_vec_l2 = f_vec_l2 + np.sum(np.square(f_vec[i * Nv:(i + 1) * Nv]) * w)

    f_vec_l2 = np.sum(f_vec_l2) * dxr * dyr

    print('ttt', f_vec_l2)

    zzzz

    rho_vec = np.zeros((Nx * Ny, 1))

    for i in range(Nx * Ny):
        rho_vec[i] = 1 / 2 / np.pi * np.sum(f_vec[i * Nv:(i + 1) * Nv] * w)

    rho_ref = rho_vec.reshape(Nx, Ny).T

    Test_xy = get_xy(x, y, Nx, Ny)
    Test = get_xyv(x, y, v, Nx, Ny, Nv)

    xt, yt, vt, wt, dxt1, dxt2, dyt = x, y, v, w, dxr1, dxr2, dyr

    xxt, yyt = np.meshgrid(xt, yt)

    f_ref = f_vec

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xxt, yyt, rho_ref)
    plt.title(r'$\rho(x)$ reference')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    ##################################################################################################################

    ################ tttttttttt ###################
    x_z = np.zeros((1, 1))

    # print('sp', n_cut, x_G.shape, x_G)
    Gamma_z = get_xyv(x_z, y, v, 1, Ny, Nv)
    Gamma_zz = G_mdl(Gamma_z)

    yz, vz = np.meshgrid(y, v)

    tmp = Gamma_zz.numpy()
    tp2 = f_vec[:Ny*Nv]
    tp2 = tp2.reshape(Ny, Nv)
    print('sss', tmp.shape, yz.shape, vz.shape)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(yz, vz, tmp.reshape(Ny, Nv).T)
    plt.title(r'$\rho(x)$ reference')
    plt.xlabel('x')
    plt.ylabel('y')

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(yz, vz, tp2.T)
    plt.title(r'$\rho(x)$ reference')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


    ################################# compute the corrector for test set #############################################

    x_G_1 = (x1 + 1) / epsi
    x_G_2 = x2

    # print('sp', n_cut, x_G.shape, x_G)
    Gamma_set_t = get_xyv(x_G_1, y, v, Nx1, Ny, Nv)
    Gamma_ori_t_1 = G_mdl(Gamma_set_t)

    lim_x = 10 * np.ones((Ny, 1))
    lim_v = np.pi * np.ones((Ny, 1))
    lim_set = np.concatenate([lim_x, y, lim_v], axis=1)
    hy = G_mdl(lim_set)

    sph = np.ones((Nv, 1))
    skh = np.ones((Nx1, 1))
    hy_vec = np.kron(skh, np.kron(hy, sph))

    Gamma_t_1 = Gamma_ori_t_1 - hy_vec
    Gamma_t_2 = np.zeros((Nx2 * Ny * Nv, 1))

    # print('s1', Gamma_ori_t_1.shape, hy_vec.shape)
    # asd

    Gamma_t = np.concatenate([Gamma_t_1, Gamma_t_2], axis=0)

    g_vec = np.zeros((Nx1 * Ny, 1))

    r_vec = np.zeros((Nx1 * Ny, 1))

    for i in range(Nx1 * Ny):
        g_vec[i] = 1 / 2 / np.pi * np.sum(Gamma_ori_t_1[i * Nv:(i + 1) * Nv] * w)
        r_vec[i] = 1 / 2 / np.pi * np.sum(f_vec[i * Nv:(i + 1) * Nv] * w)


    # print('s1', tmp.shape, Gamma_t[i * Nv:(i + 1) * Nv].shape, w.shape, g_vec.shape)
    # asd

    g_ref = g_vec.reshape(Nx1, Ny)

    r_ref = r_vec.reshape(Nx1, Ny)

    xx1, yy1 = np.meshgrid(x1, yt)

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx1, yy1, g_ref.T)
    plt.title(r'$\rho(x)$ reference')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.figure(2)
    plt.plot(y, g_ref[0,:], 'r', y, hy, 'b')

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xx1, yy1, r_ref.T)
    plt.title(r'$\rho(x)$ reference')
    plt.xlabel('x')
    plt.ylabel('y')


    plt.show()



    # print('sss', Gamma_t.shape, Gamma_t_2.shape, Gamma_t_1.shape)
    # 
    # sdfsdf

    ##################################################################################################################

    ############################ define and train model ##############################################################

    fname = 'rg_2D_bl_epsi_' + num2str_deciaml(epsi)

    file_name = fname + '.txt'

    dtype = tf.float32
    num_ad_epochs = 20001
    num_bfgs_epochs = 10000
    # define adam optimizer
    train_steps = 5
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    optimizer = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)

    mdl = stdst(epsi, Nx_f, Ny_f, Nv_f, x_f, x_p, y_f, v_f, lx, ly, lv, Bd_weight, Txy_set, Txy_pset, Tset,
                 Tpset, Train_BC_L, Train_BC_R, Train_BC_T, Train_BC_B, Train_BC_L_r, Train_BC_R_r, Train_BC_T_r,
                 Train_BC_B_r, fL_train, fR_train, fT_train, fB_train, Gamma_y, Gamma, Gamma_p, Gamma_R, Gamma_T,
                 Gamma_B, hL, nxbc, nybc, nvbc, weights_vf, dx, dy, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name,
                 nl, nr, Nx1, Nx2, Ny, Nv, xt, yt, vt, dxt1, dxt2, dyt, wt, Test_xy, Test, rho_ref, f_ref, Gamma_t)


    mdl.fit()
    model_name = fname + '.h5'
    mdl.save('mdls/' + model_name)

    rho_pred, rho_p_pred, g_pred, f_pred = mdl.predict()

    f_test_pred = mdl.get_test_f()

    rho_test = mdl.rho_test_predict()

    epoch_vec, emp_loss_vec, test_error_vec = mdl.get_loss_vec()

    xx, yy = np.meshgrid(x_f, y_f)

    npy_name = fname + '.npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, x_f)
        np.save(ss, rho_p_pred)
        np.save(ss, xx)
        np.save(ss, yy)
        np.save(ss, g_pred)
        np.save(ss, f_pred)
        np.save(ss, x)
        np.save(ss, y)
        np.save(ss, xxt)
        np.save(ss, yyt)
        np.save(ss, rho_test)
        np.save(ss, rho_ref)
        np.save(ss, f_test_pred)
        np.save(ss, epoch_vec)
        np.save(ss, emp_loss_vec)
        np.save(ss, test_error_vec)



