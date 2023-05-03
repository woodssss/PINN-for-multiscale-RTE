import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import sys
import scipy.special as sc


class stdst():
    def __init__(self, sigma_t, sigma_s, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Bd_weight, Tset, Train_BC_L, Train_BC_R,
                 fL_train, fR_train, nbc, weights, dx, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl,
                 nr, Nx, Nv, xt, vt, dxt, wt, Test, f_ref, f_ref_vec):
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

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        self.dxt, self.Nx, self.Nv = dxt, Nx, Nv
        self.wt = tf.convert_to_tensor(wt, dtype=self.dtype)

        self.wt_ori = wt

        self.xt, self.vt = tf.convert_to_tensor(xt, dtype=self.dtype), tf.convert_to_tensor(vt, dtype=self.dtype)
        self.Test = tf.convert_to_tensor(Test, dtype=self.dtype)

        self.f_ref = f_ref
        self.f_ref_vec = tf.convert_to_tensor(f_ref_vec, dtype=self.dtype)

        self.xxt, self.vvt = np.meshgrid(xt, vt)

        # track loss
        self.epoch_vec = []
        self.loss_vec = []

        self.emp_loss_vec = []
        self.test_error_vec = []

        # define BC
        self.Train_BC_L = tf.convert_to_tensor(Train_BC_L, dtype=self.dtype)
        self.Train_BC_R = tf.convert_to_tensor(Train_BC_R, dtype=self.dtype)

        self.fL_train = tf.convert_to_tensor(fL_train, dtype=self.dtype)
        self.fR_train = tf.convert_to_tensor(fR_train, dtype=self.dtype)

        self.sigma_s = tf.convert_to_tensor(sigma_s, dtype=self.dtype)
        self.sigma_t = tf.convert_to_tensor(sigma_t, dtype=self.dtype)

        # Initialize NN
        self.nn = self.get_nn()
        #self.nn.summary()

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

        input_rho_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                              kernel_initializer='glorot_normal')(
            input_rho)

        for i in range(self.nl-1):
            input_rho_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh,
                                                  kernel_initializer='glorot_normal')(
                input_rho_mid)

        output_rho = tf.keras.layers.Dense(units=1, activation=tf.nn.softplus, kernel_initializer='glorot_normal')(
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
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            rho, g = self.nn([self.x_f, Train])

            # use the first output for transportation
            g_x = tape.gradient(g, self.x_train)

        vg_x = self.v_train * g_x

        pde1 = self.get_intv(vg_x) * 0.5 - self.sigma_s*rho + self.sigma_t*rho

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
            rho, g = self.nn([self.x_f, Train])

            rho_x = tape.gradient(rho, self.x_f)

            g_x = tape.gradient(g, self.x_train)

        rho_x_vec = self.get_rho_vec(rho_x)

        rho_vec = self.get_rho_vec(rho)

        sigma_s_vec, sigma_t_vec = self.get_rho_vec(self.sigma_s), self.get_rho_vec(self.sigma_t)

        # print('sp1', g_x.shape, rho_x.shape, rho_x_vec.shape)

        pde2 = self.v_train * (rho_x_vec + self.epsi*g_x) + sigma_t_vec * g - sigma_s_vec*rho_vec + sigma_t_vec*rho_vec

        #print('sp2', pde2.shape, g_x.shape, sigma_s_vec.shape)

        return pde2

    def get_pde3(self):

        rho, g = self.nn([self.x_f, self.Tset])

        averg = self.get_intv(g)

        return averg


    def get_rho_vec(self, rho):

        sp = tf.ones((self.Nv_f, 1))

        rho_vec = self.Kron_TF(rho, sp)

        rho_vec = tf.reshape(rho_vec, [self.Nx_f * self.Nv_f, 1])

        return rho_vec

    def get_f_bc_loss(self):
        rhoL, gL = self.nn([tf.zeros((1, 1)), self.Train_BC_L])

        rhoR, gR = self.nn([self.lx * tf.ones((1, 1)), self.Train_BC_R])

        # print('spp', rhoL.shape, gL.shape)

        sp = tf.ones((self.nbc, 1))

        rhoL_vec = self.Kron_TF(rhoL, sp)

        rhoL_vec = tf.reshape(rhoL_vec, [self.nbc, 1])

        rhoR_vec = self.Kron_TF(rhoR, sp)

        rhoR_vec = tf.reshape(rhoR_vec, [self.nbc, 1])

        fL = rhoL_vec + self.epsi*gL
        fR = rhoR_vec + self.epsi*gR

        return tf.reduce_mean(tf.square(fL - 5 * tf.ones_like(fL))) + tf.reduce_mean(tf.square(fR))

    # define loss function
    def get_loss(self):
        # loss function contains 3 parts: PDE ( converted to IC), BC and Mass conservation
        # pde
        pde1 = self.get_pde1()

        pde2 = self.get_pde2()

        pde3 = self.get_pde3()

        J1 = tf.reduce_sum(tf.square(pde1))*self.dx

        J2 = tf.reduce_sum(self.get_intv(tf.square(pde2)))*self.dx

        J3 = tf.reduce_sum(tf.square(pde3))*self.dx

        # BC
        J4 = self.Bd_weight * self.get_f_bc_loss()

        loss = J1 + J2 + J3 + J4
        return loss, J1, J2, J3


    def get_test_error(self):
        rho_pred, g_pred = self.nn([self.xt, self.Test])

        rho_pred_vec = self.Kron_TF(rho_pred, tf.ones((self.Nv, 1)))

        #print('sf', rho_pred.shape, rho_pred_vec.shape)

        rho_pred_vec = tf.reshape(rho_pred_vec, [self.Nx * self.Nv, 1])

        f_pred = rho_pred_vec + self.epsi*g_pred

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

        f_diff = f_pred - self.f_ref_vec

        f_diff_sq = tf.square(f_diff)

        sp = tf.ones([self.Nv, 1])

        sk = tf.linspace(np.float32(0), self.Nx - 1, self.Nx, name="linspace")

        sk = tf.reshape(sk, (self.Nx, 1))

        id = self.Kron_TF(sk, sp)

        id = tf.cast(id, tf.int32)

        id = tf.reshape(id, [self.Nx * self.Nv])

        dup_p = tf.constant([self.Nx, 1], tf.int32)

        weights_rep = tf.tile(self.wt, dup_p)

        e1 = tf.math.segment_sum(weights_rep * f_diff_sq, id)

        e2 = tf.reduce_sum(e1)*self.dxt/2

        return e2

    def get_test_f(self):
        rho_pred, g_pred = self.nn([self.xt, self.Test])

        rho_pred_vec = self.Kron_TF(rho_pred, tf.ones((self.Nv, 1)))

        # print('sf', rho_pred.shape, rho_pred_vec.shape)

        rho_pred_vec = tf.reshape(rho_pred_vec, [self.Nx * self.Nv, 1])

        f_pred = rho_pred_vec + self.epsi * g_pred

        return f_pred

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
            loss, J1, J2, J3 = self.get_loss()

        return loss, tape.gradient(loss, self.nn.trainable_variables)

    # Extracting weights from NN, and use as initial weights for L-BFGS
    def get_weights(self):
        w = []
        for layer in self.nn.layers[2:]:
            weights_biases = layer.get_weights()
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
    #             loss, J1, J2, J3 = self.get_loss()
    #             print('loss 1-5', J1, J2, J3)
    #
    #             rho_pred, rho_vec_pred, g_pred = self.predict()
    #             plt.figure(1)
    #             plt.plot(self.x_f, rho_pred, 'r-o')
    #             f_pred = rho_vec_pred + self.epsi * g_pred
    #             print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
    #             fig = plt.figure(2)
    #             ax = fig.add_subplot(111, projection='3d')
    #             ax.plot_surface(self.xx, self.vv, f_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #             plt.title('f')
    #             fig = plt.figure(3)
    #             ax = fig.add_subplot(111, projection='3d')
    #             ax.plot_surface(self.xx, self.vv, g_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #             plt.title('g')
    #             plt.show(block=False)
    #             loss, J1, J2, J3 = self.get_loss()
    #             print('loss 1-5', J1, J2, J3)
    #
    #             with open(self.file_name, 'a') as fw:
    #                 print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
    #                 print('Epoch: %d, Loss: %.3e, Time: %.2f' %
    #                       (epoch, loss, elapsed), file=fw)
    #                 print('loss=', loss, 'J1-J4 are ', J1, J2, J3, file=fw)
    #
    #         if loss < self.stop:
    #             rho_pred, rho_vec_pred, g_pred = self.predict()
    #             plt.figure(1)
    #             plt.plot(self.x_f, rho_pred, 'r-o')
    #             f_pred = rho_vec_pred + self.epsi * g_pred
    #             print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
    #             fig = plt.figure(2)
    #             ax = fig.add_subplot(111, projection='3d')
    #             ax.plot_surface(self.xx, self.vv, f_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #             plt.title('f')
    #             fig = plt.figure(3)
    #             ax = fig.add_subplot(111, projection='3d')
    #             ax.plot_surface(self.xx, self.vv, g_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #             plt.title('g')
    #             plt.show(block=False)
    #             loss, J1, J2, J3 = self.get_loss()
    #             print('loss 1-5', J1, J2, J3)
    #
    #             print('training finished')
    #             break
    #
    #         self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))
    #
    #     def loss_and_flat_grad(w):
    #         # since we are using l-bfgs, the built-in function require
    #         # value_and_gradients_function
    #         with tf.GradientTape() as tape:
    #             self.set_weights(w)
    #             loss, J1, J2, J3 = self.get_loss()
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
    #     final_loss, J1, J2, J3 = self.get_loss()
    #     print('Final loss is %.3e' % final_loss)
    #
    #     rho_pred, rho_vec_pred, g_pred = self.predict()
    #     plt.figure(1)
    #     plt.plot(self.x_f, rho_pred, 'r-o')
    #     f_pred = rho_vec_pred + self.epsi * g_pred
    #     print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape)
    #     fig = plt.figure(2)
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_surface(self.xx, self.vv, f_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #     plt.title('f')
    #     fig = plt.figure(3)
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_surface(self.xx, self.vv, g_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
    #     plt.title('g')
    #     plt.show(block=False)
    #
    #     with open(self.file_name, 'a') as fw:
    #         print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
    #         print('Final loss=', final_loss, 'J1-J3 are ', J1, J2, J3, file=fw)


    def fit(self):
        start_time = time.time()
        ad_stop = self.num_ad_epochs
        for epoch in range(self.num_ad_epochs):
            loss, grad = self.get_grad()
            elapsed = time.time() - start_time

            if epoch % 50 == 0:

                self.epoch_vec.append(epoch)

                self.emp_loss_vec.append(loss)

                error2ref = self.get_test_error()

                self.test_error_vec.append(error2ref)

                print('Adam step: %d, Loss: %.3e, test error: %.3e' % (epoch, loss, error2ref))



            if epoch % 500 == 0:
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss, elapsed))
                loss, J1, J2, J3 = self.get_loss()
                print('loss 1-5', J1, J2, J3)

                rho_pred, rho_vec_pred, g_pred = self.predict()
                f_test_pred = self.get_test_f()
                f_pred = rho_vec_pred + self.epsi * g_pred
                plt.figure(1)
                plt.plot(self.x_f, rho_pred.numpy(), 'r-o')
                print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape, self.f_ref.shape, self.xxt.shape)
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, f_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('f pred')
                fig = plt.figure(3)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.vvt, self.f_ref.T)
                plt.title('f ref')

                fig = plt.figure(4)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.vvt, f_test_pred.numpy().reshape(self.Nx, self.Nv).T)
                plt.title('f ref')

                plt.figure(5)
                plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r-o', self.epoch_vec, self.test_error_vec, 'b-*')
                plt.xlabel('iteration')
                plt.ylabel('empirical loss')

                plt.show(block=False)
                loss, J1, J2, J3 = self.get_loss()


                with open(self.file_name, 'a') as fw:
                    print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
                    print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                          (epoch, loss, elapsed), file=fw)
                    print('loss=', loss, 'J1-J4 are ', J1, J2, J3, file=fw)

            if loss < self.stop:
                rho_pred, rho_vec_pred, g_pred = self.predict()
                print('ss', rho_pred.shape, rho_vec_pred.shape, g_pred.shape, self.x_f.shape)
                plt.figure(1)
                plt.plot(self.x_f, rho_pred.numpy(), 'r-o')
                f_pred = rho_vec_pred + self.epsi * g_pred
                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xx, self.vv, f_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
                plt.title('f')
                fig = plt.figure(3)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(self.xxt, self.vvt, self.f_ref.T)
                plt.title('f ref')
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
                loss, J1, J2, J3 = self.get_loss()

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

            loss, J1, J2, J3 = self.get_loss()

            self.epoch_vec.append(ad_stop+step_size*c)

            self.loss_vec.append(loss)

            self.emp_loss_vec.append(loss)

            error2ref = self.get_test_error()

            self.test_error_vec.append(error2ref)

            rho_pred, rho_vec_pred, g_pred = self.predict()
            f_test_pred = self.get_test_f()
            f_pred = rho_vec_pred + self.epsi * g_pred
            plt.figure(1)
            plt.plot(self.x_f, rho_pred.numpy(), 'r-o')
            print('LBFGS step: %d, Loss: %.3e, test error: %.3e' % (step_size*c, loss, error2ref))
            fig = plt.figure(2)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.xx, self.vv, f_pred.numpy().reshape(self.Nx_f, self.Nv_f).T)
            plt.title('f')
            fig = plt.figure(3)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.xxt, self.vvt, self.f_ref.T)
            plt.title('f ref')

            fig = plt.figure(4)
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(self.xxt, self.vvt, f_test_pred.numpy().reshape(self.Nx, self.Nv).T)
            plt.title('f ref')

            plt.figure(5)
            plt.semilogy(self.epoch_vec, self.emp_loss_vec, 'r-o', self.epoch_vec, self.test_error_vec, 'b-*')
            plt.xlabel('iteration')
            plt.ylabel('empirical loss')

            plt.show(block=False)


        final_loss, J1, J2, J3 = self.get_loss()
        print('Final loss is %.3e' % final_loss)

        with open(self.file_name, 'a') as fw:
            print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
            print('Final loss=', final_loss, 'J1-J3 are ', J1, J2, J3, file=fw)

    def predict(self):
        rho, g = self.nn([self.x_f, self.Tset])

        rho_vec = self.get_rho_vec(rho)

        f = rho_vec + self.epsi*g

        rho_real = self.get_intv(f)*0.5

        return rho_real, rho_vec, g

    def rho_test_predict(self):
        f_test_pred = self.get_test_f()

        f_test_pred = f_test_pred.numpy()
        f_test = f_test_pred.reshape(self.Nx, self.Nv)
        rho_test = np.zeros((self.Nx, 1))
        for i in range(self.Nx):
            tmp = np.sum(f_test[[i], :] * self.wt_ori.T) / 2
            rho_test[i] = tmp

        return rho_test

    def save(self, model_name):
        self.nn.save(model_name)

    def get_loss_vec(self):
        return self.epoch_vec, self.emp_loss_vec, self.test_error_vec


if __name__ == "__main__":

    ############################################ Initialization for PINN #####################################
    # input parameters

    # epsi= np.float32(sys.argv[1])
    # Bd_weight = int(sys.argv[2])
    # Nx_f = int(sys.argv[3])
    # nl = int(sys.argv[4])
    # nr = int(sys.argv[5])

    epsi = 1
    Bd_weight = 1
    Nx_f = 80
    nl = 4
    nr = 50

    cs = 1

    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv = 2, 1

    # define training set
    # [x_f, v_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f = 60

    dx = lx/(Nx_f+1)

    x_f = np.linspace(0, lx, Nx_f).T[:, None]

    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points = lv * points
    weights = lv * weights
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])

    Tset = np.ones((Nx_f * Nv_f, 2))  # Training set, the first column is x and the second column is v

    for i in range(Nx_f):
        Tset[i * Nv_f:(i + 1) * Nv_f, [0]] = x_f[i][0] * np.ones_like(v_f)
        Tset[i * Nv_f:(i + 1) * Nv_f, [1]] = v_f

    # define sigma function
    sigma_t = 2*np.ones((Nx_f, 1))
    sigma_s = 2*np.ones((Nx_f, 1))

    for i in range(int(Nx_f/2)):
        sigma_t[i] = 2
        sigma_s[i] = cs


    plt.plot(x_f, sigma_s,'r-o', x_f, sigma_t, 'b-*')
    plt.show()

    # For BC, there are two BCs, for v>0 and for v<0
    nbc = 60
    v_bc_pos, v_bc_neg = np.random.rand(1, nbc).T, -np.random.rand(1, nbc).T
    # v_bc_pos, v_bc_neg = np.linspace(0, lv, nbc).T[:, None], np.linspace(-lv, 0, nbc).T[:, None]
    x_bc_pos, x_bc_zeros = lx * np.ones((nbc, 1)), np.zeros((nbc, 1))

    Train_BC_L = np.float32(np.concatenate((x_bc_zeros, v_bc_pos), axis=1))
    Train_BC_R = np.float32(np.concatenate((x_bc_pos, v_bc_neg), axis=1))

    #fL_train = np.float32(5 * np.sin(v_bc_pos))
    fL_train = 5 * np.ones_like(v_bc_pos)

    fR_train = 0

    plt.plot(v_bc_pos, fL_train, 'ro')
    plt.show()

    # print('bc', Train_BC_L, Train_BC_R)

    # define parameter for model
    dtype = tf.float32
    num_ad_epochs = 12000
    num_bfgs_epochs = 10000
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
    # generate test set

    Nx = 200
    Nv = 80

    lx, lv = 2, 1

    dxr = lx / (Nx - 1)

    x = np.linspace(dxr, lx, Nx).T[:, None]

    points, weights = np.polynomial.legendre.leggauss(Nv)
    points = lv * points
    weights = lv * weights
    v, w = np.float32(points[:, None]), np.float32(weights[:, None])

    Test_x = np.kron(x, np.ones((Nv, 1)))
    Test_v = np.kron(np.ones((Nx, 1)), v)

    Test = np.concatenate([Test_x, Test_v], axis=1)

    Dp, Dm = np.zeros((Nx, Nx)), np.zeros((Nx, Nx))

    Vp, Vm = np.zeros((Nv, Nv)), np.zeros((Nv, Nv))

    for i in range(Nx):
        Dp[i][i] = 1
        Dm[i][i] = -1

    for i in range(Nx - 1):
        Dm[i][i + 1] = 1
        Dp[i + 1][i] = -1

    for i in range(int(Nv / 2)):
        Vp[i + int(Nv / 2)][i + int(Nv / 2)] = v[i + int(Nv / 2)]
        Vm[i][i] = v[i]


    Tp = np.kron(Dp, Vp)
    Tm = np.kron(Dm, Vm)

    T = (Tp + Tm) * epsi / dxr

    sigma_t_r = 2*np.eye(Nx)
    sigma_s_r = 2*np.eye(Nx)

    for i in range(int(Nx / 2)):
        sigma_t_r[i, i] = 2
        sigma_s_r[i, i] = cs

    sigma_s_mat = np.kron(sigma_s_r, np.eye(Nv))
    sigma_t_mat = np.kron(sigma_t_r, np.eye(Nv))

    sk = np.ones((Nv, 1)) / 2
    w_mat = np.kron(sk, w.T)

    L = np.matmul(sigma_s_mat, np.kron(np.eye(Nx), w_mat)) - np.matmul(sigma_t_mat, np.eye(Nx * Nv))

    BC = np.zeros((Nx * Nv, 1))
    ct = 0
    for i in range(int(Nv / 2)):
        # BC[i + int(Nv/2)] = epsi*v[i+int(Nv/2)]/dx*5*np.sin(v[i+int(Nv/2)])
        BC[i + int(Nv / 2)] = epsi * v[i + int(Nv / 2)] / dxr * 5

    f_ref_vec = np.linalg.solve(T - L, BC)

    f_ref = f_ref_vec.reshape(Nx, Nv)

    rho_ref = np.zeros((Nx, 1))
    for i in range(Nx):
        tmp = np.sum(f_ref[[i], :] * weights.T) / 2
        rho_ref[i] = tmp

    xt, vt, wt, dxt = x, v, w, dxr

    xxt, vvt = np.meshgrid(xt, vt)

    ##################################################################################################################

    ############################ define and train model ##############################################################


    fname = 'rg_sigma_epsi' + num2str_deciaml(epsi) + '_bd_' + str(Bd_weight) + '_cs_' + num2str_deciaml(cs) + '_nl_' + str(
        nl) + '_nr_' + str(nr)

    file_name = fname+'.txt'

    # define model
    mdl = stdst(sigma_t, sigma_s, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Bd_weight, Tset, Train_BC_L, Train_BC_R, fL_train,
                fR_train, nbc, weights_vf, dx, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl, nr, Nx, Nv, xt, vt, dxt, wt, Test, f_ref, f_ref_vec)

    # train model
    mdl.fit()

    model_name = fname + '.h5'
    mdl.save('mdls/' + model_name)

    rho_pred, rho_vec_pred, g_pred = mdl.predict()

    rho_pred = rho_pred.numpy()

    rho_vec_pred = rho_vec_pred.numpy().reshape(Nx_f, Nv_f)

    g_pred = g_pred.numpy().reshape(Nx_f, Nv_f)

    f_pred = rho_vec_pred + epsi * g_pred

    xx, vv = np.meshgrid(x_f, v_f)

    f_test_pred = mdl.get_test_f()

    rho_test = mdl.rho_test_predict()

    epoch_vec, emp_loss_vec, test_error_vec = mdl.get_loss_vec()

    npy_name = fname + '.npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, x_f)
        np.save(ss, rho_pred)
        np.save(ss, x)
        np.save(ss, rho_ref)
        np.save(ss, rho_test)
        np.save(ss, xx)
        np.save(ss, vv)
        np.save(ss, f_pred)
        np.save(ss, g_pred)
        np.save(ss, xxt)
        np.save(ss, vvt)
        np.save(ss, f_ref)
        np.save(ss, f_test_pred)
        np.save(ss, epoch_vec)
        np.save(ss, emp_loss_vec)
        np.save(ss, test_error_vec)








