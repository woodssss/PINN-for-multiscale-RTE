import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy as scp
import time
import sys
import scipy.special as sc
#from tensorflow.keras.models import load_model
from keras.models import load_model

# This is for rho g decomposition with solution to the half space problem as corrector.

class stdst():
    # this is for linear transport equation epsi * \partial_t f + v \partial_x f = 1/epsi * L(f)
    # L(f) = 1/2 \int_-1^1 f  - f
    # the expect limit system is \partial_t rho - 1/3 \partial_xx rho = 0
    def __init__(self, epsi, Nx_f, Nv_f, x_f, x_p, v_f, lx, lv, Bd_weight, Tset, Test, Train_BC_L, Train_BC_R, fL_train, fR_train, L1_M, L2_V, nbc, weights, dx, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl, nr):
        self.dtype=dtype
        self.epsi, self.Nx_f, self.Nv_f = epsi, Nx_f, Nv_f
        self.lx, self.lv, self.dx = lx, lv, dx
        self.Bd_weight = Bd_weight
        self.xx, self.vv = np.meshgrid(x_f, v_f)
        self.nbc=nbc

        # number of layers for rho and g
        self.nl, self.nr = nl, nr

        self.stop = 0.01

        self.file_name = file_name
        # convert np array to tensor
        self.x_f, self.v_f = tf.convert_to_tensor(x_f, dtype=self.dtype), tf.convert_to_tensor(v_f, dtype=self.dtype)  # x_f and v_f are grid on x and v direction
        self.x_train, self.v_train = tf.convert_to_tensor(Tset[:,[0]], dtype=self.dtype), tf.convert_to_tensor(Tset[:,[1]], dtype=self.dtype) # x_train and v_train are input trainning set for NN
        self.x_p = tf.convert_to_tensor(x_p, dtype=self.dtype)

        self.Tset = tf.convert_to_tensor(Tset, dtype=self.dtype)
        self.Test = tf.convert_to_tensor(Test, dtype=self.dtype)

        self.weights = tf.convert_to_tensor(weights, dtype=self.dtype)

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        # define BC
        self.Train_BC_L = tf.convert_to_tensor(Train_BC_L, dtype=self.dtype)
        self.Train_BC_R = tf.convert_to_tensor(Train_BC_R, dtype=self.dtype)

        self.fL_train = tf.convert_to_tensor(fL_train, dtype=self.dtype)
        self.fR_train = tf.convert_to_tensor(fR_train, dtype=self.dtype)

        self.L1_M = tf.cast(self.convert_sparse_matrix_to_sparse_tensor(L1_M), dtype=self.dtype)
        self.L2_V = tf.convert_to_tensor(L2_V, dtype=self.dtype)

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
            #print('ssssss', int(weights.shape[0]), int(biases.shape[0]))
            self.sizes_w.append(int(weights.shape[0]))
            self.sizes_b.append(int(biases.shape[0]))


    def convert_sparse_matrix_to_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)



    def get_nn(self):
        # define nn for rho
        input_rho = tf.keras.Input(shape=(1,))

        input_rho_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
            input_rho)

        for i in range(self.nl-1):
            input_rho_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
                input_rho_mid)

        output_rho = tf.keras.layers.Dense(units=1, activation=None, kernel_initializer='glorot_normal')(
            input_rho_mid)

        # define nn for g

        input_g = tf.keras.Input(shape=(2,))

        input_g_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
            input_g)

        for i in range(self.nl-1):
            input_g_mid = tf.keras.layers.Dense(units=self.nr, activation=tf.nn.tanh, kernel_initializer='glorot_normal')(
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

        #print('ssp', weights_rep.shape, res.shape)

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

        vg_x = self.v_train*g_x

        pde1 = self.get_intv(vg_x)

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

            g_x = tape.gradient(g,self.x_train)

        rho_x_vec = self.get_rho_vec(rho_x)

        L1g = tf.sparse.sparse_dense_matmul(self.L1_M, g)

        L2g = self.L2_V * g

        Lg = L1g - L2g

        pde2 = self.v_train*(rho_x_vec+self.epsi*g_x) -Lg

        return pde2

    def get_pde3(self):

        rho, g = self.nn([self.x_f, self.Tset])

        return self.get_intv(g)



    def get_rho_vec(self, rho):

        sp=tf.ones((self.Nv_f,1))

        rho_vec = self.Kron_TF(rho, sp)

        rho_vec = tf.reshape(rho_vec, [self.Nx_f * self.Nv_f, 1])

        return rho_vec



    def get_f_bc_loss(self):
        rhoL, gL = self.nn([tf.zeros((1,1)), self.Train_BC_L])

        rhoR, gR = self.nn([tf.ones((1,1)), self.Train_BC_R])

        #print('spp', rhoL.shape, gL.shape)

        sp = tf.ones((self.nbc, 1))

        rhoL_vec = self.Kron_TF(rhoL, sp)

        rhoL_vec = tf.reshape(rhoL_vec, [self.nbc, 1])

        rhoR_vec = self.Kron_TF(rhoR, sp)

        rhoR_vec = tf.reshape(rhoR_vec, [self.nbc, 1])

        fL = rhoL_vec + self.epsi*gL
        fR = rhoR_vec + self.epsi*gR

        return tf.reduce_mean(tf.square(fL - self.fL_train)) + tf.reduce_mean(tf.square(fR + self.fR_train))


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

        # BC rho
        J4 = self.Bd_weight*self.get_f_bc_loss()

        loss = J1 + J2 + J3 + J4
        return loss, J1, J2, J3

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
        #print('wsp',len(self.nn.trainable_variables), tf.shape_n(self.nn.trainable_variables))
        self.nn.summary()
        for layer in self.nn.layers[2:]:
            weights_biases = layer.get_weights()
            #print('wbsp', len(weights_biases), weights_biases)
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

            if loss<self.stop:
                loss, J1, J2, J3 = self.get_loss()
                print('loss 1-5', loss, J1, J2, J3)
                print('training finished')
                break

            self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))

        def loss_and_flat_grad(w):
            # since we are using l-bfgs, the built-in function require
            # value_and_gradients_function
            with tf.GradientTape() as tape:
                self.set_weights(w)
                loss , J1, J2, J3= self.get_loss()

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
            tolerance=1e-6)


        final_loss , J1, J2, J3= self.get_loss()
        print('Final loss is %.3e' % final_loss)

        with open(self.file_name, 'a') as fw:
            print('This is epsi=', self.epsi, ' Nx=', self.Nx_f, file=fw)
            print('Final loss=', final_loss, 'J1-J3 are ', J1, J2, J3, file=fw)



    def predict(self):
        rho, g = self.nn([self.x_p, self.Test])

        rho_vec = self.get_rho_vec(rho)

        return rho, rho_vec, g

    def rho_predict(self):

        rho, g = self.nn([self.x_p, self.Test])

        rho_vec = self.get_rho_vec(rho)

        f = rho_vec + self.epsi*g

        rho_real = self.get_intv(f) * 0.5

        return rho_real


    def save(self, model_name):
        self.nn.save(model_name)




if __name__ == "__main__":
    # input parameters

    epsi= np.float32(sys.argv[1])
    Bd_weight = int(sys.argv[2])
    Nx_f = int(sys.argv[3])
    nl = int(sys.argv[4])
    nr = int(sys.argv[5])

    # epsi = 0.001
    # Bd_weight = 1
    # Nx_f = 60
    # nl = 3
    # nr = 30

    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv= 1, 1

    # define training set
    # [x_f, v_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f= 60

    dx = 1 / (Nx_f - 1)
    x_f = np.linspace(dx, lx, Nx_f).T[:, None]
    x_p = np.linspace(0, lx, Nx_f).T[:, None]

    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points =  lv * points
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
    # since we have f = rho + epsi g + Gamma(x/epsi, v)
    # we need to load a pretrained Gamma NN.
    nbc=60
    v_bc_pos, v_bc_neg = np.random.rand(1, nbc).T, -np.random.rand(1, nbc).T
    x_bc_pos, x_bc_zeros = np.ones((nbc,1)), np.zeros((nbc,1))

    G_x_bc_pos = 10 * np.ones_like(v_bc_neg)

    Train_BC_L = np.float32(np.concatenate((x_bc_zeros, v_bc_pos), axis=1))
    Train_BC_R = np.float32(np.concatenate((x_bc_pos, v_bc_neg), axis=1))

    BC_R_Gamma = np.float32(np.concatenate((G_x_bc_pos, v_bc_neg), axis=1))

    # Load pretrained Gamma NN
    #G_file_name = 'half_space_with_lx10.h5'
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



    fL_train = np.ones_like(v_bc_pos)
    fR_train =  np.zeros_like(v_bc_neg)


    xx, vv = np.meshgrid(x_f, v_f)

    # define parameter for model
    dtype = tf.float32
    num_ad_epochs = 10000
    num_bfgs_epochs = 5000
    # define adam optimizer
    train_steps = 5
    lr_fn = tf.optimizers.schedules.PolynomialDecay(1e-3, train_steps, 1e-5, 2)
    optimizer = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)

    # define L1_mat and L2_vec
    # since we have L(f) = int (1+g)/(1 + g^2 - vw) * (f(w)-f(v)) dw
    #  let L1(f) = int (1+g)/(1 + g^2 - vw) * f(w) dw and L2(f) = f(v) int (1+g)/(1 + g^2 - vw) dw

    def L1_val(v, w, weight):
        return (1 + v * w) * weight


    L1_mat = np.zeros((Nv_f, Nv_f))
    for i in range(Nv_f):
        for j in range(Nv_f):
            L1_mat[i, j] = L1_val(v_f[i], v_f[j], weights_vf[j])

    L2_vec = np.zeros((Nv_f, 1))

    for i in range(Nv_f):
        tmp = 1 + v_f[i] * v_f
        L2_vec[i] = np.sum(tmp * weights_vf)

    # print('tt', L2_vec, L2_vec.shape)
    # print('tt', L1_mat, L1_mat.shape)

    sk = np.eye(Nx_f)
    # L1_M = np.kron(sk, L1_mat)
    L1_M = scp.sparse.kron(sk, L1_mat*0.5)


    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    sk = np.ones((Nx_f, 1))
    L2_V = np.kron(sk, L2_vec*0.5)

    # save model
    file_name = 'rg_aniso_epsi_' + num2str_deciaml(epsi) + '_Nx_' + str(Nx_f) + '_nl_' + str(
        nl) + '_nr_' + str(nr) +'.' + 'txt'

    # define model
    mdl = stdst(epsi, Nx_f, Nv_f, x_f, x_p, v_f, lx, lv, Bd_weight, Tset, Test, Train_BC_L, Train_BC_R, fL_train,
                fR_train, L1_M, L2_V, nbc, weights_vf, dx, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name, nl, nr)

    # train model
    mdl.fit()

    model_name = 'rg_aniso_epsi_' + num2str_deciaml(epsi) + '_Nx_' + str(Nx_f) + '_nl_' + str(
        nl) + '_nr_' + str(nr) +'.' + 'h5'
    mdl.save('mdls/' + model_name)

    rho_pred, rho_vec_pred, g_pred =mdl.predict()

    rho_real_pred = mdl.rho_predict()

    rho_vec_pred = rho_vec_pred.numpy().reshape(Nx_f,Nv_f)

    g_pred = g_pred.numpy().reshape(Nx_f,Nv_f)

    f_pred = rho_vec_pred + epsi* g_pred

    npy_name = 'rg_aniso_epsi_' + num2str_deciaml(epsi) + '_Nx_' + str(Nx_f) + '_nl_' + str(
        nl) + '_nr_' + str(nr) + '.' + 'npy'

    with open(npy_name, 'wb') as ss:
        np.save(ss, x_f)
        np.save(ss, rho_real_pred)
        np.save(ss, xx)
        np.save(ss, vv)
        np.save(ss, f_pred)
        np.save(ss, g_pred)

