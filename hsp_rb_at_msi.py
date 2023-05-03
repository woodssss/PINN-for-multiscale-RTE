import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.special as sc
import scipy as scp
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D


class stdst():
    def __init__(self, layers, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC_L, Train_BC_Lb, fL_train, fLb_train, Train_BC_R_in, Train_BC_R_out, GM, L1_M, L2_V, nbc, weights, dx, Hv, vh, wh, dtype, optimizer, num_ad_epochs, num_bfgs_epochs):
        self.layers = layers
        self.dtype=dtype
        self.epsi, self.Nx_f, self.Nv_f = epsi, Nx_f, Nv_f
        self.lx, self.lv, self.dx = lx, lv, dx
        self.xx, self.vv = np.meshgrid(x_f, v_f)
        # convert np array to tensor
        self.x_f, self.v_f = tf.convert_to_tensor(x_f, dtype=self.dtype), tf.convert_to_tensor(v_f, dtype=self.dtype)  # x_f and v_f are grid on x and v direction
        self.x_train, self.v_train = tf.convert_to_tensor(Tset[:,[0]], dtype=self.dtype), tf.convert_to_tensor(Tset[:,[1]], dtype=self.dtype) # x_train and v_train are input trainning set for NN

        self.Tset = tf.convert_to_tensor(Tset, dtype=self.dtype)

        self.weights = tf.convert_to_tensor(weights, dtype=self.dtype)

        self.num_ad_epochs, self.num_bfgs_epochs = num_ad_epochs, num_bfgs_epochs

        self.optimizer = optimizer

        self.max = 5*np.sin(1)

        self.L1_M = tf.cast(self.convert_sparse_matrix_to_sparse_tensor(L1_M),dtype=self.dtype)
        self.L2_V = tf.convert_to_tensor(L2_V, dtype=self.dtype)


        # define BC
        self.Train_BC_L = tf.convert_to_tensor(Train_BC_L, dtype=self.dtype)
        self.Train_BC_Lb = tf.convert_to_tensor(Train_BC_Lb, dtype=self.dtype)

        self.Train_BC_R_in = tf.convert_to_tensor(Train_BC_R_in, dtype=self.dtype)
        self.Train_BC_R_out = tf.convert_to_tensor(Train_BC_R_out, dtype=self.dtype)

        self.fL_train = tf.convert_to_tensor(fL_train, dtype=self.dtype)
        self.fLb_train = tf.convert_to_tensor(fLb_train, dtype=self.dtype)

        self.Hv = tf.convert_to_tensor(Hv, dtype=self.dtype)
        self.vh, self.wh = tf.convert_to_tensor(vh, dtype=self.dtype), tf.convert_to_tensor(wh, dtype=self.dtype)

        self.GM = tf.convert_to_tensor(GM, dtype=self.dtype)

        self.x_Rb = self.lx * tf.ones((self.Nv_f,1), dtype=self.dtype)

        self.stop = 0.01


        # Initialize NN
        self.nn = tf.keras.Sequential()
        self.nn.add(tf.keras.layers.InputLayer(input_shape=(layers[0],)))
        # normalise the input variable
        self.nn.add(tf.keras.layers.Lambda(
            lambda X: (X + lv) / lv - 1.0))
        for width in layers[1:-1]:
            self.nn.add(tf.keras.layers.Dense(
                width, activation=tf.nn.tanh,
                kernel_initializer='glorot_normal'))



        def my_act(x):
            return tf.nn.sigmoid(x)*self.max



        self.nn.add(tf.keras.layers.Dense(
            layers[-1], activation=my_act,
            kernel_initializer='glorot_normal'))

        # Computing the sizes of weights/biases for future decomposition
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(layers):
            if i != 1:
                self.sizes_w.append(int(width * layers[1]))
                self.sizes_b.append(int(width if i != 0 else layers[1]))


    def convert_sparse_matrix_to_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)



    def test(self):
        f = tf.exp(-50*(self.x_train-1/2)**2) * tf.exp(-20*(self.v_train)**2)
        rho = self.get_intv(f)
        rho_vec = self.get_rho_vec(rho)
        #plt.plot(self.x_f, rho)
        #fig = plt.figure(2)
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(self.xx, self.vv, rho_vec.numpy().reshape(self.Nx_f, self.Nv_f).T)

        #plt.show()

    def test_L(self):
        f = tf.exp(-tf.pow(self.x_train, 2)) * tf.exp(-6 * tf.pow(self.v_train, 2))

        L1 = tf.sparse.sparse_dense_matmul(self.L1_M, f)
        # L1 = tf.linalg.matmul(self.L1_M, f, a_is_sparse=True)

        L2 = self.L2_V * f

        Lf = L1 - L2

        avg_Lf = self.get_intv(Lf)

        # print('tt', tf.reduce_max(avg_Lf))
        #
        # sdfgsfg

        return Lf


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

    def get_rho_vec(self, rho):

        sp=tf.ones((self.Nv_f,1))

        rho_vec = self.Kron_TF(rho, sp)

        rho_vec = tf.reshape(rho_vec, [self.Nx_f * self.Nv_f, 1])

        return rho_vec



    def get_pde(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watching the two inputs we’ll need later, x and t
            tape.watch(self.x_train)
            tape.watch(self.v_train)
            # Packing together the inputs
            Train = tf.stack([self.x_train[:, 0], self.v_train[:, 0]], axis=1)

            # Getting the prediction
            f= self.nn(Train)

            # use the first output for transportation
            f_x = tape.gradient(f, self.x_train)


        #print('s', self.L1_M.shape, f.shape)

        L1 = tf.sparse.sparse_dense_matmul(self.L1_M, f)
        #L1 = tf.linalg.matmul(self.L1_M, f, a_is_sparse=True)

        L2 = self.L2_V * f

        Lf = L1 - L2

        # print('sss', L1.shape, L2.shape)
        #
        # dfsgsdfgh

        pde = self.v_train * f_x - 1 * Lf

        return pde


    def get_pde2(self):

        f = self.nn(self.Tset)

        pde2 = self.get_intv(self.v_train*f)

        return pde2



    def test_GM(self):
        Nv = self.vh.shape[0]
        test = tf.reshape(self.vh, [Nv,1])
        tmp = tf.matmul(self.GM, test)
        R = 0.5 * tf.reshape(self.Hv, [Nv, 1]) * tmp
        # plt.plot(-self.vh, R)
        # plt.show()

    def get_bc_d_loss(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.x_Rb)
            tape.watch(self.v_f)

            Train_Rb = tf.concat([self.x_Rb, self.v_f], axis=1)

            # print('ss', self.x_Rb.shape, self.v_f.shape, Train_Rb.shape)
            #
            # asfdg

            f_Rb = self.nn(Train_Rb)

            f_Rb_x = tape.gradient(f_Rb, self.x_Rb)

            f_Rb_v = tape.gradient(f_Rb, self.v_f)

        l1 = tf.reduce_mean(tf.square(f_Rb_x))

        l2 = tf.reduce_mean(tf.square(f_Rb_v))

        return l1 + l2






    def get_bc_loss(self):
        BCL = self.nn(self.Train_BC_L) - self.fL_train
        return tf.reduce_mean(tf.square(BCL)) #+ tf.reduce_mean(tf.square(BCR))



    # define loss function
    def get_loss(self):
        # loss function contains 3 parts: PDE ( converted to IC), BC and Mass conservation
        pde = self.get_pde()
        pde2 = self.get_pde2()

        J1 = tf.reduce_sum(self.get_intv(tf.square(pde)))*self.dx

        J2 = tf.reduce_sum(tf.square(pde2))*self.dx

        # B.C
        J3 = self.get_bc_loss()

        loss = J1 + J2 + J3
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

    # need some functions for extracting weights from NN
    def get_weights(self):
        w = []
        for layer in self.nn.layers[1:]:
            weights_biases = layer.get_weights()
            weights = weights_biases[0].flatten()
            biases = weights_biases[1]
            w.extend(weights)
            w.extend(biases)
        return tf.convert_to_tensor(w, dtype=self.dtype)

    def set_weights(self, w):
        for i, layer in enumerate(self.nn.layers[1:]):
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
                loss, J1, J2, J3 = self.get_loss()
                print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                      (epoch, loss, elapsed))
                print('J1, J2', J1, J2, J3)


            if loss<self.stop:
                loss, J1, J2, J3 = self.get_loss()
                print('loss 1-5', J1, J2, J3)
                print('training finished')
                break


            self.optimizer.apply_gradients(zip(grad, self.nn.trainable_variables))

        def loss_and_flat_grad(w):
            # since we are using l-bfgs, the built-in function require
            # value_and_gradients_function
            with tf.GradientTape() as tape:
                self.set_weights(w)
                #loss = self.get_loss()
                loss, J1, J2, J3 = self.get_loss()

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
        final_loss, J1, J2, J3 = self.get_loss()
        print('Final loss is %.3e' % final_loss)


    def predict(self):
        f = self.nn(self.Tset)
        rho = self.get_intv(f) / 2
        return rho, f

    def predict_bc(self):
        f = self.nn(self.Train_BC_Lb)
        return f



    def save(self, model_name):
        self.nn.save(model_name)









if __name__ == "__main__":
    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv = 10, 1

    # layers for NN, take x,v as input variable, output f^* and f^n+1
    layers = [2, 30, 30, 30, 1]

    # define training set
    # [x_f, v_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f, Nx_f = 80, 400
    x_f = np.linspace(0 , lx , Nx_f).T[:, None]

    dx = lx/(Nx_f-1)

    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points =  lv * points
    weights = lv * weights
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])

    Tset = np.ones((Nx_f * Nv_f, 2))  # Training set, the first column is x and the second column is v

    for i in range(Nx_f):
        Tset[i * Nv_f:(i + 1) * Nv_f, [0]] = x_f[i][0] * np.ones_like(v_f)
        Tset[i * Nv_f:(i + 1) * Nv_f, [1]] = v_f



    # For BC, there are two BCs, for v>0 and for v<0
    nbc=50
    v_bc_pos= np.random.rand(1, nbc).T
    x_bc_pos, x_bc_zeros = np.ones((nbc,1)), np.zeros((nbc,1))
    # print(v_bc_pos)
    # dfgdfsgh

    def G0(v):
        return 5*np.sin(v)
        #return v

    Train_BC_L = np.float32(np.concatenate((x_bc_zeros, v_bc_pos), axis=1))

    fL_train = G0(v_bc_pos)


    # Emerging bc from chandrasker H function
    Nv = 100
    points, weights = np.polynomial.legendre.leggauss(Nv)
    vh = (points + 1) * 0.5
    wh = weights * 0.5

    def myfun(Hv, v, w):
        F = np.empty((len(v)))
        for i in range(len(v)):
            # tmp = Hv * v * w / 2 / (v[i] + v)
            F[i] = Hv[i] * np.sum(Hv * v * w / 2 / (v[i] + v)) - 1
        return F

    Hv = fsolve(myfun, np.ones_like(vh), args=(vh, wh))

    def get_BK(Hv, vh, wh):
        BK = np.zeros((len(vh)))
        for i in range(len(vh)):
            Gtmp = G0(vh)
            BK[i] = Hv[i] * np.sum(Gtmp * Hv * vh / (vh[i] + vh) * wh) * 0.5
        return BK

    BK = get_BK(Hv, vh, wh)

    x_bc_Lb, v_bc_Lb = np.zeros((Nv,1)), vh[:,None]
    Train_BC_Lb = np.float32(np.concatenate((x_bc_Lb, -v_bc_Lb), axis=1))
    fLb_train = BK[:,None]


    x_bc_R = np.ones((Nv,1))*0.5
    v_bc_pos, v_bc_neg = vh[:,None], -vh[:,None]
    Train_BC_R_in = np.float32(np.concatenate((x_bc_R, v_bc_neg), axis=1))
    Train_BC_R_out = np.float32(np.concatenate((x_bc_R, v_bc_pos), axis=1))

    GM = np.zeros((Nv,Nv))
    for i in range(Nv):
        tmp = Hv*vh*wh/(vh+vh[i])
        GM[i,:] = tmp.reshape((1,Nv))

    # plt.plot(-v_bc_Lb, fLb_train, 'r-o')
    # plt.show()
    # sfgfdsg


    # define L1_mat and L2_vec
    # since we have L(f) = int (1+vw) * (f(w)-f(v)) dw
    #  let L1(f) = int (1+vw) * f(w) dw and L2(f) = f(v) int (1+vw) dw = f(v)

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

    #print('tt', L2_vec, L2_vec.shape)
    #print('tt', L1_mat, L1_mat.shape)

    sk = np.eye(Nx_f)
    #L1_M = np.kron(sk, L1_mat)
    L1_M = scp.sparse.kron(sk, L1_mat*0.5)


    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    #L1_M_tf = convert_sparse_matrix_to_sparse_tensor(L1_M)

    sk = np.ones((Nx_f, 1))
    L2_V = np.kron(sk, L2_vec*0.5)


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


    # define parameter for model
    epsi = 1
    dtype = tf.float32
    num_ad_epochs = 10000
    num_bfgs_epochs = 5000
    # define adam optimizer

    optimizer = tf.keras.optimizers.Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08)





    # define model
    mdl = stdst(layers, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC_L, Train_BC_Lb, fL_train, fLb_train, Train_BC_R_in, Train_BC_R_out, GM, L1_M, L2_V, nbc, weights_vf, dx, Hv, vh, wh, dtype, optimizer, num_ad_epochs, num_bfgs_epochs)

    # train model
    mdl.fit()

    model_name = 'half_space_aniso_with_lx' + str(int(lx)) + '.h5'
    mdl.save('mdls/' + model_name)

    rho_pred, f_pred =mdl.predict()

    f_b = mdl.predict_bc()

    xx,vv=np.meshgrid(x_f,v_f)









