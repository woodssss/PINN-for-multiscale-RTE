import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import scipy.special as sc
from scipy.optimize import fsolve
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing
import os


# def train_process(i):
#     y_temp = y[i]
#     f_BC_train_tmp = (1 - y_temp ** 2) * theta_BC
#     mdl_tmp = stdst(layers, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC, f_BC_train_tmp, nbc, weights_vf, dtype,
#                     optimizer, num_ad_epochs, num_bfgs_epochs)
#
#     mdl_tmp.fit()
#
#     model_name_tmp = 'half_space_aux_with_lx' + str(int(lx)) + '_i_' + str(int(i)) + '.h5'
#     mdl.save('mdls/' + model_name_tmp)


def train_process(i, y_i, layers, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC, theta_BC, nbc, weights_vf, dx, dtype,
                  optimizer, num_ad_epochs, num_bfgs_epochs):
    print('parent process:', os.getppid())
    print('process id:', os.getpid(), "\n\n")
    y_temp = y_i
    f_BC_train_tmp = (1 - y_temp ** 2) * theta_BC

    file_name_tmp = 'half_space_aux_with_lx' + str(int(lx)) + '_Ny_' + str(int(Ny)) + '_i_' + str(int(i)) + '.txt'

    mdl_tmp = stdst(layers, i, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC, f_BC_train_tmp, nbc, weights_vf, dx, dtype,
                    optimizer, num_ad_epochs, num_bfgs_epochs, file_name_tmp)

    mdl_tmp.fit()

    model_name_tmp = 'half_space_aux_with_lx' + str(int(lx)) + '_Ny_' + str(int(Ny)) + '_i_' + str(int(i)) + '.h5'
    mdl_tmp.save('mdls/' + model_name_tmp)

class stdst():
    # this is for linear transport equation epsi * \partial_t f + v \partial_x f = 1/epsi * L(f)
    # L(f) = 1/2 \int_-1^1 f  - f
    # the expect limit system is \partial_t rho - 1/3 \partial_xx rho = 0
    def __init__(self, layers, i, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC, f_BC_train, nbc, weights, dx, dtype, optimizer, num_ad_epochs, num_bfgs_epochs, file_name):
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

        self.max = 3*np.pi

        self.stop = 0.01

        self.file_name = file_name

        self.i = i

        # define BC

        self.Train_BC = tf.convert_to_tensor(Train_BC, dtype=self.dtype)
        self.f_BC_train = tf.convert_to_tensor(f_BC_train, dtype=self.dtype)
        self.nbc= tf.convert_to_tensor(nbc, dtype=self.dtype)

        #self.BCR = tf.concat([self.lx*tf.ones((self.Nv_f, 1)), self.v_f])

        self.x_Rb = self.lx*tf.ones((self.Nv_f, 1))

        #self.test()
        #self.test_GM()


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



    def test(self):
        f = tf.exp(-50*(self.x_train-1/2)**2) * tf.exp(-20*(self.v_train)**2)
        rho = self.get_intv(f)
        rho_vec = self.get_rho_vec(rho)
        #plt.plot(self.x_f, rho)



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

        rho = self.get_intv(f)

        rho_vec = self.get_rho_vec(rho)

        Lf = 1/2/np.pi * rho_vec - f

        vx = tf.cos(self.v_train)

        pde = vx * f_x - 1 * Lf

        return pde


    def get_pde2(self):

        f = self.nn(self.Tset)

        vx = tf.cos(self.v_train)

        pde2 = self.get_intv(vx*f)

        return pde2



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
        BCL = self.nn(self.Train_BC) - self.f_BC_train
        return tf.reduce_mean(tf.square(BCL)) #+ tf.reduce_mean(tf.square(BCR))


    # define loss function
    def get_loss(self):
        # loss function contains 3 parts: PDE ( converted to IC), BC and Mass conservation
        pde = self.get_pde()
        pde2 = self.get_pde2()
        J11 = tf.reduce_sum(self.get_intv(tf.square(pde)))*self.dx
        J12 = tf.reduce_sum(tf.square(pde2))*self.dx
        J1 = J11 + J12

        # B.C
        J2 = self.get_bc_loss()

        J3 = self.get_bc_d_loss()

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
                with open(self.file_name, 'a') as fw:
                    print('This is y i=', self.i, file=fw)
                    print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                          (epoch, loss, elapsed), file=fw)
                    print('loss=', loss, 'J1-J4 are ', J1, J2, J3, file=fw)
                print('loss and J', loss, J1, J2, J3)
            if loss<self.stop:
                print('training finished')
                loss, J1, J2, J3= self.get_loss()
                print('loss 1-5', loss, J1, J2, J3)
                with open(self.file_name, 'a') as fw:
                    print('This is y i=', self.i, file=fw)
                    print('Epoch: %d, Loss: %.3e, Time: %.2f' %
                          (epoch, loss, elapsed), file=fw)
                    print('loss=', loss, 'J1-J4 are ', J1, J2, J3, file=fw)
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

        loss, J1, J2, J3 = self.get_loss()
        with open(self.file_name, 'a') as fw:
            print('This is y i=', self.i, file=fw)
            print('Final loss=', loss, 'J1-J4 are ', J1, J2, J3, file=fw)


    def predict(self):
        f = self.nn(self.Tset)
        rho = self.get_intv(f) / 2 / np.pi
        return rho, f

    def predict_bc(self):
        f = self.nn(self.Train_BC)
        return f



    def save(self, model_name):
        self.nn.save(model_name)









if __name__ == "__main__":
    # Initialize, let x \in [0,1], v \in [-1, 1]
    lx, lv = 10, 1

    # layers for NN, take x,v as input variable, output f^* and f^n+1
    layers = [2, 30, 30, 30, 1]

    # define training set
    # the auxiliary problem is a 1-d hsp
    # [x_f, theta_f] for pde, since we need to evaluate integral, here we use quadrature rule
    Nv_f, Nx_f = 30, 200
    x_f = np.linspace(0 , lx , Nx_f).T[:, None]
    dx = lx/(Nx_f-1)

    points, weights = np.polynomial.legendre.leggauss(Nv_f)
    points = (points + 1) * np.pi
    weights = weights * np.pi
    v_f, weights_vf = np.float32(points[:, None]), np.float32(weights[:, None])
    v_f = np.reshape(v_f, (Nv_f, 1))
    weights_vf = np.reshape(weights_vf, (Nv_f, 1))

    Tset = np.ones((Nx_f * Nv_f, 2))  # Training set, the first column is x and the second column is v

    for i in range(Nx_f):
        Tset[i * Nv_f:(i + 1) * Nv_f, [0]] = x_f[i][0] * np.ones_like(v_f)
        Tset[i * Nv_f:(i + 1) * Nv_f, [1]] = v_f



    # For BC, we consider the inflow B.C. at z=0, i.e. \cos \theta >0
    # we need z=0 and \theta
    nbc=100

    def sample_theta(a,b,c,d,N):
        r = np.random.uniform(a - b, d - c, N)
        r += np.where(r < 0, b, c)
        r = np.reshape(r, (N, 1))
        return r


    # Since we consider inflow boundary condition, at BL, we require v_x>0, which is same
    # as cos theta >0, i.e. theta \in [0, pi/2]U[3*pi/2, 2*pi]
    BCz = np.zeros((nbc,1))

    a, b, c, d = 0, np.pi/2, 3*np.pi/2, 2*np.pi
    theta_BC = sample_theta(a,b,c,d,nbc)

    Train_BC = np.float32(np.concatenate((BCz, theta_BC), axis=1))

    y_temp = 0
    f_BC_train = (1 - y_temp**2)*theta_BC


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


    # let Ny be number of grid points on y
    Ny = 100
    ly = 1
    y = np.linspace(-ly, ly, Ny).T[:, None]

    i_list = np.arange(Ny)

    # define process
    # def train_process(y_i, layers, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC, theta_BC, nbc, weights_vf, dtype, optimizer, num_ad_epochs, num_bfgs_epochs):
    #     y_temp = y_i
    #     f_BC_train_tmp = (1 - y_temp**2)*theta_BC
    #     mdl_tmp = stdst(layers, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC, f_BC_train_tmp, nbc, weights_vf, dtype, optimizer, num_ad_epochs, num_bfgs_epochs)
    #
    #     mdl_tmp.fit()
    #
    #     model_name_tmp = 'half_space_aux_with_lx' + str(int(lx)) + '_yi_' + num2str_deciaml(y_i) + '.h5'
    #     mdl.save('mdls/' + model_name_tmp)

    print(f'starting computations on {multiprocessing.cpu_count()} cores')

    my_process = []
    #
    for i in range(80, Ny):
        y_i = y[i]
        p = multiprocessing.Process(target=train_process, args=(i, y_i, layers, epsi, Nx_f, Nv_f, x_f, v_f, lx, lv, Tset, Train_BC, theta_BC, nbc, weights_vf, dx, dtype, optimizer, num_ad_epochs, num_bfgs_epochs))
        my_process.append(p)
        p.start()
        print('process ', i)
    #
    for process in my_process:
        process.join()

    print('process done')
    # with multiprocessing.Pool() as pool:
    #     pool.map(train_process, i_list)
        #pool.map(lambda param: mdl)
